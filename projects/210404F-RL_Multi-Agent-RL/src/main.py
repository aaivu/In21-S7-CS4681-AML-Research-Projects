# main.py
# Examples:
# Baseline:
#   python main.py --updates 500 --horizon 25 --epochs 10 --clip_eps 0.1
# Entropy schedule:
#   python main.py --ent_schedule linear --ent_coef_start 0.03 --ent_coef_end 0.006 --ent_decay_steps 400
# Auto-tuned entropy (target):
#   python main.py --ent_schedule none --ent_coef_start 0.03 --ent_target 1.1 --ent_adapt_lr 1e-3
# Recurrent MAPPO (GRU) + schedule:
#   python main.py --recurrent --chunk_len 10 --horizon 100 --epochs 6 --ent_schedule linear --ent_coef_start 0.03 --ent_coef_end 0.008 --ent_decay_steps 400

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from pettingzoo.mpe import simple_spread_v3

from util import (
    set_seed, build_agent_specific_states, ValueNorm,
    whiten, compute_gae_returns, iter_chunks
)
from model import Actor, CentralCritic
from sampler import ParallelRolloutSampler


def make_env(seed: int, n_agents: int = 3):
    env = simple_spread_v3.parallel_env(
        continuous_actions=False, local_ratio=0.5, N=n_agents
    )
    env.reset(seed=seed)
    return env


def get_scheduled_ent(step, *, kind, start, end, decay_steps):
    """Entropy schedule: const/none, linear, cosine, exp"""
    if kind in ("const", "none"):
        return start
    t = min(step / max(decay_steps, 1), 1.0)
    if kind == "linear":
        return start + (end - start) * t
    if kind == "cosine":
        return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))
    if kind == "exp":
        if start <= 0:
            return end
        k = -math.log(max(end, 1e-8) / start) / max(decay_steps, 1)
        return max(end, start * math.exp(-k * step))
    return start


def train(args):
    device = args.device
    set_seed(args.seed)

    # ----- Env & dimensions -----
    env = make_env(args.seed, args.n_agents)
    agents = list(env.possible_agents) if hasattr(env, "possible_agents") else list(env.agents)
    sample_obs, _ = env.reset(seed=args.seed)
    obs_dim = sample_obs[agents[0]].shape[0]
    act_dim = env.action_space(agents[0]).n
    state_dim = obs_dim * args.n_agents + args.n_agents  # concat all obs + one-hot agent id

    # ----- Networks & optimizers -----
    if args.recurrent:
        from model import RecurrentActorGRU, RecurrentCentralCriticGRU
        actor = RecurrentActorGRU(obs_dim, act_dim, hidden=args.hidden).to(device)
        critic = RecurrentCentralCriticGRU(state_dim, hidden=args.hidden).to(device)
    else:
        actor = Actor(obs_dim, act_dim, hidden=args.hidden).to(device)
        critic = CentralCritic(state_dim, hidden=args.hidden).to(device)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor, eps=1e-5, betas=(0.9, 0.98))
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, eps=1e-5, betas=(0.9, 0.98))

    # ----- Entropy coefficient source: schedule OR auto-tuning -----
    use_ent_adapt = args.ent_target is not None
    if use_ent_adapt:
        # Start from ent_coef_start; optimize log_beta
        init_beta = max(args.ent_coef_start, 1e-6)
        log_beta = torch.tensor(np.log(init_beta), device=device, requires_grad=True)
        opt_beta = torch.optim.Adam([log_beta], lr=args.ent_adapt_lr)

        def get_beta(_update_idx: int) -> torch.Tensor:
            with torch.no_grad():
                b = log_beta.exp().clamp_(args.ent_beta_min, args.ent_beta_max)
            return b
    else:
        def get_beta(update_idx: int) -> torch.Tensor:
            b = get_scheduled_ent(
                update_idx, kind=args.ent_schedule,
                start=args.ent_coef_start, end=args.ent_coef_end,
                decay_steps=args.ent_decay_steps
            )
            return torch.tensor(b, device=device)

    # ----- Helpers -----
    valnorm = ValueNorm()
    sampler = ParallelRolloutSampler(
        env=env,
        n_agents=args.n_agents,
        horizon=args.horizon,
        device=device,
        build_state_fn=lambda obs_dict, _: build_agent_specific_states(obs_dict, agents),
        return_sequences=args.recurrent,
        fixed_agents=agents
    )

    for it in range(1, args.updates + 1):
        # ========== 1) Collect on-policy trajectories ==========
        out = sampler.sample(actor, critic)

        if not args.recurrent:
            # Flat tensors: [T*n_agents, ...] on the correct device
            obs_t, state_t, act_t, old_logp_t, val_t, rew_t, done_t = out

            # ========== 2) Compute advantages & returns ==========
            with torch.no_grad():
                adv_t, ret_t = compute_gae_returns(
                    rew=rew_t, val=val_t, done=done_t,
                    gamma=args.gamma, lam=args.lam, n_agents=args.n_agents
                )
            # Normalize returns by ValueNorm snapshot
            valnorm.update(ret_t)
            ret_norm = valnorm.normalize(ret_t)

            # Advantage normalization
            adv_norm = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8) if args.adv_norm else whiten(adv_t)

            # ========== 3) PPO updates ==========
            early_stop = False
            for ep in range(args.epochs):
                if early_stop:
                    break

                dist = Categorical(logits=actor(obs_t))
                logp = dist.log_prob(act_t)
                entropy = dist.entropy()
                if entropy.dim() > 1:
                    entropy = entropy.sum(-1)
                entropy_mean = entropy.mean()

                ratio = torch.exp(logp - old_logp_t)
                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_norm

                beta = get_beta(it)
                policy_loss = -(torch.min(surr1, surr2) + beta * entropy).mean()

                # Value loss with PPO-style clipping (on normalized targets)
                v_pred = critic(state_t).squeeze(-1)
                v_pred_norm = valnorm.normalize(v_pred)
                if args.vclip > 0:
                    v_clipped = torch.clamp(v_pred_norm, ret_norm - args.vclip, ret_norm + args.vclip)
                    v_loss = 0.5 * torch.max(
                        (v_pred_norm - ret_norm).pow(2),
                        (v_clipped - ret_norm).pow(2)
                    ).mean()
                else:
                    v_loss = 0.5 * (v_pred_norm - ret_norm).pow(2).mean()

                # Approximate KL and guardrails
                approx_kl = (old_logp_t - logp).mean().abs()
                if args.kl_report and ep == 0:
                    print(f"    approx_KL={approx_kl.item():.4f}  beta={float(beta):.4f}")

                total_loss = policy_loss + args.vf_coef * v_loss + args.kl_coef * approx_kl

                # Backprop
                opt_actor.zero_grad(set_to_none=True)
                opt_critic.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                opt_actor.step()
                opt_critic.step()

                # Entropy auto-tuning step
                if use_ent_adapt:
                    # ascend on (H - H_target) by minimizing negative
                    ent_mean_det = entropy_mean.detach()
                    loss_beta = -(log_beta * (ent_mean_det - args.ent_target))
                    opt_beta.zero_grad(set_to_none=True)
                    loss_beta.backward()
                    with torch.no_grad():
                        log_beta.clamp_(math.log(args.ent_beta_min), math.log(max(args.ent_beta_max, args.ent_beta_min + 1e-8)))
                    opt_beta.step()

                # Early stop if KL too large
                if approx_kl.item() > args.kl_stop:
                    early_stop = True

            # Simple log (avg team reward per step)
            if it % args.log_every == 0:
                # average total reward per step across agents
                ep_ret = rew_t.view(args.n_agents, -1).sum(0).mean().item()
                print(
                    f"[Upd {it:04d}] pi_loss={policy_loss.item():.3f}  V_loss={v_loss.item():.3f}  "
                    f"Ent={entropy_mean.item():.3f}  AvgStepReward={ep_ret:.3f}  "
                    f"beta={float(get_beta(it)):.4f}"
                    + (f"  KL≈{approx_kl.item():.3f}" if args.kl_report else "")
                )

        else:
            # Sequences: [T, n_agents, ...]
            obs_seq, state_seq, act_seq, old_logp_seq, val_seq, rew_seq, done_seq = out
            T, nA = obs_seq.shape[0], obs_seq.shape[1]

            with torch.no_grad():
                adv_flat, ret_flat = compute_gae_returns(
                    rew=rew_seq.flatten(),
                    val=val_seq.flatten(),
                    done=done_seq.flatten(),
                    gamma=args.gamma, lam=args.lam, n_agents=nA
                )
                adv_seq = adv_flat.view(T, nA)
                ret_seq = ret_flat.view(T, nA)

            valnorm.update(ret_seq.flatten())
            ret_norm_seq = valnorm.normalize(ret_seq.flatten()).view(T, nA)
            adv_norm_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8) if args.adv_norm else \
                           (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

            policy_loss_acc = 0.0
            value_loss_acc = 0.0
            ent_track = []
            approx_kl_last = 0.0

            for ep in range(args.epochs):
                hA = None
                hC = None
                early_stop = False

                for t0, t1 in iter_chunks(T, args.chunk_len):
                    if early_stop:
                        break

                    # [nA, L, ...] and contiguous
                    o_chunk = obs_seq[t0:t1].transpose(0, 1).contiguous().to(device)
                    s_chunk = state_seq[t0:t1].transpose(0, 1).contiguous().to(device)
                    a_chunk = act_seq[t0:t1].transpose(0, 1).contiguous().to(device)
                    logp_old = old_logp_seq[t0:t1].transpose(0, 1).contiguous().to(device).detach()
                    adv_chunk = adv_norm_seq[t0:t1].transpose(0, 1).contiguous().to(device).detach()
                    retN_chunk = ret_norm_seq[t0:t1].transpose(0, 1).contiguous().to(device).detach()

                    logits, hA = actor(o_chunk, hA)    # [nA, L, act_dim]
                    v_seq,  hC = critic(s_chunk, hC)   # [nA, L]

                    dist = Categorical(logits=logits)
                    logp = dist.log_prob(a_chunk)      # [nA, L]
                    ent = dist.entropy()
                    if ent.dim() > 2:
                        ent = ent.sum(-1)
                    ent_mean = ent.mean()
                    ent_track.append(ent_mean)

                    beta = get_beta(it)
                    ratio = torch.exp(logp - logp_old)
                    surr1 = ratio * adv_chunk
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_chunk
                    pol_loss = -(torch.min(surr1, surr2) + beta * ent).mean()

                    # Value loss (normalized targets) + clipping
                    v_pred_norm = valnorm.normalize(v_seq.flatten()).view_as(v_seq)
                    if args.vclip > 0:
                        v_clip = torch.clamp(v_pred_norm, retN_chunk - args.vclip, retN_chunk + args.vclip)
                        vf_loss = 0.5 * torch.max(
                            (v_pred_norm - retN_chunk).pow(2),
                            (v_clip - retN_chunk).pow(2)
                        ).mean()
                    else:
                        vf_loss = 0.5 * (v_pred_norm - retN_chunk).pow(2).mean()

                    # Approx KL and guardrails
                    approx_kl = (logp_old - logp).mean().abs()
                    approx_kl_last = approx_kl.item()

                    total_loss = pol_loss + args.vf_coef * vf_loss + args.kl_coef * approx_kl

                    opt_actor.zero_grad(set_to_none=True)
                    opt_critic.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    opt_actor.step()
                    opt_critic.step()

                    if use_ent_adapt:
                        ent_mean_det = ent_mean.detach()
                        loss_beta = -(log_beta * (ent_mean_det - args.ent_target))
                        opt_beta.zero_grad(set_to_none=True)
                        loss_beta.backward()
                        with torch.no_grad():
                            log_beta.clamp_(math.log(args.ent_beta_min), math.log(max(args.ent_beta_max, args.ent_beta_min + 1e-8)))
                        opt_beta.step()

                    # detach hidden state between chunks
                    if hA is not None:
                        hA = hA.detach()
                    if hC is not None:
                        hC = hC.detach()

                    policy_loss_acc += float(pol_loss.item())
                    value_loss_acc += float(vf_loss.item())

                    if approx_kl.item() > args.kl_stop:
                        early_stop = True

            if it % args.log_every == 0:
                ep_ret = rew_seq.sum(dim=1).mean().item()
                ent_mean_all = torch.stack(ent_track).mean().item() if len(ent_track) else 0.0
                print(
                    f"[Upd {it:04d}] pi_loss={policy_loss_acc:.3f}  V_loss={value_loss_acc:.3f}  "
                    f"Ent={ent_mean_all:.3f}  AvgStepReward={ep_ret:.3f}  "
                    f"beta={float(get_beta(it)):.4f}"
                    + (f"  KL≈{approx_kl_last:.3f}" if args.kl_report else "")
                )

    env.close()


def parse_args():
    p = argparse.ArgumentParser("PettingZoo MAPPO (MPE)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n_agents", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)

    # PPO core
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.1)
    p.add_argument("--lr_actor", type=float, default=5e-4)
    p.add_argument("--lr_critic", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)

    # Entropy schedule
    p.add_argument("--ent_schedule", type=str, default="const",
                   choices=["none", "const", "linear", "cosine", "exp"])
    p.add_argument("--ent_coef_start", type=float, default=0.01)
    p.add_argument("--ent_coef_end", type=float, default=0.005)
    p.add_argument("--ent_decay_steps", type=int, default=300)

    # Entropy auto-tuning (enable by setting --ent_target)
    p.add_argument("--ent_target", type=float, default=None,
                   help="Target policy entropy; if set, enables auto-tuning β via Adam on log_beta.")
    p.add_argument("--ent_adapt_lr", type=float, default=1e-3)
    p.add_argument("--ent_beta_min", type=float, default=1e-4)
    p.add_argument("--ent_beta_max", type=float, default=0.5)

    # Advantage normalization & value loss
    p.add_argument("--adv_norm", action="store_true")
    p.add_argument("--vclip", type=float, default=0.2, help="PPO value clipping epsilon (0 disables).")
    p.add_argument("--vf_coef", type=float, default=0.5)

    # KL control
    p.add_argument("--kl_stop", type=float, default=0.06,
                   help="Early-stop PPO epochs if approx_KL exceeds this.")
    p.add_argument("--kl_coef", type=float, default=0.1,
                   help="Soft KL penalty coefficient added to total loss.")
    p.add_argument("--kl_report", action="store_true")

    # Sampling / training
    p.add_argument("--horizon", type=int, default=25)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)

    # Recurrent
    p.add_argument("--recurrent", action="store_true", help="Use GRU actor/critic")
    p.add_argument("--chunk_len", type=int, default=10, help="BPTT unroll length")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
