# main.py
# Run examples:
#   Baseline:
#     python main.py --updates 500 --horizon 25 --epochs 10 --clip_eps 0.1
#   + Entropy schedule:
#     python main.py --ent_schedule linear --ent_coef_start 0.03 --ent_coef_end 0.006 --ent_decay_steps 400
#   + Recurrent MAPPO (GRU):
#     python main.py --recurrent --chunk_len 10 --horizon 100 --epochs 6 --ent_schedule linear --ent_coef_start 0.03 --ent_coef_end 0.008 --ent_decay_steps 400

import argparse
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


def get_ent_coef(step, *, kind, start, end, decay_steps):
    """Entropy schedule: const / linear / cosine / exp."""
    if kind == "const":
        return start
    t = min(step / max(decay_steps, 1), 1.0)
    if kind == "linear":
        return start + (end - start) * t
    if kind == "cosine":
        import math
        return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))
    if kind == "exp":
        import math
        if start <= 0:
            return end
        k = -math.log(max(end, 1e-8) / start) / max(decay_steps, 1)
        return max(end, start * math.exp(-k * step))
    return start


def train(args):
    set_seed(args.seed)

    # ----- Env & dimensions -----
    env = make_env(args.seed, args.n_agents)
    # Fix a consistent agent order:
    agents = list(env.possible_agents) if hasattr(env, "possible_agents") else list(env.agents)
    sample_obs, _ = env.reset(seed=args.seed)
    obs_dim = sample_obs[agents[0]].shape[0]
    act_dim = env.action_space(agents[0]).n

    # Centralized/agent-specific critic input: concat(all_obs) + one_hot(agent_id)
    state_dim = obs_dim * args.n_agents + args.n_agents

    # ----- Networks & optimizers -----
    if args.recurrent:
        from model import RecurrentActorGRU, RecurrentCentralCriticGRU
        actor = RecurrentActorGRU(obs_dim, act_dim, hidden=args.hidden).to(args.device)
        critic = RecurrentCentralCriticGRU(state_dim, hidden=args.hidden).to(args.device)
    else:
        actor = Actor(obs_dim, act_dim, hidden=args.hidden).to(args.device)
        critic = CentralCritic(state_dim, hidden=args.hidden).to(args.device)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor, eps=1e-5)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic, eps=1e-5)

    # ----- Helpers -----
    valnorm = ValueNorm()
    sampler = ParallelRolloutSampler(
        env=env,
        n_agents=args.n_agents,
        horizon=args.horizon,
        device=args.device,
        build_state_fn=lambda obs_dict, _: build_agent_specific_states(obs_dict, agents),
        return_sequences=args.recurrent,     # << NEW: sequences when recurrent
        fixed_agents=agents                  # << ensure consistent order
    )

    for it in range(1, args.updates + 1):
        # ========== 1) Collect on-policy trajectories ==========
        out = sampler.sample(actor, critic)
        if not args.recurrent:
            # Flat tensors: [T*n_agents, ...]
            obs_t, state_t, act_t, old_logp_t, val_t, rew_t, done_t = out
            # ========== 2) Compute advantages & returns ==========
            with torch.no_grad():
                adv_t, ret_t = compute_gae_returns(
                    rew=rew_t, val=val_t, done=done_t,
                    gamma=args.gamma, lam=args.lam, n_agents=args.n_agents
                )
            valnorm.update(ret_t)
            ret_norm = valnorm.normalize(ret_t)
            adv_norm = whiten(adv_t)

            # ========== 3) PPO updates ==========
            for _ in range(args.epochs):
                pi = Categorical(logits=actor(obs_t))
                logp = pi.log_prob(act_t)
                entropy = pi.entropy().mean()

                ent_coef_now = get_ent_coef(
                    it, kind=args.ent_schedule, start=args.ent_coef_start,
                    end=args.ent_coef_end, decay_steps=args.ent_decay_steps
                )

                ratio = torch.exp(logp - old_logp_t)
                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_norm
                policy_loss = -(torch.min(surr1, surr2).mean() + ent_coef_now * entropy)

                v_pred = critic(state_t)
                v_pred_norm = valnorm.normalize(v_pred)
                v_clip = torch.clamp(v_pred_norm, ret_norm - args.clip_eps, ret_norm + args.clip_eps)
                v_loss_unclipped = (v_pred_norm - ret_norm).pow(2)
                v_loss_clipped = (v_clip - ret_norm).pow(2)
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                opt_actor.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                opt_actor.step()

                opt_critic.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                opt_critic.step()

            # Simple log (avg team reward per step)
            if it % args.log_every == 0:
                ep_ret = rew_t.view(args.n_agents, -1).sum(0).mean().item()
                print(
                    f"[Upd {it:04d}] π_loss={policy_loss.item():.3f}  V_loss={value_loss.item():.3f}  "
                    f"Ent={entropy.item():.3f}  AvgStepReward={ep_ret:.3f}"
                )

        else:
            # Sequences: each is [T, n_agents, ...]
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
            adv_norm_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)

            policy_loss_acc = 0.0
            value_loss_acc = 0.0
            entropies = []

            # Truncated BPTT over time
            for _ in range(args.epochs):
                hA = None
                hC = None
                for t0, t1 in iter_chunks(T, args.chunk_len):
                    # [nA, L, ...], make contiguous!
                    o_chunk = obs_seq[t0:t1].transpose(0, 1).contiguous().to(args.device)
                    s_chunk = state_seq[t0:t1].transpose(0, 1).contiguous().to(args.device)
                    a_chunk = act_seq[t0:t1].transpose(0, 1).contiguous().to(args.device)
                    logp_old = old_logp_seq[t0:t1].transpose(0, 1).contiguous().to(args.device).detach()
                    adv_chunk = adv_norm_seq[t0:t1].transpose(0, 1).contiguous().to(args.device).detach()
                    retN_chunk = ret_norm_seq[t0:t1].transpose(0, 1).contiguous().to(args.device).detach()

                    logits, hA = actor(o_chunk, hA)          # [nA, L, act_dim]
                    v_seq,  hC = critic(s_chunk, hC)         # [nA, L]

                    pi = Categorical(logits=logits)
                    logp = pi.log_prob(a_chunk)              # [nA, L]
                    entropy = pi.entropy().mean()
                    entropies.append(entropy)

                    ent_coef_now = get_ent_coef(
                        it, kind=args.ent_schedule, start=args.ent_coef_start,
                        end=args.ent_coef_end, decay_steps=args.ent_decay_steps
                    )

                    ratio = torch.exp(logp - logp_old)
                    surr1 = ratio * adv_chunk
                    surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_chunk
                    policy_loss = -(torch.min(surr1, surr2).mean() + ent_coef_now * entropy)

                    # normalize critic outputs with current ValueNorm snapshot
                    v_pred_norm = valnorm.normalize(v_seq.flatten()).view_as(v_seq)
                    v_clip = torch.clamp(v_pred_norm, retN_chunk - args.clip_eps, retN_chunk + args.clip_eps)
                    vf_loss = 0.5 * torch.max(
                        (v_pred_norm - retN_chunk).pow(2),
                        (v_clip - retN_chunk).pow(2)
                    ).mean()

                    opt_actor.zero_grad()
                    policy_loss.backward()     # no retain_graph
                    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    opt_actor.step()

                    opt_critic.zero_grad()
                    vf_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                    opt_critic.step()

                    # IMPORTANT: truncate graph across chunks
                    if hA is not None:
                        hA = hA.detach()
                    if hC is not None:
                        hC = hC.detach()

                    policy_loss_acc += float(policy_loss.item())
                    value_loss_acc += float(vf_loss.item())


            if it % args.log_every == 0:
                ep_ret = rew_seq.sum(dim=1).mean().item()
                ent_mean = torch.stack(entropies).mean().item() if len(entropies) else 0.0
                print(
                    f"[Upd {it:04d}] π_loss={policy_loss_acc:.3f}  V_loss={value_loss_acc:.3f}  "
                    f"Ent={ent_mean:.3f}  AvgStepReward={ep_ret:.3f}"
                )

    env.close()


def parse_args():
    p = argparse.ArgumentParser("PettingZoo MAPPO (MPE)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n_agents", type=int, default=3)
    p.add_argument("--hidden", type=int, default=64)

    # PPO
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.1)    # keep < 0.2 for MARL stability
    p.add_argument("--lr_actor", type=float, default=5e-4)
    p.add_argument("--lr_critic", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)         # ≤10 (hard) / ≤15 (easy)

    # Entropy schedule (NEW)
    p.add_argument("--ent_schedule", type=str, default="const", choices=["const", "linear", "cosine", "exp"])
    p.add_argument("--ent_coef_start", type=float, default=0.01)
    p.add_argument("--ent_coef_end", type=float, default=0.005)
    p.add_argument("--ent_decay_steps", type=int, default=300)

    # Sampling / training
    p.add_argument("--horizon", type=int, default=25)        # increase (e.g., 100) for recurrent
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)

    # Recurrent (NEW)
    p.add_argument("--recurrent", action="store_true", help="Use GRU actor/critic")
    p.add_argument("--chunk_len", type=int, default=10, help="BPTT unroll length")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
