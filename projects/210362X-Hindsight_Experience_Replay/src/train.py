# train.py - Complete TD3 implementation with RND, Reward Ensemble, HER, and PER
import os
import time
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config as cfg
from models import Actor, TwinCritic, RewardEnsemble, RNDModel
from replay_buffer import PERBuffer, make_her_transitions
from utils import soft_update, set_seed, scale_action
from save_load import save_all

# Try to import gym or gymnasium
try:
    import gymnasium as gym
except Exception:
    raise RuntimeError("gym/gymnasium is required. Install with 'pip install gymnasium' or 'gym'.")

def make_env(env_name):
    try:
        env = gym.make(env_name)
        print("Created env:", env_name)
        return env
    except Exception as e:
        fallback = "LunarLanderContinuous-v3"
        print(f"Could not create {env_name} ({e}), falling back to {fallback}")
        env = gym.make(fallback)
        return env

# Wrapper to add a goal to the observation
class GoalWrapper(gym.ObservationWrapper):
    def __init__(self, env, goal=None):
        super().__init__(env)
        obs_space = env.observation_space
        low = np.concatenate([obs_space.low, np.array([-np.inf, -np.inf])])
        high = np.concatenate([obs_space.high, np.array([np.inf, np.inf])])
        try:
            from gymnasium.spaces import Box
        except ImportError:
            from gym.spaces import Box
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.goal = goal if goal is not None else np.array([0.0, 0.0])
    
    def observation(self, obs):
        return np.concatenate([obs, np.array(self.goal, dtype=np.float32)])
    
    def set_goal(self, goal):
        self.goal = np.array(goal, dtype=np.float32)

def compute_env_reward_for_goal(s, a, sp, goal):
    pos = sp[:2]
    dist = np.linalg.norm(pos - goal)
    r = -dist
    if dist < 0.2:
        r += 100.0
    return r

# Running mean std for RND normalization
class RunningMeanStd:
    def __init__(self, eps=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
    
    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = x.numel()
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * (self.count * batch_count / tot_count)
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

def train():
    env = make_env(cfg.ENV_NAME)
    env = GoalWrapper(env, goal=np.array([0.0, 0.0]))
    set_seed(cfg.SEED, env)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    device = cfg.DEVICE

    # Actor and target actor
    actor = Actor(obs_dim, act_dim).to(device)
    target_actor = Actor(obs_dim, act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    actor_opt = optim.Adam(actor.parameters(), lr=cfg.ACTOR_LR)

    # Twin Critic (TD3) and target
    critic = TwinCritic(obs_dim, act_dim).to(device)
    target_critic = TwinCritic(obs_dim, act_dim).to(device)
    target_critic.load_state_dict(critic.state_dict())
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.CRITIC_LR, weight_decay=0.0)

    # Reward ensemble + RND initialization
    reward_ensemble = None
    reward_ensemble_opt = None
    if cfg.USE_REWARD_ENSEMBLE:
        reward_ensemble = RewardEnsemble(obs_dim - 2, act_dim, ensemble_size=cfg.REWARD_ENSEMBLE_SIZE).to(device)
        reward_ensemble_opt = optim.Adam(reward_ensemble.parameters(), lr=cfg.REWARD_ENSEMBLE_LR)

    rnd = None
    rnd_opt = None
    rnd_rms = None
    if cfg.USE_RND:
        rnd = RNDModel(obs_dim - 2, hidden=(128, 128)).to(device)
        rnd_opt = optim.Adam(rnd.predictor.parameters(), lr=cfg.RND_PRED_LR)
        rnd_rms = RunningMeanStd()

    buffer = PERBuffer(cfg.BUFFER_SIZE, alpha=cfg.PRIORITIZED_ALPHA, eps=cfg.PRIORITIZED_EPS)

    # Bookkeeping for plotting
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    td_errors_track = []
    success_rates = []

    global_step = 0
    policy_update_iter = 0

    for ep in trange(cfg.MAX_EPISODES):
        s_raw, info = env.reset()
        ep_transitions = []
        ep_reward_sum = 0.0
        
        for step in range(cfg.MAX_STEPS):
            s = np.array(s_raw, dtype=np.float32)
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action = actor(s_tensor).cpu().numpy().squeeze(0)
            
            # Add exploration noise
            if cfg.EXPLORATION_NOISE > 0:
                noise = np.random.normal(0, cfg.EXPLORATION_NOISE, size=action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
            
            # Scale action to environment's action range
            a_env = scale_action(action, act_low, act_high)
            sp_raw, r_env, terminated, truncated, info = env.step(a_env)
            done = terminated or truncated
            
            ep_reward_sum += r_env

            # Store un-augmented states (without goal)
            s_un = s[:-2]
            sp_un = np.array(sp_raw, dtype=np.float32)[:-2]

            # Compute learned reward if needed
            r_used = r_env
            if cfg.USE_REWARD_ENSEMBLE and reward_ensemble is not None:
                reward_ensemble.eval()
                with torch.no_grad():
                    s_t = torch.tensor(s_un, dtype=torch.float32, device=device).unsqueeze(0)
                    a_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                    sp_t = torch.tensor(sp_un, dtype=torch.float32, device=device).unsqueeze(0)
                    mean_pred, var_pred, _ = reward_ensemble(s_t, a_t, sp_t)
                    r_learned = mean_pred.cpu().item()
                    # Mix env reward with learned reward
                    r_used = 0.5 * r_env + 0.5 * r_learned

            ep_transitions.append((
                s.copy(), action.copy(), r_used, np.array(sp_raw, dtype=np.float32).copy(), 
                done, info, s_un, a_env, sp_un, r_env
            ))
            
            s_raw = sp_raw
            global_step += 1

            # Training updates
            if len(buffer) >= cfg.BATCH_SIZE:
                beta = min(1.0, cfg.PRIORITIZED_BETA0 + (ep / cfg.MAX_EPISODES) * (1.0 - cfg.PRIORITIZED_BETA0))
                idxs, samples, is_weights = buffer.sample(cfg.BATCH_SIZE, beta=beta)

                # Prepare batch tensors
                batch_s = torch.tensor(np.vstack([x[0] for x in samples]), dtype=torch.float32, device=device)
                batch_a = torch.tensor(np.vstack([x[1] for x in samples]), dtype=torch.float32, device=device)
                batch_r = torch.tensor(np.array([x[2] for x in samples]), dtype=torch.float32, device=device)
                batch_sp = torch.tensor(np.vstack([x[3] for x in samples]), dtype=torch.float32, device=device)
                batch_done = torch.tensor(np.array([x[4] for x in samples]), dtype=torch.float32, device=device)

                # --- TD3 Critic Update ---
                with torch.no_grad():
                    # Target actions with smoothing
                    a_next = target_actor(batch_sp)
                    noise = (torch.randn_like(a_next) * cfg.TARGET_POLICY_NOISE).clamp(
                        -cfg.TARGET_POLICY_NOISE_CLIP, cfg.TARGET_POLICY_NOISE_CLIP
                    )
                    a_next = (a_next + noise).clamp(-1.0, 1.0)
                    
                    # Twin Q-values, take minimum
                    q1_next, q2_next = target_critic(batch_sp, a_next)
                    q_next = torch.min(q1_next, q2_next)
                    y = batch_r + (1.0 - batch_done) * cfg.GAMMA * q_next

                q1, q2 = critic(batch_s, batch_a)
                td1 = (y - q1)
                td2 = (y - q2)
                
                # Weighted MSE loss for both critics
                is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32, device=device)
                critic_loss = (is_weights_tensor * td1.pow(2)).mean() + (is_weights_tensor * td2.pow(2)).mean()

                critic_opt.zero_grad()
                critic_loss.backward()
                if cfg.MAX_GRAD_NORM:
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.MAX_GRAD_NORM)
                critic_opt.step()

                critic_losses.append(critic_loss.item())

                # --- Delayed Actor Update (TD3) ---
                if (global_step % cfg.POLICY_DELAY) == 0:
                    actor_loss = -critic.q1(batch_s, actor(batch_s)).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    if cfg.MAX_GRAD_NORM:
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.MAX_GRAD_NORM)
                    actor_opt.step()

                    actor_losses.append(actor_loss.item())

                    # Soft update both target networks
                    soft_update(target_actor, actor, cfg.TAU)
                    for p, tp in zip(critic.parameters(), target_critic.parameters()):
                        tp.data.copy_(tp.data * (1.0 - cfg.TAU) + p.data * cfg.TAU)

                # Update PER priorities using mean absolute TD
                abs_td = (np.abs(td1.detach().cpu().numpy()) + np.abs(td2.detach().cpu().numpy())) / 2.0
                abs_td = abs_td + cfg.PRIORITIZED_EPS
                buffer.update_priorities(idxs, abs_td.tolist())
                td_errors_track.append(np.mean(abs_td))

                # --- Reward Ensemble Training ---
                if cfg.USE_REWARD_ENSEMBLE and reward_ensemble is not None:
                    reward_ensemble.train()
                    s_un_batch = torch.tensor(np.vstack([x[6] for x in samples]), dtype=torch.float32, device=device)
                    a_env_batch = torch.tensor(np.vstack([x[7] for x in samples]), dtype=torch.float32, device=device)
                    sp_un_batch = torch.tensor(np.vstack([x[8] for x in samples]), dtype=torch.float32, device=device)
                    r_env_batch = torch.tensor(np.array([x[9] for x in samples]), dtype=torch.float32, device=device)

                    mean_pred, var_pred, stacked = reward_ensemble(s_un_batch, a_env_batch, sp_un_batch)
                    reward_loss = F.mse_loss(mean_pred, r_env_batch)
                    
                    reward_ensemble_opt.zero_grad()
                    reward_loss.backward()
                    reward_ensemble_opt.step()

                # --- RND Update & Intrinsic Reward ---
                if cfg.USE_RND and rnd is not None:
                    rnd.predictor.train()
                    s_rnd = torch.tensor(np.vstack([x[6] for x in samples]), dtype=torch.float32, device=device)
                    p, t = rnd(s_rnd)
                    rnd_err = (p - t).pow(2).mean(dim=-1)
                    
                    # Update running stats and normalize
                    rnd_rms.update(rnd_err.detach())
                    normalized = (rnd_err - rnd_rms.mean) / (torch.sqrt(rnd_rms.var) + 1e-8)
                    intrinsic_batch = normalized.detach().cpu().numpy()
                    
                    # Train predictor
                    rnd_loss = rnd_err.mean()
                    rnd_opt.zero_grad()
                    rnd_loss.backward()
                    rnd_opt.step()

            if done:
                break

        episode_rewards.append(ep_reward_sum)

        # HER transitions
        if cfg.HER_ON:
            her_transitions = make_her_transitions(
                [(t[0], t[1], t[2], t[3], t[4], t[5]) for t in ep_transitions], 
                k=cfg.HER_K, 
                goal_extractor=None
            )
            for s_her, a_her, _, sp_her, done_her, info_her in her_transitions:
                sp_un = sp_her[:-2]
                goal = sp_her[-2:]
                r_her_env = compute_env_reward_for_goal(None, None, sp_un, goal)
                buffer.add((s_her.copy(), a_her.copy(), r_her_env, sp_her.copy(), 
                           float(done_her), info_her, sp_un.copy(), a_her.copy(), sp_un.copy(), r_her_env))

        # Store actual episode transitions
        for (s, a, r_used, sp, done, info, s_un, a_env, sp_un, r_env) in ep_transitions:
            buffer.add((s.copy(), a.copy(), r_used, sp.copy(), float(done), info, 
                       s_un.copy(), a_env.copy(), sp_un.copy(), r_env))

        # Success metric
        final_pos = ep_transitions[-1][3][:2]
        success = (np.linalg.norm(final_pos - np.array(env.goal)) < 0.2)
        success_rates.append(1.0 if success else 0.0)

        # Logging
        if (ep + 1) % cfg.LOG_INTERVAL == 0:
            print(f"Episode {ep+1} | Reward: {ep_reward_sum:.2f} | Buffer: {len(buffer)} | "
                  f"Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")

        # Plotting
        if (ep + 1) % cfg.PLOT_FREQ == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(episode_rewards, label='Episode Reward')
            plt.title('Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(cfg.PLOTS_DIR, f"reward_ep_{ep+1}.png"))
            plt.close()

            if actor_losses:
                plt.figure()
                plt.plot(actor_losses)
                plt.title('Actor Loss')
                plt.savefig(os.path.join(cfg.PLOTS_DIR, f"actor_loss_ep_{ep+1}.png"))
                plt.close()
            
            if critic_losses:
                plt.figure()
                plt.plot(critic_losses)
                plt.title('Critic Loss')
                plt.savefig(os.path.join(cfg.PLOTS_DIR, f"critic_loss_ep_{ep+1}.png"))
                plt.close()
            
            if td_errors_track:
                plt.figure()
                plt.plot(td_errors_track)
                plt.title('TD-error (mean per update)')
                plt.savefig(os.path.join(cfg.PLOTS_DIR, f"td_errors_ep_{ep+1}.png"))
                plt.close()
            
            plt.figure()
            plt.plot(np.convolve(success_rates, np.ones(20)/20, mode='same'))
            plt.title('Success Rate (moving avg)')
            plt.savefig(os.path.join(cfg.PLOTS_DIR, f"success_ep_{ep+1}.png"))
            plt.close()

        # Save checkpoints
        if (ep + 1) % 100 == 0:
            save_all(actor, critic, reward_ensemble if reward_ensemble else None, buffer, cfg)

    # Final save
    save_all(actor, critic, reward_ensemble if reward_ensemble else None, buffer, cfg)
    print("Training complete. Final stats:")
    print(f"Average reward last 50: {np.mean(episode_rewards[-50:]):.2f}")

if __name__ == "__main__":
    train()