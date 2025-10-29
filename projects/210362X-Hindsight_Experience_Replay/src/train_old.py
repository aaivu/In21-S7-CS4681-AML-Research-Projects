

# train_old.py
import os
import time
import math
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from models import Actor, Critic, RewardModel
from replay_buffer import PERBuffer, make_her_transitions
from utils import soft_update, set_seed, scale_action
from save_load import save_all

# Try to import gym or gymnasium and handle env name fallback
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
        # fallback to continuous version name
        fallback = "LunarLanderContinuous-v2"
        print(f"Could not create {env_name} ({e}), falling back to {fallback}")
        env = gym.make(fallback)
        return env

# Wrapper to add a goal to the observation (goal = target landing x,y); for HER compatibility
class GoalWrapper(gym.ObservationWrapper):
    def __init__(self, env, goal=None):
        super().__init__(env)
        # We'll augment observation by appending goal (x,y) to end
        obs_space = env.observation_space
        # observation is Box(dtype)
        low = np.concatenate([obs_space.low, np.array([-np.inf, -np.inf])])
        high = np.concatenate([obs_space.high, np.array([np.inf, np.inf])])
        try:
            from gymnasium.spaces import Box
        except ImportError:
            from gym.spaces import Box

        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.goal = goal if goal is not None else np.array([0.0, 0.0])  # target landing coords
    def observation(self, obs):
        # append goal at end
        return np.concatenate([obs, np.array(self.goal, dtype=np.float32)])
    def set_goal(self, goal):
        self.goal = np.array(goal, dtype=np.float32)

def compute_env_reward_for_goal(s, a, sp, goal):
    # s, sp are full observations before wrapper (without appended goal)
    # We'll compute reward as negative distance to goal, plus original env reward's main terms if available.
    # As a simple function:
    pos = sp[:2]
    dist = np.linalg.norm(pos - goal)
    # high reward for being close to goal, rough shaping for leg contact will be ignored for simplicity
    r = -dist
    # bonus for being within a small tolerance
    if dist < 0.2:
        r += 100.0
    return r

def train():
    env = make_env(cfg.ENV_NAME)
    # wrap with goal wrapper
    env = GoalWrapper(env, goal=np.array([0.0, 0.0]))
    set_seed(cfg.SEED, env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    device = cfg.DEVICE

    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    target_actor = Actor(obs_dim, act_dim).to(device)
    target_critic = Critic(obs_dim, act_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.ACTOR_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.CRITIC_LR)

    reward_model = RewardModel(obs_dim - 2, act_dim).to(device)  # reward model uses original obs dim (without appended goal)
    reward_opt = optim.Adam(reward_model.parameters(), lr=cfg.REWARD_MODEL_LR)
    mse = nn.MSELoss()

    buffer = PERBuffer(cfg.BUFFER_SIZE, alpha=cfg.PRIORITIZED_ALPHA, eps=cfg.PRIORITIZED_EPS)

    # bookkeeping for plotting
    episode_rewards = []
    actor_losses = []
    critic_losses = []
    td_errors_track = []
    success_rates = []

    global_step = 0

    for ep in trange(cfg.MAX_EPISODES):
        
        s_raw, info = env.reset()
        # s_raw is observation with appended goal already via wrapper
        ep_transitions = []
        ep_reward_sum = 0.0
        success = 0
        for step in range(cfg.MAX_STEPS):
            s = np.array(s_raw, dtype=np.float32)
            s_tensor = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = actor(s_tensor).cpu().numpy().squeeze(0)
            # scale action from [-1,1] to environment's action range
            a_env = scale_action(action, act_low, act_high)
            sp_raw, r_env, terminated, truncated, info = env.step(a_env)
            done = terminated or truncated
            
            ep_reward_sum += r_env

            # store raw transitions but we will append goal in state so buffer sees goal-augmented states
            # For reward model training we pass un-augmented s/sp (strip appended goal)
            s_un = s[:-2]
            sp_un = np.array(sp_raw, dtype=np.float32)[:-2]

            # compute learned reward if needed
            if cfg.USE_LEARNED_REWARD:
                # predict learned reward using current model (no grad)
                reward_model.eval()
                with torch.no_grad():
                    s_t = torch.tensor(s_un, dtype=torch.float32, device=device).unsqueeze(0)
                    a_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                    sp_t = torch.tensor(sp_un, dtype=torch.float32, device=device).unsqueeze(0)
                    r_hat = reward_model(s_t, a_t, sp_t).cpu().item()
                # we choose to use a mixture: 50% env reward + 50% learned reward
                r_used = 0.5 * r_env + 0.5 * r_hat
            else:
                r_used = r_env

            ep_transitions.append((s.copy(), action.copy(), r_used, np.array(sp_raw, dtype=np.float32).copy(), done, info, s_un, a_env, sp_un, r_env))
            s_raw = sp_raw
            global_step += 1

            # If enough samples, perform updates
            if len(buffer) >= cfg.BATCH_SIZE:
                # sample
                beta = min(1.0, cfg.PRIORITIZED_BETA0 + (ep / cfg.MAX_EPISODES) * (1.0 - cfg.PRIORITIZED_BETA0))
                idxs, samples, is_weights = buffer.sample(cfg.BATCH_SIZE, beta=beta)

                # samples are stored as (s, a, r, sp, done, info, s_un, a_env, sp_un, r_env)
                batch_s = torch.tensor(np.vstack([x[0] for x in samples]), dtype=torch.float32, device=device)
                batch_a = torch.tensor(np.vstack([x[1] for x in samples]), dtype=torch.float32, device=device)
                batch_r = torch.tensor(np.array([x[2] for x in samples]), dtype=torch.float32, device=device)
                batch_sp = torch.tensor(np.vstack([x[3] for x in samples]), dtype=torch.float32, device=device)
                batch_done = torch.tensor(np.array([x[4] for x in samples]), dtype=torch.float32, device=device)

                # critic update
                with torch.no_grad():
                    a_next = target_actor(batch_sp)
                    q_next = target_critic(batch_sp, a_next)
                    y = batch_r + (1.0 - batch_done) * cfg.GAMMA * q_next
                q_val = critic(batch_s, batch_a)
                td_errors = (y - q_val).detach().cpu().numpy()
                critic_loss = (torch.tensor(is_weights, dtype=torch.float32, device=device) * (q_val - y).pow(2)).mean()
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                # actor update (policy gradient)
                actor_loss = -critic(batch_s, actor(batch_s)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # update priorities in PER using absolute TD errors
                abs_td = np.abs(td_errors) + cfg.PRIORITIZED_EPS
                buffer.update_priorities(idxs, abs_td.tolist())

                # soft update targets
                soft_update(target_actor, actor, cfg.TAU)
                soft_update(target_critic, critic, cfg.TAU)

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                td_errors_track.append(np.mean(abs_td))

                # train learned reward model using stored (s_un, a, sp_un, r_env) to predict actual env reward
                # build a small mini-batch from sampled experiences that have s_un stored
                s_un_batch = torch.tensor(np.vstack([x[6] for x in samples]), dtype=torch.float32, device=device)
                a_env_batch = torch.tensor(np.vstack([x[7] for x in samples]), dtype=torch.float32, device=device)
                sp_un_batch = torch.tensor(np.vstack([x[8] for x in samples]), dtype=torch.float32, device=device)
                r_env_batch = torch.tensor(np.array([x[9] for x in samples]), dtype=torch.float32, device=device)

                reward_model.train()
                r_pred = reward_model(s_un_batch, a_env_batch, sp_un_batch)
                reward_loss = mse(r_pred, r_env_batch)
                reward_opt.zero_grad()
                reward_loss.backward()
                reward_opt.step()

            if done:
                break

        episode_rewards.append(ep_reward_sum)
        # compute HER transitions and add to buffer
        if cfg.HER_ON:
            # create HER transitions where we replace goal in s/sp and recompute reward according to new goal
            her_transitions = make_her_transitions([(t[0], t[1], t[2], t[3], t[4], t[5]) for t in ep_transitions], k=cfg.HER_K, goal_extractor=None)
            # For each her transition recompute reward with compute_env_reward_for_goal using un-augmented sp to get env reward
            for s_her, a_her, _, sp_her, done_her, info_her in her_transitions:
                # extract un-augmented sp (drop appended goal)
                sp_un = sp_her[:-2]
                # goal is the last two entries in sp_her (we substituted them)
                goal = sp_her[-2:]
                # recompute reward to reflect this goal
                r_her_env = compute_env_reward_for_goal(None, None, sp_un, goal)
                # store transition with recomputed reward
                buffer.add((s_her.copy(), a_her.copy(), r_her_env, sp_her.copy(), float(done_her), info_her, sp_un.copy(), a_her.copy(), sp_un.copy(), r_her_env))

        # store actual episode transitions into buffer
        for (s, a, r_used, sp, done, info, s_un, a_env, sp_un, r_env) in ep_transitions:
            buffer.add((s.copy(), a.copy(), r_used, sp.copy(), float(done), info, s_un.copy(), a_env.copy(), sp_un.copy(), r_env))

        # success criteria: final landing near goal
        final_pos = ep_transitions[-1][3][:2]
        success = (np.linalg.norm(final_pos - np.array(env.goal)) < 0.2)
        success_rates.append(1.0 if success else 0.0)

        # logging & plotting
        if (ep + 1) % cfg.LOG_INTERVAL == 0:
            print(f"Episode {ep+1} | Reward: {ep_reward_sum:.2f} | Buffer size: {len(buffer)} | Avg Reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")

        if (ep + 1) % cfg.PLOT_FREQ == 0:
            # produce 4+ plots: episode reward, actor loss, critic loss, TD errors, success rate
            plt.figure(figsize=(10,6))
            plt.plot(episode_rewards, label='episode_reward')
            plt.title('Episode Reward')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True)
            path = os.path.join(cfg.PLOTS_DIR, f"reward_ep_{ep+1}.png")
            plt.savefig(path); plt.close()

            if actor_losses:
                plt.figure(); plt.plot(actor_losses); plt.title('Actor Loss'); plt.savefig(os.path.join(cfg.PLOTS_DIR, f"actor_loss_ep_{ep+1}.png")); plt.close()
            if critic_losses:
                plt.figure(); plt.plot(critic_losses); plt.title('Critic Loss'); plt.savefig(os.path.join(cfg.PLOTS_DIR, f"critic_loss_ep_{ep+1}.png")); plt.close()
            if td_errors_track:
                plt.figure(); plt.plot(td_errors_track); plt.title('TD-error (mean per update)'); plt.savefig(os.path.join(cfg.PLOTS_DIR, f"td_errors_ep_{ep+1}.png")); plt.close()
            plt.figure(); plt.plot(np.convolve(success_rates, np.ones(20)/20, mode='same')); plt.title('Success Rate (moving avg)'); plt.savefig(os.path.join(cfg.PLOTS_DIR, f"success_ep_{ep+1}.png")); plt.close()

        # occasionally save checkpoint
        if (ep + 1) % 100 == 0:
            save_all(actor, critic, reward_model, buffer, cfg)

    # final save
    save_all(actor, critic, reward_model, buffer, cfg)
    print("Training complete. Final stats:")
    print("Average reward last 50:", np.mean(episode_rewards[-50:]))

if __name__ == "__main__":
    train()
