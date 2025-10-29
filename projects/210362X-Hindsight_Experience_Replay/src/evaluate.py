# evaluate.py
import time
import torch
import numpy as np
import gym
import config as cfg
from models import Actor, Critic, RewardModel
from replay_buffer import PERBuffer
from save_load import load_all
from utils import set_seed, scale_action

def make_env(env_name):
    try:
        return gym.make(env_name)
    except:
        return gym.make("LunarLanderContinuous-v3")

def run_eval(n_episodes=10, render=False):
    env = make_env(cfg.ENV_NAME)
    from train import GoalWrapper
    env = GoalWrapper(env, goal=np.array([0.0,0.0]))
    set_seed(cfg.SEED, env)
    device = cfg.DEVICE

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    critic = Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    reward_model = RewardModel(env.observation_space.shape[0]-2, env.action_space.shape[0]).to(device)
    buffer = PERBuffer(cfg.BUFFER_SIZE)
    load_all(actor, critic, reward_model, buffer, cfg, device=device)

    for ep in range(n_episodes):
        s = env.reset()
        total = 0.0
        done = False
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = actor(s_t).cpu().numpy().squeeze(0)
            a_env = scale_action(a, env.action_space.low, env.action_space.high)
            s, r, done, info = env.step(a_env)
            total += r
            if render:
                env.render()
                time.sleep(0.02)
        print(f"Eval Episode {ep+1} reward: {total:.2f}")

if __name__ == "__main__":
    run_eval(n_episodes=5, render=True)
