# simulate_from_pickles.py
import torch
import numpy as np
import config as cfg
from models import Actor
from replay_buffer import PERBuffer
from save_load import load_all
from utils import set_seed, scale_action

try:
    import gymnasium as gym
except Exception:
    raise RuntimeError("gym/gymnasium is required. Install with 'pip install gymnasium' or 'gym'.")

def make_env(env_name):
    try:
        return gym.make(env_name)
    except:
        return gym.make("LunarLanderContinuous-v3")

def simulate(n_episodes=5):
    env = make_env(cfg.ENV_NAME)
    from train import GoalWrapper
    env = GoalWrapper(env, goal=np.array([0.0,0.0]))
    set_seed(cfg.SEED, env)
    device = cfg.DEVICE

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    critic = None
    reward_model = None
    buffer = PERBuffer(cfg.BUFFER_SIZE)
    load_all(actor, critic if critic else actor, reward_model if reward_model else actor, buffer, cfg, device=device)  # safe load: only actor really needed

    for ep in range(n_episodes):
        s, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            s_t = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a = actor(s_t).cpu().numpy().squeeze(0)
            a_env = scale_action(a, env.action_space.low, env.action_space.high)
            
            s, r_env, terminated, truncated, info = env.step(a_env)
            done = terminated or truncated

            env.render()
            total += r
        print(f"Sim Episode {ep+1} total reward: {total:.2f}")

if __name__ == "__main__":
    simulate(n_episodes=3)
