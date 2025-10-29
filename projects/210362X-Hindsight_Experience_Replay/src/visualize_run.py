# visualize_run_cv2.py
import time
import torch
import numpy as np
import cv2
import config as cfg
from models import Actor, RewardModel, Critic
from replay_buffer import PERBuffer
from save_load import load_all
from utils import set_seed, scale_action

try:
    import gymnasium as gym
except Exception:
    raise RuntimeError("gym/gymnasium is required. Install with 'pip install gymnasium' or 'gym'.")


def make_env(env_name):
    try:
        return gym.make(env_name, render_mode="rgb_array")  # use rgb_array for frame capture
    except:
        return gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")


def visualize_cv2(n_episodes=5, use_learned_reward=False):
    env = make_env(cfg.ENV_NAME)
    from train import GoalWrapper
    env = GoalWrapper(env, goal=np.array([0.0, 0.0]))
    set_seed(cfg.SEED, env)
    device = cfg.DEVICE

    actor = Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    critic = Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    reward_model = RewardModel(env.observation_space.shape[0]-2, env.action_space.shape[0]).to(device)
    buffer = PERBuffer(cfg.BUFFER_SIZE)
    load_all(actor, critic, reward_model, buffer, cfg, device=device)

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

            # optionally compute learned reward for display
            if use_learned_reward:
                s_un = s[:-2]
                a_env_t = torch.tensor(a_env, dtype=torch.float32, device=device).unsqueeze(0)
                s_un_t = torch.tensor(s_un, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    r_hat = reward_model(s_un_t, a_env_t, s_un_t).cpu().item()
                print(f"Env reward {r_env:.2f} | Learned reward {r_hat:.2f}")
            total += r_env

            # Render with OpenCV
            frame = env.render()  # returns rgb array
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Agent", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True
                break

            time.sleep(0.02)

        print(f"Episode {ep+1}: total reward {total:.2f}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize_cv2(n_episodes=3, use_learned_reward=True)
