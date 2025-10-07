# main.py (Gymnasium-compatible)
import argparse
import os
import numpy as np
import torch
import gymnasium as gym

import utils
import TD3
import OurDDPG
import DDPG

def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    obs, _ = eval_env.reset(seed=seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        done = False
        ep_ret = 0.0
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
        avg_reward += ep_ret
        obs, _ = eval_env.reset()
    avg_reward /= float(eval_episodes)

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                 # TD3, DDPG, OurDDPG
    parser.add_argument("--env", default="HalfCheetah-v5")         # v5 for Gymnasium mujoco
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25_000, type=int)
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    os.makedirs("./results", exist_ok=True)
    if args.save_model:
        os.makedirs("./models", exist_ok=True)

    # Env + seeding
    env = gym.make(args.env)
    obs, _ = env.reset(seed=args.seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Spaces
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(np.max(np.abs(env.action_space.high)))

    # Policy init
    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
    )
    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state = obs
    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Action selection
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state))
            noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            action = (action + noise).clip(env.action_space.low, env.action_space.high)

        # Env step (Gymnasium API)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # time-limit truncation => don't cut bootstrap target
        done_bool = 0.0 if truncated else float(terminated)

        # Store in replay
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += float(reward)

        # Train
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            print(
                f"Total T: {t+1}  Episode Num: {episode_num+1}  "
                f"Episode T: {episode_timesteps}  Reward: {episode_reward:.3f}"
            )
            state, _ = env.reset()
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1

        # Periodic eval + save
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", np.array(evaluations, dtype=np.float32))
            if args.save_model:
                policy.save(f"./models/{file_name}")