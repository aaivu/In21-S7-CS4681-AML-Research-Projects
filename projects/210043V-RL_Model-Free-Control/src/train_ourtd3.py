import argparse, os, numpy as np, gymnasium as gym, torch
from our_td3.policy import OurTD3, OurTD3Config

def evaluate(env, agent, episodes=5):
    avg = 0.0
    for _ in range(episodes):
        o, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(o, noise_scale=0.0)
            o, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_ret += r
        avg += ep_ret
    return avg / episodes

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="HalfCheetah-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--start-steps", type=int, default=25_000)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--replay", type=str, default="uniform", choices=["uniform","per","vmfer"])
    p.add_argument("--use-vi", action="store_true")
    p.add_argument("--actor-delay", type=int, default=2)
    p.add_argument("--policy-noise", type=float, default=0.2)
    p.add_argument("--noise-clip", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    # new: output controls
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--tag", type=str, default=None, help="Override label in filenames (e.g., OurTD3, TD3, TD3BC)")
    args = p.parse_args()

    # Set up envs and reproducibility
    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    torch.manual_seed(args.seed); np.random.seed(args.seed); env.reset(seed=args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    cfg = OurTD3Config(
        obs_dim=obs_dim, act_dim=act_dim, act_limit=act_limit,
        replay=args.replay, use_vi=args.use_vi, actor_delay=args.actor_delay,
        policy_noise=args.policy_noise, noise_clip=args.noise_clip,
        tau=args.tau, batch_size=args.batch_size
    )
    agent = OurTD3(cfg, device=args.device)

    # Decide filename tag
    default_tag = "OurTD3" if (args.use_vi or args.replay != "uniform") else "TD3"
    tag = args.tag or default_tag
    env_name = args.env.replace("/", "-")
    prefix = os.path.join(args.outdir, f"{tag}{env_name}{args.seed}")
    os.makedirs(args.outdir, exist_ok=True)

    # Eval-only branch (does not write npy)
    if args.eval_only and args.checkpoint:
        agent.actor.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        avg_ret = evaluate(eval_env, agent, args.eval_episodes)
        print("Loaded:", args.checkpoint)
        print("Eval avg return:", avg_ret)
        return

    # Arrays to store evaluation curve
    eval_returns = []
    eval_steps = []

    o, _ = env.reset()
    ep_ret, ep_len = 0.0, 0
    for t in range(1, args.steps + 1):
        if t < args.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(o, noise_scale=0.1)

        o2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        agent.push(o, a, r, o2, float(term))
        o = o2
        ep_ret += r; ep_len += 1

        info = agent.update()

        if done:
            o, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        # periodic evaluation + save curves as .npy
        if t % args.eval_every == 0:
            avg_ret = evaluate(eval_env, agent, args.eval_episodes)
            eval_returns.append(float(avg_ret))
            eval_steps.append(int(t))

            # write three files for compatibility with your comparison scripts
            np.save(f"{prefix}_train_returns.npy", np.asarray(eval_returns, dtype=np.float32))
            np.save(f"{prefix}_train_steps.npy", np.asarray(eval_steps, dtype=np.int64))
            np.save(f"{prefix}.npy", np.asarray(eval_returns, dtype=np.float32))

            print(f"step {t} | eval_avg_return {avg_ret:.1f} | info {info} | saved {prefix}_train_returns.npy")

    # Final saves
    torch.save(agent.actor.state_dict(), f"{prefix}_final.pt")
    print("Saved final model:", f"{prefix}_final.pt")
    # save the final npy again to ensure completeness
    np.save(f"{prefix}_train_returns.npy", np.asarray(eval_returns, dtype=np.float32))
    np.save(f"{prefix}_train_steps.npy", np.asarray(eval_steps, dtype=np.int64))
    np.save(f"{prefix}.npy", np.asarray(eval_returns, dtype=np.float32))
    print("Saved curves:", f"{prefix}_train_returns.npy", f"{prefix}_train_steps.npy")

if _name_ == "_main_":
    main()