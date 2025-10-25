# sampler.py
from typing import Callable, Dict, Tuple, List
import numpy as np
import torch
from torch.distributions import Categorical


class ParallelRolloutSampler:
    """
    If return_sequences=False (default): returns flat tensors [T*n_agents, ...].
    If return_sequences=True: returns time-major sequences [T, n_agents, ...].
    """

    def __init__(
        self,
        env,
        n_agents: int,
        horizon: int,
        device: str,
        build_state_fn: Callable[[Dict[str, np.ndarray], list], Dict[str, np.ndarray]],
        return_sequences: bool = False,
        fixed_agents: List[str] = None,
    ):
        self.env = env
        self.agents = list(fixed_agents) if fixed_agents is not None else list(env.agents)
        self.n_agents = n_agents
        self.horizon = horizon
        self.device = device
        self.build_state = build_state_fn
        self.return_sequences = return_sequences

    @torch.no_grad()
    def sample(self, actor, critic) -> Tuple[torch.Tensor, ...]:
        if not self.return_sequences:
            return self._sample_flat(actor, critic)
        else:
            return self._sample_seq(actor, critic)

    @torch.no_grad()
    def _sample_flat(self, actor, critic):
        buf_obs, buf_state, buf_act = [], [], []
        buf_logp, buf_val, buf_rew, buf_done = [], [], [], []

        obs, _ = self.env.reset(seed=np.random.randint(1e9))
        steps_collected = 0

        while steps_collected < self.horizon:
            states = self.build_state(obs, self.agents)

            actions = {}
            for a in self.agents:
                o = torch.tensor(obs[a], dtype=torch.float32, device=self.device).unsqueeze(0)
                s = torch.tensor(states[a], dtype=torch.float32, device=self.device).unsqueeze(0)

                pi = Categorical(logits=actor(o))
                a_t = pi.sample()
                logp = pi.log_prob(a_t)
                v_t = critic(s)

                buf_obs.append(o)
                buf_state.append(s)
                buf_act.append(a_t)
                buf_logp.append(logp)
                buf_val.append(v_t)

                actions[a] = a_t.item()

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            dones = {a: bool(terminations[a] or truncations[a]) for a in self.agents}

            for a in self.agents:
                buf_rew.append(torch.tensor([rewards[a]], dtype=torch.float32, device=self.device))
                buf_done.append(torch.tensor([float(dones[a])], dtype=torch.float32, device=self.device))

            obs = next_obs
            steps_collected += 1

            if any(dones.values()):
                obs, _ = self.env.reset(seed=np.random.randint(1e9))

        obs_t = torch.cat(buf_obs, dim=0)
        state_t = torch.cat(buf_state, dim=0)
        act_t = torch.cat(buf_act, dim=0).squeeze(-1)
        logp_t = torch.cat(buf_logp, dim=0).squeeze(-1)
        val_t = torch.cat(buf_val, dim=0).squeeze(-1)
        rew_t = torch.cat(buf_rew, dim=0).squeeze(-1)
        done_t = torch.cat(buf_done, dim=0).squeeze(-1)
        return obs_t, state_t, act_t, logp_t, val_t, rew_t, done_t

    @torch.no_grad()
    def _sample_seq(self, actor, critic):
        # Time-major sequences: [T, n_agents, ...]
        obs_seq, state_seq = [], []
        act_seq, logp_seq, val_seq = [], [], []
        rew_seq, done_seq = [], []

        obs, _ = self.env.reset(seed=np.random.randint(1e9))
        for t in range(self.horizon):
            states = self.build_state(obs, self.agents)

            step_obs, step_state = [], []
            step_act, step_logp, step_val = [], [], []

            actions = {}
            for a in self.agents:
                o = torch.tensor(obs[a], dtype=torch.float32, device=self.device).unsqueeze(0)
                s = torch.tensor(states[a], dtype=torch.float32, device=self.device).unsqueeze(0)

                # For sampling, compute per-step logits/values (we'll unroll during training)
                if hasattr(actor, "net"):  
                    # Feedforward actor
                    logits = actor(o)
                else:
                    # Recurrent actor: add time dim [B=1, T=1, obs]
                    logits, _ = actor(o.unsqueeze(1))   # → [1, 1, act_dim]
                    logits = logits[:, -1, :]           # take last step [1, act_dim]

                pi = Categorical(logits=logits)
                a_t = pi.sample()
                logp = pi.log_prob(a_t)

                if hasattr(critic, "net"):
                    v_t = critic(s)
                else:
                    v_seq, _ = critic(s.unsqueeze(1))   # → [1, 1]
                    v_t = v_seq[:, -1]                  # → [1]

                step_obs.append(o)
                step_state.append(s)
                step_act.append(a_t)
                step_logp.append(logp)
                step_val.append(v_t)

                actions[a] = a_t.item()

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            dones = {a: bool(terminations[a] or truncations[a]) for a in self.agents}

            obs_seq.append(torch.stack(step_obs, dim=1).squeeze(0))      # [nA, obs]
            state_seq.append(torch.stack(step_state, dim=1).squeeze(0))  # [nA, state]
            act_seq.append(torch.stack(step_act, dim=1).squeeze(0))      # [nA]
            logp_seq.append(torch.stack(step_logp, dim=1).squeeze(0))    # [nA]
            val_seq.append(torch.stack(step_val, dim=1).squeeze(0))      # [nA]
            rew_seq.append(torch.tensor([rewards[a] for a in self.agents], device=self.device, dtype=torch.float32))
            done_seq.append(torch.tensor([float(dones[a]) for a in self.agents], device=self.device, dtype=torch.float32))

            obs = next_obs
            if any(dones.values()):
                obs, _ = self.env.reset(seed=np.random.randint(1e9))

        # Stack -> [T, nA, ...]
        obs_t   = torch.stack(obs_seq, dim=0)
        state_t = torch.stack(state_seq, dim=0)
        act_t   = torch.stack(act_seq, dim=0).long()
        logp_t  = torch.stack(logp_seq, dim=0)
        val_t   = torch.stack(val_seq, dim=0).squeeze(-1)
        rew_t   = torch.stack(rew_seq, dim=0)
        done_t  = torch.stack(done_seq, dim=0)
        return obs_t, state_t, act_t, logp_t, val_t, rew_t, done_t
