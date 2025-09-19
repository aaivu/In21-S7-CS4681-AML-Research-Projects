import copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def _init_(self, state_dim, action_dim, max_action):
        super()._init_()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def _init_(self, state_dim, action_dim):
        super()._init_()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1


class TD3_BC(object):
    """
    TD3-BC actor loss:
        L_actor = -E[ Q1(s, π(s)) ] + w_bc * || π(s) - a_bc ||^2
    where w_bc = alpha / mean(|Q1(s, π(s))|).  (from TD3-BC paper)
    - If you have demonstrations: pass demo_buffer + demo_ratio>0 to .train()
    - If not, it uses replay actions (online regularizer).
    """
    def _init_(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        # --- TD3-BC knobs ---
        bc_alpha=2.5,         # strength of BC (0 disables)
        bc_start_it=0,        # start applying BC after this many gradient steps
        bc_anneal_steps=0,    # exp decay of BC after start (0 = no decay)
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.bc_alpha = float(bc_alpha)
        self.bc_start_it = int(bc_start_it)
        self.bc_anneal_steps = int(bc_anneal_steps)

        self.total_it = 0

    # ----- utils -----
    @staticmethod
    def _to_tensor(x):
        if isinstance(x, torch.Tensor): return x
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _bc_weight(self, q_abs_mean: float, it: int) -> float:
        if self.bc_alpha <= 0:
            return 0.0
        w = self.bc_alpha / (q_abs_mean + 1e-6)
        if self.bc_anneal_steps > 0 and it > self.bc_start_it:
            # exponential decay after bc_start_it
            w *= math.exp(-(it - self.bc_start_it) / self.bc_anneal_steps)
        return float(w)

    # ----- API -----
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            return self.actor(state).cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=256, demo_buffer=None, demo_ratio: float = 0.0):
        """
        demo_buffer: optional second buffer with expert demos (same .sample() API)
        demo_ratio:  fraction of the ACTOR batch to pull from demo_buffer (0..1)
        """
        self.total_it += 1

        # === Critic update (same as TD3) ===
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Delayed actor + target updates ===
        if self.total_it % self.policy_freq == 0:
            # Build actor batch. By default we reuse the critic batch (online BC).
            actor_state = state
            a_bc = action

            # If demos provided, mix a portion of the batch from demo_buffer
            if demo_buffer is not None and demo_ratio > 0.0:
                k = int(batch_size * demo_ratio)
                if k > 0:
                    ds, da, _, _, _ = demo_buffer.sample(k)  # only need (s, a)
                    # combine (state[:B-k] from replay) + (ds from demo)
                    actor_state = torch.cat([state[:batch_size - k], ds], dim=0)
                    a_bc       = torch.cat([action[:batch_size - k], da], dim=0)

            # TD3 policy gradient term
            pi = self.actor(actor_state)
            q1_pi = self.critic.Q1(actor_state, pi)
            actor_loss = -q1_pi.mean()

            # TD3-BC term (if enabled & after start)
            if self.bc_alpha > 0 and self.total_it >= self.bc_start_it:
                with torch.no_grad():
                    q_abs_mean = q1_pi.abs().mean().item()
                w_bc = self._bc_weight(q_abs_mean, self.total_it)
                if w_bc > 0:
                    bc_loss = F.mse_loss(pi, a_bc)  # L2 toward dataset/replay actions
                    actor_loss = actor_loss + w_bc * bc_loss

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft target updates
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    # ---- I/O ----
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
        self.actor_target = copy.deepcopy(self.actor)