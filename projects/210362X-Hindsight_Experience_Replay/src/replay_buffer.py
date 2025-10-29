# replay_buffer.py
import random
import numpy as np
import math
import pickle

class SumTree:
    """Simple sum-tree for proportional prioritization."""
    def __init__(self, capacity):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
        self.tree = np.zeros(2*self.capacity)
        self.data = [None]*self.capacity
        self.write = 0
        self.n_entries = 0
    def _propagate(self, idx, change):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2
    def _retrieve(self, idx, s):
        left = idx*2
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    def total(self):
        return self.tree[1]
    def add(self, p, data):
        idx = self.write + self.capacity
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(1, s)
        dataIdx = idx - self.capacity
        return idx, self.tree[idx], self.data[dataIdx]

class PERBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.size = 0
    def add(self, experience):
        p = (self.max_priority + self.eps) ** self.alpha
        self.tree.add(p, experience)
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size, beta=0.4):
        idxs = []
        segment = self.tree.total() / batch_size
        samples = []
        priorities = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            idxs.append(idx)
            samples.append(data)
            priorities.append(p)
        sampling_prob = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.size * sampling_prob, -beta)
        is_weights /= is_weights.max() + 1e-8
        return idxs, samples, is_weights
    def update_priorities(self, idxs, priorities):
        for idx, p in zip(idxs, priorities):
            p_adj = (abs(p) + self.eps) ** self.alpha
            self.tree.update(idx, p_adj)
            if p_adj > self.max_priority:
                self.max_priority = p_adj
    def __len__(self):
        return self.size
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "tree": self.tree.tree,
                "data": self.tree.data,
                "write": self.tree.write,
                "n_entries": self.tree.n_entries,
                "max_priority": self.max_priority,
                "capacity": self.tree.capacity
            }, f)
    def load(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.tree.tree = d["tree"]
        self.tree.data = d["data"]
        self.tree.write = d["write"]
        self.tree.n_entries = d["n_entries"]
        self.max_priority = d["max_priority"]
        self.size = min(self.tree.n_entries, self.tree.capacity)

# HER helper: store episodes, produce hindsight transitions
def make_her_transitions(episode_transitions, k=4, goal_extractor=None):
    """
    episode_transitions: list of (s, a, r, s', done, info)
    goal_extractor: function to get goal from state, default: use lander position from s'
    returns: list of transitions including HER substituted goals
    """
    transitions = list(episode_transitions)
    T = len(transitions)
    her_samples = []
    for t in range(T):
        s, a, r, sp, done, info = transitions[t]
        # pick k future indices
        for _ in range(k):
            future_idx = random.randint(t, T-1)
            # get goal from that future state's achieved_goal (via extractor)
            if goal_extractor is None:
                # default: last two elements of state are position in LunarLander: (x,y) are indices 0,1 typically
                # We'll rely on wrapper to ensure goal fields are known.
                achieved = sp[:2]
            else:
                achieved = goal_extractor(transitions[future_idx][3])
            # Create substituted transition where goal becomes achieved
            # We'll expect the state to be augmented with goal in a wrapper, so we substitute relevant part
            s_her = np.copy(s)
            sp_her = np.copy(sp)
            # assume goal is appended at the end of state vector
            s_her[-2:] = achieved
            sp_her[-2:] = achieved
            # recompute reward externally (caller should recompute reward when storing in buffer)
            her_samples.append((s_her, a, None, sp_her, done, info))
    return her_samples
