import numpy as np
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.data = np.empty(self.capacity, dtype=object)
        self.write = 0
        self.size = 0

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def update(self, idx, priority):
        tree_idx = idx + self.capacity
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def add(self, priority, data):
        idx = self.write
        self.data[idx] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _retrieve(self, idx, s):
        left = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[1]

    def get(self, s):
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return (idx, self.tree[idx], self.data[data_idx], data_idx)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=1000000, eps=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.eps = eps
        self.max_prio = 1.0

    def push(self, transition):
        self.tree.add(self.max_prio, transition)

    def sample(self, batch_size):
        import numpy as np, random
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / max(1, batch_size)
        for i in range(batch_size):
            s = random.random() * segment + i * segment
            idx, p, data, _ = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        N = self.tree.size
        probs = np.array(priorities) / (self.tree.total() + 1e-12)
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1
        weights = (N * probs) ** (-beta)
        weights /= (weights.max() + 1e-12)
        return batch, idxs, weights.astype(np.float32)

    def update_priorities(self, idxs, prios):
        for idx, prio in zip(idxs, prios):
            p = (abs(prio) + self.eps) ** self.alpha
            self.tree.update(idx - self.tree.capacity, p)
            self.max_prio = max(self.max_prio, p)