import random

class AgreementReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = int(capacity)
        self.data = [None]*self.capacity
        self.ptr = 0
        self.size = 0

    def push(self, transition):
        self.data[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size:int):
        idxs = random.sample(range(self.size), batch_size)
        return [self.data[i] for i in idxs], idxs