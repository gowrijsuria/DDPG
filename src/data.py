import random
import numpy as np
from collections import namedtuple, deque

# Experience = namedtuple('Experience',('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, args, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.args  = args
        random.seed(self.args.seed_val)

    def push(self, experience):
        # Adds experience tuple ('state', 'action', 'next_state', 'reward') to memory
        self.memory.append(experience)

    def sampleBatch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)
