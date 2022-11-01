from collections import deque

import numpy as np

class ReplayBuffer():
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
        self.bs = batch_size
    
    def sample(self):
        indices = np.random.choice(len(self.buffer), self.bs, replace=False)

        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        experiences = (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
        return experiences

    def __len__(self):
        return len(self.buffer)
        
