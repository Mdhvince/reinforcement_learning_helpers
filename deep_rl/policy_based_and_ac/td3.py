import random
import configparser
from pathlib import Path
from itertools import count
from collections import deque
from matplotlib import animation
import matplotlib.pyplot as plt
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
import torch.optim as optim

"""
TD3: Twin Delayed DDPG add some improvement to the ddpg algorithm
- Double learning technique as in DDQN but using a single twin network
- Add noise, not only to the action but also to the target action
- Delays updates of the policy network, the target network and twin target network
"""