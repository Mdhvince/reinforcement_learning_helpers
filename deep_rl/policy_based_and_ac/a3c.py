import random
from pathlib import Path
from itertools import count
from collections import deque
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from fc import FCDAP, FCV
from shared_optimizers import SharedAdam, SharedRMSprop

"""Asynchronous Advantage Actor-Critic A3C

VPG still uses MC returns. In A3C we use n-step return collected from multiple workers.
These workers update their local networks and a shared network asynchronously.

Each worker have (As in VPG):
- A local policy network
- A local value network

There is a Shared Policy Network and a Shared Value Network

"""

# TODO: Implement a clean A3C


if __name__ == "__main__":
    pass