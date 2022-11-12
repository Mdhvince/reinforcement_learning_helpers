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

"""Advantage Actor-Critic A2C (appears after A3C despite the name)

The benefit of A3C is not the fact of updated multiple networks thanks to multiple learners that
update networks asynchronously. But it it the fact of selecting experiences through multiple workers.

In A2C we have a single learner and multiple workers on the environment.
- We share one model:  use a single neural net for both the policy and the value network
(as per the dueling network). Sharing the model can be beneficial when working with images because
feature extraction can be computation-intensive.
"""

# TODO: Implement a clean A2C

if __name__ == "__main__":
    pass