import random
import configparser
from pathlib import Path
from itertools import count
from collections import deque
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
import torch.optim as optim

"""
Advanced AC methods: DDPG
"""



class DDPG:
    pass



if __name__ == "__main__":
    
    folder = Path("/home/medhyvinceslas/Documents/courses/gdrl_rl_spe/deep_rl/policy_based_and_ac")
    config_file = folder / "config.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    
    conf = config["DEFAULT"]
    conf_ddpg = config["DDPG"]

    seed = conf.getint("seed")
    model_path = Path(folder / conf_ddpg.get("model_name"))
    is_evaluation = conf.getboolean("evaluate_only")

    # to get nA, nS and for evaluation
    env_name = conf_ddpg.get("env_name")
    env_eval = gym.make(env_name)
    nS, nA = env_eval.observation_space.shape[0], env_eval.action_space.n
    conf_ddpg["nS"] = f"{nS}"
    conf_ddpg["nA"] = f"{nA}"
    
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DDPG(conf_ddpg, seed, device)



