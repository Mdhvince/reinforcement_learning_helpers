import random
import configparser
from pathlib import Path
from itertools import count
from collections import deque, namedtuple

import gym
import torch
import numpy as np
import pybullet_envs
import matplotlib.pyplot as plt
from matplotlib import animation






def make_pybullet_env(env_name, render):
    spec = gym.envs.registry.spec(env_name)
    spec._kwargs["render"] = render
    env = gym.make(env_name)
    return env


def get_project_configuration(project_id="TD3"):
    folder = Path("/home/medhyvinceslas/Documents/courses/gdrl_rl_spe/deep_rl/policy_based_and_ac")
    config_file = folder / "config.ini"
    config = configparser.ConfigParser()
    config.read(config_file)

    conf_default, conf_project = config["DEFAULT"], config[project_id]

    return folder, conf_default, conf_project


def save_frames_as_gif(frames, filepath):

    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(filepath, writer='imagemagick', fps=60)

    
def inference(model, env, seed, eval_strategy):
    total_rewards = 0
    frames = []

    s, d = env.reset(seed=seed)[0], False
    
    for _ in count():
        with torch.no_grad():
            a = eval_strategy.select_action(model, s)
        
        print(a)
        frames.append(env.render())
        s, r, d, trunc, _ = env.step(a)
        total_rewards += r
        if d or trunc: break
    
    env.close()

    return total_rewards, frames



class GreedyStrategyContinuous():
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)


class NormalNoiseStrategyContinuous():
    def __init__(self, bounds, exploration_noise_ratio=0.1):
        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.ratio_noise_injected = 0

    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.exploration_noise_ratio * self.high

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
        return action


class NormalNoiseDecayStrategyContinuous():
    def __init__(self, bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=10000):
        self.t = 0
        self.low, self.high = bounds
        self.noise_ratio = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.decay_steps = decay_steps
        self.ratio_noise_injected = 0

    def _noise_ratio_update(self):
        noise_ratio = 1 - self.t / self.decay_steps
        noise_ratio = (self.init_noise_ratio - self.min_noise_ratio) * noise_ratio + self.min_noise_ratio
        noise_ratio = np.clip(noise_ratio, self.min_noise_ratio, self.init_noise_ratio)
        self.t += 1
        return noise_ratio

    def select_action(self, model, state, max_exploration=False):
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.noise_ratio * self.high

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)

        self.noise_ratio = self._noise_ratio_update()
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
        return action


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, device):
        """Randomly sample a batch of experiences from memory."""

        # give me batch_size "randomly" selected <S, A, Rₜ₊₁, Sₜ₊₁> in a list
        experiences = random.sample(self.memory, k=self.batch_size)

        # stack all the states together and convert them to a tensor
        states = torch.from_numpy(
                np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)

        actions = torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)

        rewards = torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)

        next_states = torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)

        dones = torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)
  
        return (states, actions, rewards, next_states, dones)  # (batch_size x 5)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

if __name__ == "__main__":
    pass