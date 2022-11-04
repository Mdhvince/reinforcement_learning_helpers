from collections import deque
import os
import gc
import random
from pathlib import Path
from itertools import count
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from action_selection import GreedyStrategy, EGreedyExpStrategy

DDQN = False
TAU = 1e-3 


class DQN(nn.Module):
    """
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        - in_dim: state dimention as input (if state composed of [x, y, z] location, in_dim=3)
        - out_dim: number of action (will output the q(s, a) for all actions)
        - hidden_dims: (32, 32, 16) will create 3 hidden layers of 32, 32, 16 units.
        - activation: activation function
        """
        super(DQN, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)
        self.to(self.device)

    def _format(self, x):
        """
        Convert state to tensor if not and shape it correctly for the training process
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))
        
        x = self.out_layer(x)
        return x
    
    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable
    
    def format_experiences(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals


class Agent():
    """
    """
    def __init__(self, env_conf, agent_conf, training_conf):
        self.seed = training_conf["seed"]
        self.batch_size = training_conf["batch_size"]
        lr = training_conf["lr"]
        self.gamma = training_conf["gamma"]
        self.device = training_conf["device"]
        self.strategy = training_conf["strategy"]

        self.memory_capacity = agent_conf["memory_capacity"]
        self.memory = ReplayBuffer(self.memory_capacity, self.batch_size, self.seed)

        nS = env_conf["nS"]
        nA = env_conf["nA"]

        self.behavior_policy = DQN(self.device, nS, nA, hidden_dims=(512, 128)).to(self.device)
        self.target_policy = DQN(self.device, nS, nA, hidden_dims=(512, 128)).to(self.device)
        self.optimizer = optim.RMSprop(self.behavior_policy.parameters(), lr=lr)

        print("\nAll good, let's start\n")
    

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)    
    

    def interact_with_environment(self, env, state, nA):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.strategy.select_action(self.behavior_policy, state, nA)
        next_state, reward, done, _, _ = env.step(action)
        return action, reward, next_state, done

    def sample_and_learn(self):
        # - get and sample data from the replay buffer
        # - perform inference using the TARGET network (DQN as well) with Sₜ₊₁ as input
        # - Get the MAX Q value predicted: Q_target_next
        # - Compute the first part of the error term: Q_target = R + (gamma * Q_targets_next) * (1 - done) this last term is because Q_target = R is the episode is over
        # - perform inference using the LOCAL network (DQN as well) with S as input
        # - Get the values predicted for the actions selected A during interractions : Q_local
        # - Compute the second part using the LOCAL network: Q_expected
        # - Get the loss using Q_expected and Q_target as input (mse loss)
        # - Minimize the loss : self.optimizer.zero_grad(), loss.backward(), self.optimizer.step()
        # - Finally update the Target network

        states, actions, rewards, next_states, dones = self.memory.sample(self.device)
        
        if DDQN:
            argmax_q_next = self.behavior_policy(next_states).detach().argmax(dim=1).unsqueeze(-1)
            q_next = self.target_policy(next_states).gather(1, argmax_q_next)
            Q_targets = rewards + (self.gamma * q_next * (1 - dones))
        else:
            # forward prop to get actions-values + take the max Q(Sₜ₊₁,a) of the next state
            Q_targets_next = self.target_policy(next_states).detach().max(1)[0].unsqueeze(1)
            # use bellman equation to compute optimal action-value function of the current state
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        
        Q_expected = self.behavior_policy(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def sync_weights(self):
        for t, b in zip(self.target_policy.parameters(), self.behavior_policy.parameters()):
            t.data.copy_(b.data)



if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    nS, nA = env.observation_space.shape[0], env.action_space.n

    ENV_CONF = { "nS": nS, "nA": nA }

    AGENT_CONF = { "memory_capacity": 10000 }

    TRAIN_CONF = {
        "seed": 0, "batch_size": 64, "gamma": .99, "lr": .005,
        "n_episodes": 1000,
        "update_every": 50,
        "warmup_batch_size": 5,
        "strategy": EGreedyExpStrategy(),
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

    agent = Agent(ENV_CONF, AGENT_CONF, TRAIN_CONF)
    
    bs = TRAIN_CONF["batch_size"]
    warmup_bs = TRAIN_CONF["warmup_batch_size"]
    every = TRAIN_CONF["update_every"]
    total_steps = 0

    # stats
    last_n_score = 100
    scores_window = deque(maxlen=last_n_score)

    for i_episode in range(1, TRAIN_CONF["n_episodes"] + 1):
        state, is_terminal = env.reset()[0], False
        score = 0

        for t_step in count():
            total_steps += 1
            action, reward, next_state, done = agent.interact_with_environment(env, state, nA)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > bs * warmup_bs:
                agent.sample_and_learn()  # one step optimization on the behavior policy

            if total_steps % every == 0:
                agent.sync_weights()

            if done: break

            score += reward
        scores_window.append(score)

        if i_episode % last_n_score == 0:
            print(f"Episode {i_episode}\tAverage {last_n_score} scores: {np.mean(scores_window)}")
                
    
