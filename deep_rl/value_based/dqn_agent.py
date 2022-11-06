from collections import deque
from itertools import count
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from dqns import DQN, DuelingDQN
from replay_buffer import ReplayBuffer
from action_selection import EGreedyExpStrategy


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
        self.use_ddqn = training_conf["use_ddqn"]
        self.use_dueling = training_conf["use_dueling"]
        self.tau = training_conf["tau"]

        self.memory_capacity = agent_conf["memory_capacity"]
        self.memory = ReplayBuffer(self.memory_capacity, self.batch_size, self.seed)

        nS = env_conf["nS"]
        nA = env_conf["nA"]

        if training_conf["use_dueling"]:
            self.behavior_policy = DuelingDQN(self.device, nS, nA, hidden_dims=(512, 128)).to(self.device)
            self.target_policy = DuelingDQN(self.device, nS, nA, hidden_dims=(512, 128)).to(self.device)
        else:
            self.behavior_policy = DQN(self.device, nS, nA, hidden_dims=(512, 128)).to(self.device)
            self.target_policy = DQN(self.device, nS, nA, hidden_dims=(512, 128)).to(self.device)

        self.optimizer = optim.RMSprop(self.behavior_policy.parameters(), lr=lr)

        print("\nAll good, let's start\n")
        print(f"- Running on: {self.device}")
        print(f"- Use Double DQN for estimation: {self.use_ddqn}")
        print(f"- Use Dueling architecture: {self.use_dueling}")
        print(f"- Network: {self.behavior_policy}\n")
    

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)    
    

    def interact_with_environment(self, env, state, nA):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.strategy.select_action(self.behavior_policy, state, nA)
        next_state, reward, done, _, _ = env.step(action)
        return action, reward, next_state, done

    def sample_and_learn(self):

        states, actions, rewards, next_states, dones = self.memory.sample(self.device)
        
        if self.use_ddqn:
            """
            Instead of asking the target policy what is the highest action values Q_targets.
            We split the responsability between the target policy and the behavior policy:
            First we ask the behavior policy : What is the action with highest value
            Then we ask the target : What is the value of that action.

            This sharing of responsability reduce the bias by reducing the overestimation of
            Q-values.
            """
            # Action that have the highest value: Index of action ==> FROM THE BEHAVIOR POLICY
            argmax_q_next = self.behavior_policy(next_states).detach().argmax(dim=1).unsqueeze(-1)

            # Action-values of "best" actions  ==> FROM THE TARGET POLICY
            Q_targets_next = self.target_policy(next_states).gather(1, argmax_q_next)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        else:
            # hisghest action-values : Q(Sₜ₊₁,a)
            Q_targets_next = self.target_policy(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        
        Q_expected = self.behavior_policy(states).gather(1, actions)

        loss = F.huber_loss(Q_expected, Q_targets, delta=np.inf)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def sync_weights(self, use_polyak_averaging=True):
        if(use_polyak_averaging):
            """
            Instead of freezing the target and doing a big update every n steps, we can slow down
            the target by mixing a big % of weight from the target and a small % from the 
            behavior policy. So the update will be smoother and continuous at each time step.
            For example we add 1% of new information learned by the behavior policy to the target
            policy at every step.

            - self.tau: ratio of the behavior network that will be mixed into the target network.
            tau = 1 means full update (100%)
            """
            if self.tau is None:
                raise Exception("You are using Polyak averaging but TAU is None")
            
            for t, b in zip(self.target_policy.parameters(), self.behavior_policy.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)
        else:
            """
            target network was frozen during n steps, now we are update it with the behavior network
            weight.
            """
            for t, b in zip(self.target_policy.parameters(), self.behavior_policy.parameters()):
                t.data.copy_(b.data)



if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    nS, nA = env.observation_space.shape[0], env.action_space.n

    ENV_CONF = {
        "nS": nS, "nA": nA
    }
    AGENT_CONF = {
        "memory_capacity": 50000
    }
    TRAIN_CONF = {
        "seed": 0, "batch_size": 64, "gamma": .99, "lr": .01, "tau": 0.1,
        "use_ddqn": True, "use_dueling": True,
        "n_episodes": 1000,
        "update_every": 20,
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
        state, is_terminal = env.reset(seed=TRAIN_CONF["seed"])[0], False
        score = 0

        for t_step in count():
            total_steps += 1
            action, reward, next_state, done = agent.interact_with_environment(env, state, nA)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > bs * warmup_bs:
                agent.sample_and_learn()  # one step optimization on the behavior policy
                agent.sync_weights(use_polyak_averaging=True)

            if done: break
            score += reward

        scores_window.append(score)

        if i_episode % 10 == 0:
            print(f"Episode {i_episode}\tAverage {last_n_score} scores: {np.mean(scores_window)}")
                
    
    env.close()
