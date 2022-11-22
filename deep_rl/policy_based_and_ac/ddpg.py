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

from fc import FCQV, FCDP
from replay_buffer import ReplayBuffer

"""
Advanced AC methods: DDPG

Pendulum env

### Action Space
The action is a `ndarray` representing the torque applied to free end of the pendulum.
| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

### Observation Space
The observation is a `ndarray` representing the x-y coordinates of the pendulum's free
end and its angular velocity.
| Num | Observation      | Min  | Max |
|-----|------------------|------|-----|
| 0   | x = cos(theta)   | -1.0 | 1.0 |
| 1   | y = sin(theta)   | -1.0 | 1.0 |
| 2   | Angular Velocity | -8.0 | 8.0 |
"""


class GreedyStrategy():
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)
    

class NormalNoiseStrategy():
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



class DDPG:
    def __init__(self, action_bounds, config, seed, device):

        self.config = config
        self.device = device
        buffer_size = config.getint("buffer_size")
        bs = config.getint("batch_size")
        nS = config.getint("nS")
        nA = config.getint("nA")
        hidden_dims = eval(config.get("hidden_dims"))
        lr = config.getfloat("lr")
        self.tau = config.getfloat("tau")
        self.gamma = config.getfloat("gamma")
        self.n_warmup_batches = config.getint("n_warmup_batches")

        self.memory = ReplayBuffer(buffer_size, bs, seed)

        self.behavior_value_net = FCQV(device, nS, nA, hidden_dims)  # using ReLu by default
        self.target_value_net = FCQV(device, nS, nA, hidden_dims)

        self.behavior_policy_net = FCDP(device, nS, action_bounds, hidden_dims)  # ReLu + Tanh
        self.target_policy_net = FCDP(device, nS, action_bounds, hidden_dims)

        self.value_optimizer = optim.Adam(self.behavior_value_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.behavior_policy_net.parameters(), lr=lr)

        self.max_grad = float('inf')

        self.training_strategy = NormalNoiseStrategy(action_bounds, exploration_noise_ratio=0.1)
        self.eval_strategy = GreedyStrategy(action_bounds)
    
    
    def interact_with_environment(self, state, env):
        min_samples = self.memory.batch_size * self.n_warmup_batches

        use_max_exploration = len(self.memory) < min_samples

        action = self.training_strategy.select_action(self.behavior_policy_net,
                                                      state,
                                                      use_max_exploration)
        
        next_state, reward, is_terminal, is_truncated, info = env.step(action)
        is_failure = is_terminal or is_truncated

        experience = (state, action, reward, next_state, float(is_failure))
        return experience
    

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)  


    def sample_and_learn(self):
        states, actions, rewards, next_states, is_terminals = self.memory.sample(self.device)
        
        # from here:
        # 1) use targets (policy & value) networks to get respectively a' & Q(s', a')
        # 2) use behavior value network to get Q(s, a)
        # 3) compute the value loss and optimize the behavior value network

        target_best_actions = self.target_policy_net(next_states)  # targets running on next_states
        target_Qsa = self.target_value_net(next_states, target_best_actions)
        target_Qsa = rewards + self.gamma * target_Qsa * (1 - is_terminals)
        behavior_Qsa = self.behavior_value_net(states, actions)  # behaviors running on states
        
        
        # Li(θ) = ( r + γQ(s′,μ(s′; ϕ); θ) − Q(s,a;θi) )^2
        error = behavior_Qsa - target_Qsa.detach()
        value_loss = error.pow(2).mul(0.5).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.behavior_value_net.parameters(), self.max_grad)
        self.value_optimizer.step()

        # from here:
        # 1) use behaviors (policy & value) networks to get respectively a_best & Q(s, a_best)
        # 3) compute the policy loss and optimize    
        predicted_best_action = self.behavior_policy_net(states)  # behaviors running on states
        predicted_Qsa = self.behavior_value_net(states, predicted_best_action)

        # Li(ϕ) = -1/N * sum of Q(s, μ(s; ϕi); θi) 
        policy_loss = -predicted_Qsa.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.behavior_policy_net.parameters(), self.max_grad)        
        self.policy_optimizer.step()
 

    def evaluate_one_episode(self, env, seed):
        eval_scores = []

        s, d = env.reset(seed=seed)[0], False
        eval_scores.append(0)

        for _ in count():
            with torch.no_grad():
                a = self.eval_strategy.select_action(self.behavior_policy_net, s)

            s, r, d, trunc, _ = env.step(a)
            eval_scores[-1] += r
            if d or trunc: break
    
        return np.mean(eval_scores), np.std(eval_scores)
    

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
            
            # mixe value networks
            for t, b in zip(self.target_value_net.parameters(), self.behavior_value_net.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)
            
            # mix policy networks
            for t, b in zip(self.target_policy_net.parameters(), self.behavior_policy_net.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)
        else:
            """
            target network was frozen during n steps, now we are update it with the behavior network
            weight.
            """
            for t, b in zip(self.target_value_net.parameters(), self.behavior_value_net.parameters()):
                t.data.copy_(b.data)
            
            for t, b in zip(self.target_policy_net.parameters(), self.behavior_policy_net.parameters()):
                t.data.copy_(b.data)



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

    env_name = conf_ddpg.get("env_name")
    env = gym.make(env_name, render_mode="human") if is_evaluation else gym.make(env_name)
    action_bounds = env.action_space.low, env.action_space.high
    nS, nA = env.observation_space.shape[0], env.action_space.shape[0]

    conf_ddpg["nS"] = f"{nS}"
    conf_ddpg["nA"] = f"{nA}"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = DDPG(action_bounds, conf_ddpg, seed, device)

    if is_evaluation:
        agent.behavior_policy_net.load_state_dict(torch.load(model_path))
        mean_eval_score, _ = agent.evaluate_one_episode(env, seed=seed)
    else:

        evaluation_scores = deque(maxlen=100)
        n_episodes = conf_ddpg.getint("n_episodes")
        goal_mean_100_reward = conf_ddpg.getint("goal_mean_100_reward")

        for i_episode in range(1, n_episodes + 1):
            state, is_terminal = env.reset(seed=seed)[0], False

            for t_step in count():
                state, action, reward, next_state, is_terminal = (
                        agent.interact_with_environment(state, env)
                )
                agent.store_experience(state, action, reward, next_state, is_terminal)
                state = next_state

                if len(agent.memory) > agent.memory.batch_size * agent.n_warmup_batches:
                    agent.sample_and_learn()
                    agent.sync_weights(use_polyak_averaging=True)
                
                if is_terminal: break
            
            # Evaluate
            mean_eval_score, _ = agent.evaluate_one_episode(env, seed=seed)
            evaluation_scores.append(mean_eval_score)
            
            if len(evaluation_scores) >= 10:
                mean_100_eval_score = np.mean(evaluation_scores)
                print(f"Episode {i_episode}\tAverage mean 100 eval score: {mean_100_eval_score}")
            
                if(mean_100_eval_score >= goal_mean_100_reward):
                    torch.save(agent.behavior_policy_net.state_dict(), model_path)
                    break




