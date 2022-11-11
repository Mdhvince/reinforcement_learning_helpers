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

"""Asynchronous Advantage Actor-Critic

VPG still uses MC returns. In A3C we use n-step return collected from multiple workers.
These workers update their local networks and a shared network asynchronously.

Each worker have (As in VPG):
- A local policy network
- A local value network

There is a Shared Policy Network and a Shared Value Network

"""

class A3C():

    def __init__(self, ENV_CONF, TRAIN_CONF):

        nS = ENV_CONF["nS"]
        nA = ENV_CONF["nA"]
        self.device = TRAIN_CONF["device"]
        self.gamma = TRAIN_CONF["gamma"]
        lr_p = TRAIN_CONF["lrs"][0]
        lr_v = TRAIN_CONF["lrs"][1]
        self.seed = TRAIN_CONF["seed"]

        # Define policy network and shared policy network
        hidden_dims = (128, 64)
        self.policy = FCDAP(self.device, nS, nA, hidden_dims=hidden_dims).to(self.device)
        self.p_optimizer = SharedAdam(self.policy.parameters(), lr=lr_p)
        self.policy_model_max_grad_norm = 1

        self.shared_policy = FCDAP(
            self.device, nS, nA, hidden_dims=hidden_dims).to(self.device).share_memory()
        self.shared_p_optimizer = SharedAdam(self.shared_policy.parameters(), lr=lr_p)
        # -------------------------------------------------

        # Define value network and shared value network
        hidden_dims=(256, 128)
        self.value_model = FCV(self.device, nS, hidden_dims=hidden_dims).to(self.device)
        self.v_optimizer = SharedRMSprop(self.value_model.parameters(), lr=lr_v)
        self.value_model_max_grad_norm = float('inf')

        self.shared_value_model = FCV(
            self.device, nS, hidden_dims=hidden_dims).to(self.device).share_memory()
        self.shared_v_optimizer = SharedRMSprop(self.shared_value_model.parameters(), lr=lr_v)
        # -------------------------------------------------

        self.get_out_lock = mp.Lock()
        self.get_out_signal = torch.zeros(1, dtype=torch.int).share_memory_()

        self.entropy_loss_weight = 0.001
        self.max_n_steps = TRAIN_CONF["max_n_steps"]
        self.n_workers = TRAIN_CONF["n_workers"]

        print("Initialized")


    def work(self, rank):
        local_seed = self.seed + rank
        env = gym.make("CartPole-v1")

        torch.manual_seed(local_seed)
        np.random.seed(local_seed)
        random.seed(local_seed)

        nS, nA = env.observation_space.shape[0], env.action_space.n

        hidden_dims = (128, 64)
        local_policy_model = FCDAP(self.device, nS, nA, hidden_dims=hidden_dims).to(self.device)
        local_policy_model.load_state_dict(self.shared_policy.state_dict())

        hidden_dims=(256, 128)
        local_value_model = FCV(self.device, nS, hidden_dims=hidden_dims).to(self.device)
        local_value_model.load_state_dict(self.shared_value_model.state_dict())

        # while not self.get_out_signal:
        for _ in range(2):
            state, is_terminal = env.reset(seed=local_seed)[0], False

            n_steps_start = 0
            logpas, entropies, rewards, values = [], [], [], []
            for t_step in count(start=1):
                next_state, reward, is_terminal = self.interact_with_environment(
                    local_policy_model, local_value_model, state, env,
                    logpas, entropies, rewards, values
                )
                print(logpas)
                state = next_state

                if is_terminal or t_step - n_steps_start == self.max_n_steps:
                    next_value = 0 if is_terminal else local_value_model(state).detach().item()
                    rewards.append(next_value)

                    self.learn(
                        logpas, entropies, rewards, values, local_policy_model, local_value_model)
                    
                    logpas, entropies, rewards, values = [], [], [], []
                
                if is_terminal:
                    break

    
    def interact_with_environment(
        self, local_policy_model, local_value_model, state, env,
        logpas, entropies, rewards, values):

        action, logpa, entropy = local_policy_model.full_pass(state)
        next_state, reward, is_terminal, _, _ = env.step(action)

        logpas.append(logpa)
        rewards.append(reward)
        entropies.append(entropy)
        values.append(local_value_model(state))

        return next_state, reward, is_terminal
    

    def learn(self, logpas, entropies, rewards, values, local_policy_model, local_value_model):
        """
        Learn once n-step trajectory is collected
        """
        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        logpas = torch.cat(logpas)
        entropies = torch.cat(entropies) 
        values = torch.cat(values)

        # --------------------------------------------------------------------
        # A(St, At) = Gt - V(St)
        # Loss = -1/N * sum_0_to_N( A(St, At) * log πθ(At|St) + βH )

        advantage = returns - values
        policy_loss = -(discounts * advantage.detach() * logpas).mean()
        entropy_loss_H = -entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss_H

        self.shared_p_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            local_policy_model.parameters(), self.policy_model_max_grad_norm)

        self._update_shared_network(local_policy_model, self.shared_policy_model)
        self.shared_p_optimizer.step()
        local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        # --------------------------------------------------------------------
        # A(St, At) = Gt - V(St)
        # Loss = 1/N * sum_0_to_N( A(St, At)² )

        value_loss = advantage.pow(2).mul(0.5).mean()
        self.shared_v_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            local_value_model.parameters(), self.value_model_max_grad_norm)

        self._update_shared_network(local_value_model, self.shared_value_model)
        self.shared_v_optimizer.step()
        local_value_model.load_state_dict(self.shared_value_model.state_dict())


    def _update_shared_network(self, local, shared):
        for param, shared_param in zip(local.parameters(), shared.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad


    def evaluate(self, env, n_episodes=1):
        self.policy.eval()
        eval_scores = []
        for _ in range(n_episodes):
            s, d = env.reset(seed=self.seed)[0], False
            eval_scores.append(0)

            for _ in count():
                with torch.no_grad():
                    a = self.policy.select_greedy_action(s)

                s, r, d, _, _ = env.step(a)
                eval_scores[-1] += r
                if d: break
    
        return np.mean(eval_scores), np.std(eval_scores)



if __name__ == "__main__":

    env = gym.make("CartPole-v1")    
    nS, nA = env.observation_space.shape[0], env.action_space.n

    ENV_CONF = { "nS": nS, "nA": nA }
    TRAIN_CONF = {
        "seed": 42, "gamma": .99, "lrs": [0.0005, 0.0007], "n_episodes": 5000,
        "goal_mean_100_reward": 700, "max_n_steps": 50, "n_workers": 8,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }
    model_path = Path("deep_rl/policy_based_policy_gradient/A3C_cartpolev1.pt")

    torch.manual_seed(TRAIN_CONF["seed"])
    np.random.seed(TRAIN_CONF["seed"])
    random.seed(TRAIN_CONF["seed"])

    agent = A3C(ENV_CONF, TRAIN_CONF)
    
    evaluation_scores = deque(maxlen=100)
    workers = [mp.Process(target=agent.work, args=(rank,)) for rank in range(agent.n_workers)]
    [w.start() for w in workers]
    [w.join() for w in workers]

    print("Completed!")

    env.close()