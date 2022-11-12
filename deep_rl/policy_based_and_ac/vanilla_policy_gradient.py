import sys
import configparser
from pathlib import Path
from itertools import count
from collections import deque
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from fc import FCDAP, FCV
import deep_rl.helper_plots as hp


"""Vanilla Policy Gradient (VPG) or REINFORCE with baseline

In REINFORCE we full monte carlo returns to calculate the gradient : High variance because of the
accumulation of random event along a trajectory.
To deal with this variance, we use Vanilla Policy Gradient.

Issue:
Log probabilities are changing proportionally to the return : Gt(τ) ▽θ log πθ(At|St).
If we are in an evironement with only positive rewards (like the cartpole environment) we need
a way to differenciate "ok actions" & "best actions" : 
- this can be achieved by collecting a lot of data => But will lead to high variance

The other solution is to use: The action-advantage Function intstead of the return

A(St, At) = Gt - V(St)

the action-advantage center scores around 0 so:
- better than average actions will have a positive value
- worst than average actions will have a negative value

We can create 2 neural-networks, one to learn the policy and another to learn the state-value
function V.

We cannot call this actor-critic because only methods that learn V-function using bootstrapping
are, because they add bias so they can be qulified as a "critic".

"""

class VPG():

    def __init__(self, config, device):

        self.device = device
        nS = config.getint("nS")
        nA = config.getint("nA")
        self.gamma = config.getfloat("gamma")
        lrs = eval(config.get("lrs"))
        lr_p = lrs[0]
        lr_v = lrs[1]
        hidden_dims_p = eval(config.get("hidden_dims_policy_net"))
        hidden_dims_v = eval(config.get("hidden_dims_value_net"))
        self.entropy_loss_weight = config.getfloat("entropy_loss_weight")

        # Define policy network, value network and max gradient for gradient clipping
        self.policy = FCDAP(self.device, nS, nA, hidden_dims=hidden_dims_p).to(self.device)
        self.p_optimizer = optim.Adam(self.policy.parameters(), lr=lr_p)
        self.p_max_grad = config.getint("max_gradient_policy_net")

        self.value_model = FCV(self.device, nS, hidden_dims=hidden_dims_v).to(self.device)
        self.v_optimizer = optim.RMSprop(self.value_model.parameters(), lr=lr_v)
        self.v_max_grad = float(eval(config.get("max_gradient_value_net")))

        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []

    
    def interact_with_environment(self, state, env):
        action, logpa, entropy = self.policy.full_pass(state)
        next_state, reward, is_terminal, _, _ = env.step(action)

        self.logpas.append(logpa)
        self.rewards.append(reward)
        self.entropies.append(entropy)
        self.values.append(self.value_model(state))

        return next_state, is_terminal
    

    def learn(self):
        """
        Learn once full trajectory is collected
        """
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        self.logpas = torch.cat(self.logpas)
        self.entropies = torch.cat(self.entropies) 
        self.values = torch.cat(self.values)

        # --------------------------------------------------------------------
        # A(St, At) = Gt - V(St)
        # Loss = -1/N * sum_0_to_N( A(St, At) * log πθ(At|St) + βH )

        advantage = returns - self.values
        policy_loss = -(discounts * advantage.detach() * self.logpas).mean()
        entropy_loss_H = -self.entropies.mean()
        loss = policy_loss + self.entropy_loss_weight * entropy_loss_H

        self.p_optimizer.zero_grad()
        loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.p_max_grad)
        self.p_optimizer.step()

        # --------------------------------------------------------------------
        # A(St, At) = Gt - V(St)
        # Loss = 1/N * sum_0_to_N( A(St, At)² )

        value_loss = advantage.pow(2).mul(0.5).mean()
        self.v_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.v_max_grad)
        self.v_optimizer.step()


    def evaluate_one_episode(self, env, seed):
        self.policy.eval()
        eval_scores = []

        s, d = env.reset(seed=seed)[0], False
        eval_scores.append(0)

        for _ in count():
            with torch.no_grad():
                a = self.policy.select_greedy_action(s)

            s, r, d, _, _ = env.step(a)
            eval_scores[-1] += r
            if d: break
    
        self.policy.train()
        return np.mean(eval_scores), np.std(eval_scores)


    def reset_metrics(self):
        self.logpas = []
        self.rewards = []
        self.entropies = []
        self.values = []


if __name__ == "__main__":
    folder = Path("/home/medhyvinceslas/Documents/courses/gdrl_rl_spe/deep_rl/policy_based_and_ac")
    config_file = folder / "config.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    
    conf = config["DEFAULT"]
    conf_vpg = config["VPG"]

    model_path = Path(folder / conf_vpg.get("model_name"))
    is_evaluation = conf.getboolean("evaluate_only")
    env_name = conf_vpg.get("env_name")

    env = gym.make(env_name, render_mode="human") if is_evaluation else gym.make(env_name)
    nS, nA = env.observation_space.shape[0], env.action_space.n
    conf_vpg["nS"] = f"{nS}"
    conf_vpg["nA"] = f"{nA}"

    # Vanilla Policy Gradient
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = VPG(conf_vpg, device)
    seed = conf.getint("seed")
    moving_avg_100 = []

    if is_evaluation:
        agent.policy.load_state_dict(torch.load(model_path))
        mean_eval_score, _ = agent.evaluate_one_episode(env, seed=seed)
    else:
        evaluation_scores = deque(maxlen=100)
        n_episodes = conf_vpg.getint("n_episodes")
        goal_mean_100_reward = conf_vpg.getint("goal_mean_100_reward")

        for i_episode in range(1, n_episodes + 1):
            state, is_terminal = env.reset(seed=seed)[0], False

            agent.reset_metrics()
            for t_step in count():
                new_state, is_terminal = agent.interact_with_environment(state, env)
                state = new_state
                if is_terminal: break
            
            next_value = 0 if is_terminal else agent.value_model(state).detach().item()
            agent.rewards.append(next_value)
            
            agent.learn()
            mean_eval_score, _ = agent.evaluate_one_episode(env, seed=seed)
            evaluation_scores.append(mean_eval_score)

            if len(evaluation_scores) >= 100:
                mean_100_eval_score = np.mean(evaluation_scores)
                print(f"Episode {i_episode}\tAverage mean 100 eval score: {mean_100_eval_score}")
                moving_avg_100.append(mean_100_eval_score)

            if(mean_100_eval_score >= goal_mean_100_reward):
                torch.save(agent.policy.state_dict(), model_path)
                break

    env.close()

    if not is_evaluation:
        hp.basis_plotting_style(
            "Moving Avg. reward per episode (Evaluation)", "Episodes", "Avg. rewards")
        plt.plot(moving_avg_100)
        plt.show()