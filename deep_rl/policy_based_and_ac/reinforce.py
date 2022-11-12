import configparser
from pathlib import Path
from itertools import count
from collections import deque
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
import torch.optim as optim

from fc import FCDAP

"""Policy Based

Goal is to maximize the true value function of a parameterized policy from all initial states.
So maximize the true value function by changing the policy (without touching the V-Fucntion)

So we want to find the gradient which will help reaching this objective
From a trajectory τ, we obtain at each step:

>>>> Gt(τ) ▽θ log πθ(At|St)

For the entire trajectory, we just sum over each step from t=0 to T:
>>>> sum( Gt(τ) ▽θ log πθ(At|St) )

So the gradient we are trying to estimate is:

>>>> ▽θ J(θ) = sum( Gt(τ) ▽θ log πθ(At|St) )

In plain words:
- We sample a trajectory
- For each step, we calculate the return from that step
- We use the return from step t to weight the probability of the action taken at step t

That mean if the return is bad at time t, it is because action taken at time t was bad
so by multiplying the bad return with the probability of that action, we reduce the likelihood
of that action being selected at that step.
"""


class Reinforce():

    def __init__(self, config, device):

        nS = config.getint("nS")
        nA = config.getint("nA")
        self.device = device
        self.gamma = config.getfloat("gamma")
        lr = config.getfloat("lr")

        hidden_dims = eval(config.get("hidden_dims"))
        self.policy = FCDAP(self.device, nS, nA, hidden_dims=hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.logpas = []
        self.rewards = []

    
    def interact_with_environment(self, state, env):
        self.policy.train()
        action, logpa, _ = self.policy.full_pass(state)
        next_state, reward, is_terminal, _, _ = env.step(action)

        self.logpas.append(logpa)
        self.rewards.append(reward)

        return next_state, is_terminal
    

    def learn(self):
        """
        Learn once full trajectory is collected
        """
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])

        discounts = torch.FloatTensor(discounts).unsqueeze(1)
        returns = torch.FloatTensor(returns).unsqueeze(1)
        self.logpas = torch.cat(self.logpas)

        # ▽θ J(θ) = sum( Gt(τ) ▽θ log πθ(At|St) ) we add negative because we perform gradient ascent
        loss = -(discounts * returns * self.logpas).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def evaluate(self, env, n_episodes, seed):
        self.policy.eval()
        eval_scores = []
        for _ in range(n_episodes):
            s, d = env.reset(seed=seed)[0], False
            eval_scores.append(0)

            for _ in count():
                with torch.no_grad():
                    a = self.policy.select_greedy_action(s)

                s, r, d, _, _ = env.step(a)
                eval_scores[-1] += r
                if d: break
    
        return np.mean(eval_scores), np.std(eval_scores)


    def reset_metrics(self):
        self.logpas = []
        self.rewards = []



if __name__ == "__main__":
    folder = Path("/home/medhyvinceslas/Documents/courses/gdrl_rl_spe/deep_rl/policy_based_and_ac")
    config_file = folder / "config.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    
    conf = config["DEFAULT"]
    conf_reinforce = config["REINFORCE"]

    model_path = Path(folder / conf_reinforce.get("model_name"))
    is_evaluation = conf.getboolean("evaluate_only")
    env_name = conf_reinforce.get("env_name")

    env = gym.make(env_name, render_mode="human") if is_evaluation else gym.make(env_name)
    nS, nA = env.observation_space.shape[0], env.action_space.n
    conf_reinforce["nS"] = f"{nS}"
    conf_reinforce["nA"] = f"{nA}"


    # Monte-Carlo reinforce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = Reinforce(conf_reinforce, device)
    seed = conf.getint("seed")
    
    if is_evaluation:
        agent.policy.load_state_dict(torch.load(model_path))
        mean_eval_score, _ = agent.evaluate(env, n_episodes=1, seed=seed)
    else:

        evaluation_scores = deque(maxlen=100)
        n_episodes = conf_reinforce.getint("n_episodes")
        goal_mean_100_reward = conf_reinforce.getint("goal_mean_100_reward")

        for i_episode in range(1, n_episodes + 1):
            state, is_terminal = env.reset(seed=seed)[0], False

            agent.reset_metrics()
            for t_step in count():
                new_state, is_terminal = agent.interact_with_environment(state, env)
                state = new_state
                if is_terminal: break
            
            agent.learn()
            mean_eval_score, _ = agent.evaluate(env, n_episodes=1, seed=seed)
            evaluation_scores.append(mean_eval_score)

            if len(evaluation_scores) >= 100:
                mean_100_eval_score = np.mean(evaluation_scores)
                print(f"Episode {i_episode}\tAverage mean 100 eval score: {mean_100_eval_score}")

                if(mean_100_eval_score >= goal_mean_100_reward):
                    torch.save(agent.policy.state_dict(), model_path)
                    break

    env.close()