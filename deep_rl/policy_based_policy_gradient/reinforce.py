from pathlib import Path
from itertools import count
from collections import deque
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

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


class FCDAP(nn.Module):  # Fully connected discret action policy
    """
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        """
        super(FCDAP, self).__init__()

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
    
    def full_pass(self, state):
        logits = self.forward(state)  # preferences over actions

        # sample action from the probability distribution
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        log_p_action = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)

        return action.item(), log_p_action, entropy
    
    def select_action(self, state):
        """Helper function for when we just need to sample an action"""
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())


class Reinforce():

    def __init__(self, ENV_CONF, TRAIN_CONF):

        nS = ENV_CONF["nS"]
        nA = ENV_CONF["nA"]
        self.device = TRAIN_CONF["device"]
        self.gamma = TRAIN_CONF["gamma"]
        lr = TRAIN_CONF["lr"]
        self.seed = TRAIN_CONF["seed"]

        hidden_dims = (128, 64)
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

    

    def reset_metrics(self):
        self.logpas = []
        self.rewards = []
    





if __name__ == "__main__":

    EVALUATE_ONLY = True

    if EVALUATE_ONLY:
        env = gym.make("CartPole-v1", render_mode="human")
    else:
        env = gym.make("CartPole-v1")
        
    nS, nA = env.observation_space.shape[0], env.action_space.n

    ENV_CONF = {
        "nS": nS, "nA": nA
    }
    AGENT_CONF = {
        "memory_capacity": 50000
    }
    TRAIN_CONF = {
        "seed": 0, "batch_size": 64, "gamma": .99, "lr": 0.0005,
        "n_episodes": 5000,
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }
    goal_mean_100_reward = 475

    # Monte-Carlo reinforce
    agent = Reinforce(ENV_CONF, TRAIN_CONF)
    model_path = Path("deep_rl/policy_based_policy_gradient/reinforce_cartpolev1.pt")

    if EVALUATE_ONLY:
        agent.policy.load_state_dict(
            torch.load(model_path)
        )
        mean_eval_score, _ = agent.evaluate(env, n_episodes=1)
    else:

        evaluation_scores = deque(maxlen=100)

        for i_episode in range(1, TRAIN_CONF["n_episodes"] + 1):
            state, is_terminal = env.reset(seed=TRAIN_CONF["seed"])[0], False

            agent.reset_metrics()
            for t_step in count():
                new_state, is_terminal = agent.interact_with_environment(state, env)
                state = new_state

                if is_terminal: break
            
            agent.learn()
            mean_eval_score, _ = agent.evaluate(env, n_episodes=1)
            evaluation_scores.append(mean_eval_score)

            if len(evaluation_scores) >= 100:
                mean_100_eval_score = np.mean(evaluation_scores)
                print(f"Episode {i_episode}\tAverage mean 100 eval score: {mean_100_eval_score}")

                if(mean_100_eval_score >= goal_mean_100_reward):
                    torch.save(agent.policy.state_dict(), model_path)
                    break

    env.close()