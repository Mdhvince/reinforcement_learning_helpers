import random
import configparser
from pathlib import Path
from itertools import count
from collections import deque
from matplotlib import animation
import matplotlib.pyplot as plt
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


def save_frames_as_gif(frames, filepath):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(filepath, writer='imagemagick', fps=60)


def inference(model, env, seed, action_bounds):
    total_rewards = 0
    frames = []

    eval_strategy = GreedyStrategy(action_bounds)
    s, d = env.reset(seed=seed)[0], False
    
    for _ in count():
        with torch.no_grad():
            a = eval_strategy.select_action(model, s)
        
        frames.append(env.render())
        s, r, d, trunc, _ = env.step(a)
        total_rewards += r
        if d or trunc: break
    
    env.close()

    return total_rewards, frames


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

        self.critic = FCQV(device, nS, nA, hidden_dims)  # using ReLu by default
        self.critic_target = FCQV(device, nS, nA, hidden_dims)

        self.actor = FCDP(device, nS, action_bounds, hidden_dims)  # ReLu + Tanh
        self.actor_target = FCDP(device, nS, action_bounds, hidden_dims)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.max_grad = float('inf')

        self.training_strategy = NormalNoiseStrategy(action_bounds, exploration_noise_ratio=0.1)
        self.eval_strategy = GreedyStrategy(action_bounds)
    
    
    def interact_with_environment(self, state, env):
        min_samples = self.memory.batch_size * self.n_warmup_batches

        use_max_exploration = len(self.memory) < min_samples

        action = self.training_strategy.select_action(self.actor,
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
        
        # update the critic: Li(θ) = ( r + γQ(s′,μ(s′; ϕ); θ) − Q(s,a;θi) )^2

        a_next = self.actor_target(next_states)
        Q_next = self.critic_target(next_states, a_next)
        Q_target = rewards + self.gamma * Q_next * (1 - is_terminals)
        Q = self.critic(states, actions)
        
        error = Q - Q_target.detach()
        critic_loss = error.pow(2).mul(0.5).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
        self.critic_optimizer.step()

        # update the actor: Li(ϕ) = -1/N * sum of Q(s, μ(s; ϕi); θi) 
          
        a_pred = self.actor(states)
        Q_pred = self.critic(states, a_pred)

        actor_loss = -Q_pred.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)        
        self.actor_optimizer.step()
 

    def evaluate_one_episode(self, env, seed):
        total_rewards = 0

        s, d = env.reset(seed=seed)[0], False
        
        for _ in count():
            with torch.no_grad():
                a = self.eval_strategy.select_action(self.actor, s)

            s, r, d, trunc, _ = env.step(a)
            total_rewards += r
            if d or trunc: break


        return total_rewards
    

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
            for t, b in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)
            
            # mix policy networks
            for t, b in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_ratio = (1.0 - self.tau) * t.data
                behavior_ratio = self.tau * b.data
                mixed_weights = target_ratio + behavior_ratio
                t.data.copy_(mixed_weights.data)
        else:
            """
            target network was frozen during n steps, now we are update it with the behavior network
            weight.
            """
            for t, b in zip(self.critic_target.parameters(), self.critic.parameters()):
                t.data.copy_(b.data)
            
            for t, b in zip(self.actor_target.parameters(), self.actor.parameters()):
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
    env = gym.make(env_name, render_mode="rgb_array") if is_evaluation else gym.make(env_name)
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
        agent.actor.load_state_dict(torch.load(model_path))
        total_rewards, frames = inference(agent.actor, env, seed, action_bounds)
        save_frames_as_gif(frames, filepath=str(folder / "ddpg.gif"))
    else:

        last_100_score = deque(maxlen=100)
        mean_of_last_100 = deque(maxlen=100)

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
            total_rewards = agent.evaluate_one_episode(env, seed=seed)
            last_100_score.append(total_rewards)
            
            if len(last_100_score) >= 100:
                mean_100_score = np.mean(last_100_score)
                print(f"Episode {i_episode}\tAverage mean 100 eval score: {mean_100_score}")
            
                if(mean_100_score >= goal_mean_100_reward):
                    torch.save(agent.actor.state_dict(), model_path)
                    break
            else:
                print(f"Length eval score: {len(last_100_score)}")
    
        env.close()



