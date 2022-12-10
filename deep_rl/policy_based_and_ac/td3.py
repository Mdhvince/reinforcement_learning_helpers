import random
from itertools import count
from collections import deque

import torch
import numpy as np
import torch.optim as optim

import utils
from fc import FCTQV, FCDP

"""
TD3: Twin Delayed DDPG add some improvement to the ddpg algorithm
- Double learning technique as in DDQN but using a single twin network for the critic
- Add noise, not only to the online action but also to the target action
- Delays updates of the actor, such that the critic get updated more frequently
"""


class TD3():
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

        self.memory = utils.ReplayBuffer(buffer_size, bs, seed)

        self.actor = FCDP(device, nS, action_bounds, hidden_dims)  # ReLu + Tanh
        self.actor_target = FCDP(device, nS, action_bounds, hidden_dims)

        self.critic = FCTQV(device, nS, nA, hidden_dims)  # using ReLu by default
        self.critic_target = FCTQV(device, nS, nA, hidden_dims)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_grad = float('inf')

        self.training_strategy = utils.NormalNoiseDecayStrategyContinuous(
            action_bounds, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=200000)

        self.eval_strategy = utils.GreedyStrategyContinuous(action_bounds)

        self.policy_noise_ratio = 0.1
        self.policy_noise_clip_ratio = 0.5
        self.train_actor_every = 2

        self.sync_weights()

    def interact(self, state, env):
        """same as ddpg"""

        min_samples = self.memory.batch_size * self.n_warmup_batches
        use_max_exploration = len(self.memory) < min_samples
        action = self.training_strategy.select_action(self.actor, state, use_max_exploration)

        next_state, reward, is_terminal, _ = env.step(action)

        experience = (state, action, reward, next_state, is_terminal)
        return experience

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def sample_and_learn(self, t_step):
        states, actions, rewards, next_states, is_terminals = self.memory.sample(self.device)

        with torch.no_grad():
            # compute noise for target action (in ddpg noise is only applied on the online action)
            # training the policy with noisy targets can be seen as a regularizer
            # the network is now forced to generalize over similar actions 
            a_ran = self.actor_target.upper - self.actor_target.lower
            a_noise = torch.randn_like(actions) * self.policy_noise_ratio * a_ran
            n_min = self.actor_target.lower * self.policy_noise_clip_ratio
            n_max = self.actor_target.upper * self.policy_noise_clip_ratio
            a_noise = torch.max(torch.min(a_noise, n_max), n_min)

            # Get the target noisy action
            a_next = self.actor_target(next_states)
            noisy_a_next = a_next + a_noise
            noisy_a_next = torch.max(
                torch.min(noisy_a_next, self.actor_target.upper), self.actor_target.lower
            )

            # Get Q_next from the TWIN critic, which is the min Q between the two streams
            Q_target_stream_a, Q_target_stream_b = self.critic_target(next_states, noisy_a_next)
            Q_next = torch.min(Q_target_stream_a, Q_target_stream_b)
            Q_target = rewards + self.gamma * Q_next * (1 - is_terminals)

        # update the critic
        Q_stream_a, Q_stream_b = self.critic(states, actions)
        error_a = Q_stream_a - Q_target
        error_b = Q_stream_b - Q_target

        critic_loss = error_a.pow(2).mul(0.5).mean() + error_b.pow(2).mul(0.5).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
        self.critic_optimizer.step()

        # delay actor update, so the critic is updated at higher rate. This give the critic the time
        # to settle into more accurate values because it is more sensible
        if t_step % self.train_actor_every == 0:
            a_pred = self.actor(states)

            # here we choose one of the 2 streams and we stick to it
            Q_pred = self.critic.Qa(states, a_pred)

            actor_loss = -Q_pred.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
            self.actor_optimizer.step()

    def evaluate_one_episode(self, env, seed):
        total_rewards = 0

        s, d = env.reset(), False

        for _ in count():
            with torch.no_grad():
                a = self.eval_strategy.select_action(self.actor, s)

            s, r, d, _ = env.step(a)
            total_rewards += r
            if d: break

        return total_rewards

    def sync_weights(self, use_polyak_averaging=True):
        if (use_polyak_averaging):

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

            for t, b in zip(self.critic_target.parameters(), self.critic.parameters()):
                t.data.copy_(b.data)

            for t, b in zip(self.actor_target.parameters(), self.actor.parameters()):
                t.data.copy_(b.data)


if __name__ == "__main__":

    folder, conf_default, conf_project = utils.get_project_configuration(project_id="TD3")

    seed = conf_default.getint("seed")
    is_evaluation = conf_default.getboolean("evaluate_only")
    env_name = conf_project.get("env_name")
    n_episodes = conf_project.getint("n_episodes")
    goal_mean_100_reward = conf_project.getint("goal_mean_100_reward")
    model_path = folder / conf_project.get("model_name")

    render = True if is_evaluation else False
    env = utils.make_pybullet_env(env_name, render)

    action_bounds = env.action_space.low, env.action_space.high
    nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
    conf_project["nS"] = f"{nS}"
    conf_project["nA"] = f"{nA}"

    torch.manual_seed(seed);
    np.random.seed(seed);
    random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    agent = TD3(action_bounds, conf_project, seed, device)

    if is_evaluation:
        agent.actor.load_state_dict(torch.load(model_path))
        total_rewards = agent.evaluate_one_episode(env, seed=seed)
    else:
        last_100_score = deque(maxlen=100)
        mean_of_last_100 = deque(maxlen=100)

        for i_episode in range(1, n_episodes + 1):
            state, is_terminal = env.reset(), False

            for t_step in count():
                state, action, reward, next_state, is_terminal = agent.interact(state, env)
                agent.store_experience(state, action, reward, next_state, is_terminal)
                state = next_state

                if len(agent.memory) > agent.memory.batch_size * agent.n_warmup_batches:
                    agent.sample_and_learn(t_step=t_step)

                if t_step % 2 == 0:
                    agent.sync_weights(use_polyak_averaging=True)

                if is_terminal: break

            # Evaluate
            total_rewards = agent.evaluate_one_episode(env, seed=seed)
            last_100_score.append(total_rewards)
            mean_100_score = np.mean(last_100_score)

            if i_episode % 100 == 0:
                print(f"Episode {i_episode}\tAverage mean {len(last_100_score)} eval score: {mean_100_score}")

            enough_sample = len(last_100_score) >= 100
            goal_reached = mean_100_score >= goal_mean_100_reward
            training_done = i_episode >= n_episodes

            if ((enough_sample and goal_reached) or training_done):
                torch.save(agent.actor.state_dict(), model_path)
                break

        env.close()
