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
import torch.multiprocessing as mp

from fc import FCAC


class MultiprocessEnv(object):

    def __init__(self, config, seed):
        self.n_workers = config.getint("n_workers")
        self.env_name = config.get("env_name")
        self.seed = seed

        # In A2C there is one learner in the main process and several workers in the env.
        # So we need a way for the agent to send command from the main process (parent) to the
        # workers (childs). We can achieve this using Pipe.
        self.pipes = [mp.Pipe() for worker_id in range(self.n_workers)]
        
        self.workers = []  # hold the workers so we can use .join() later in .close()
        
        for worker_id in range(self.n_workers):
            w = mp.Process(target=self.work, args=(worker_id, self.pipes[worker_id][1]))
            self.workers.append(w)
            w.start()


    def work(self, worker_id, child_process):
        seed = self.seed + worker_id
        env = gym.make(self.env_name)

        # Execute the received command 
        while True:
            cmd, kwargs = child_process.recv()
            if cmd == 'reset': child_process.send(env.reset(seed=seed)[0])
            elif cmd == 'step': child_process.send(env.step(**kwargs))
            else:
                env.close()
                del env
                child_process.close()
                break
    
    def reset(self, worker_id=None):
        """
        - If worker_id is not None: Send the reset message from the parent to the child.
        The child will receive the meesage in .work() then send the result of env.reset() to the
        parent here.
        - Otherwise, send the reset to all childs and get + stack their results (states)
        """
        if worker_id is not None:
            main_process, _ = self.pipes[worker_id]
            self.send_msg(('reset', {}), worker_id)
            state = main_process.recv()[0]
            return state
        
        self.broadcast_msg(('reset', {}))
        return np.vstack([main_process.recv() for main_process, _ in self.pipes])
    

    def step(self, actions):
        # batch of actions. each workers should have take an action, so len(actions) should be
        # equal to the number of workers.
        assert len(actions) == self.n_workers

        for worker_id in range(self.n_workers):
            msg = ('step', {'action': actions[worker_id]}) # dictionary will be pass as kwargs
                                                           # so argument will be key=value in the
                                                           # env.step()
                                                           
            self.send_msg(msg, worker_id)

        results = []
        for worker_id in range(self.n_workers):
            main_process, _ = self.pipes[worker_id]
            state, reward, done, info, _ = main_process.recv()
            results.append(
                (state, np.array(reward, dtype=np.float), np.array(done, dtype=np.float), info)
            )
        
        # return array of 2d arrays.
        # index 0 contains 2d arrays of states
        # index 1 contains 2d arrays of rewards ... 
        return [np.vstack(block) for block in np.array(results).T]
    

    def close(self):
        self.broadcast_msg(('close', {}))
        [w.join() for w in self.workers]
    
    
    def send_msg(self, msg, worker_id):
        main_process, _ = self.pipes[worker_id]
        main_process.send(msg)
    

    def broadcast_msg(self, msg):    
        [main_process.send(msg) for main_process, _ in self.pipes]




class A2C():
    def __init__(self, config, seed, device):
        self.device = device
        self.nS = config.getint("nS")
        self.nA = config.getint("nA")
        self.config = config
        self.seed = seed
        self.gamma = config.getfloat("gamma")
        self.hidden_dims = eval(config.get("hidden_dims"))
        self.lr = config.getfloat("lr")

        self.ac_model = FCAC(self.device, self.nS, self.nA, hidden_dims=self.hidden_dims).to(self.device)
        self.optimizer = optim.RMSprop(self.ac_model.parameters(), lr=self.lr)
        self.max_grad = config.getint("max_gradient")

        self.policy_loss_weight = config.getfloat("policy_loss_weight")
        self.value_loss_weight = config.getfloat("value_loss_weight")
        self.entropy_loss_weight = config.getfloat("entropy_loss_weight")

        self.max_n_steps = config.getint("max_n_steps")
        self.n_workers = config.getint("n_workers")
        self.tau = config.getfloat("tau")
    

    def interact_with_environment(self, states, mp_env):
       # Infer on batch of states
        actions, logpas, entropies, values = self.ac_model.full_pass(states)

        # send the 'step' cmd from main process to child process
        new_states, rewards, dones, _ = mp_env.step(actions)

        self.logpas.append(logpas)
        self.entropies.append(entropies)
        self.rewards.append(rewards)
        self.values.append(values)
        
        return new_states, dones
    

    def learn(self):
        logpas = torch.stack(self.logpas).squeeze()
        entropies = torch.stack(self.entropies).squeeze()
        values = torch.stack(self.values).squeeze()

        

        T = len(self.rewards)  # length of rewards (+ last boostraping value "next_values")

        # the sequence starts at base**start and ends with base**stop.
        discounts = np.logspace(start=0, stop=T, num=T, base=self.gamma, endpoint=False)
        rewards = np.array(self.rewards).squeeze()

        returns = []
        for w in range(self.n_workers):
            for t_step in range(T):  # each t_step contains n number of rewards (with n = n_workers)
                discounted_reward = discounts[:T-t_step] * rewards[t_step:, w]
                returns.append(np.sum(discounted_reward))
        
        returns = np.array(returns).reshape(self.n_workers, T)  # All returns per worker

        # here we use GAE to estimate robust targets for the action-advantage funtion
        # - use of a exponentially weighted combination of n-step action-advantage function targets
        
        np_values = values.data.numpy()
        # T-1 because the recall the last value in T=len(rewards) is a bootsrapping value
        tau_discounts = np.logspace(
            start=0, stop=T-1, num=T-1, base=self.gamma*self.tau, endpoint=False)
        
        advs = rewards[:-1] + self.gamma * np_values[1:] - np_values[:-1]

        gaes = []
        for w in range(self.n_workers):
            for t_step in range(T-1):
                discounted_advantage = tau_discounts[:T-1-t_step] * advs[t_step:, w]
                gaes.append(np.sum(discounted_advantage))

        gaes = np.array(gaes).reshape(self.n_workers, T-1)
        discounted_gaes = discounts[:-1] * gaes
        
        # For some tensors we use reshape instead of view because view only works on
        # contiguous tensors. When transposing the tensor, it becomes non-contiguous in memory.
        # we could have used also: x.contiguous().view(-1)

        # :-1, ... remove last row on the first dimension but keep all other dimensions
        values = values[:-1, ...].view(-1).unsqueeze(1)
        logpas = logpas.view(-1).unsqueeze(1)
        entropies = entropies.view(-1).unsqueeze(1)
        returns = torch.FloatTensor(returns.T[:-1]).reshape(-1, 1)
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).reshape(-1, 1)
        
        T -= 1
        T *= self.n_workers
        assert returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert logpas.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = self.policy_loss_weight * policy_loss + \
                self.value_loss_weight * value_loss + \
                self.entropy_loss_weight * entropy_loss        

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), self.max_grad)
        self.optimizer.step()


    def evaluate_one_episode(self, env, seed):
        self.ac_model.eval()
        eval_scores = []

        s, d = env.reset(seed=seed)[0], False
        eval_scores.append(0)

        for _ in count():
            with torch.no_grad():
                a = self.ac_model.select_action(s)

            s, r, d, _, _ = env.step(a)
            eval_scores[-1] += r
            if d: break
    
        self.ac_model.train()
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
    conf_a2c = config["A2C"]

    seed = conf.getint("seed")
    model_path = Path(folder / conf_a2c.get("model_name"))
    is_evaluation = conf.getboolean("evaluate_only")

    # just to get nA, nS and for evaluation
    env_name = conf_a2c.get("env_name")
    env_eval = gym.make(env_name)
    nS, nA = env_eval.observation_space.shape[0], env_eval.action_space.n
    conf_a2c["nS"] = f"{nS}"
    conf_a2c["nA"] = f"{nA}"
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = A2C(conf_a2c, seed, device)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if is_evaluation:
        env_inference = gym.make(env_name, render_mode="human")
        agent.ac_model.load_state_dict(torch.load(model_path))
        mean_eval_score, _ = agent.evaluate_one_episode(env_inference, seed=seed)
        print(mean_eval_score)
    else:

        # Train
        mp_env = MultiprocessEnv(conf_a2c, seed)
        states = mp_env.reset()
        
        episode, n_steps_start = 0, 0
        max_n_steps = conf_a2c.getint("max_n_steps")
        evaluation_scores = deque(maxlen=100)
        goal_mean_100_reward = conf_a2c.getint("goal_mean_100_reward")

        agent.reset_metrics()
        for t_step in count(start=1):

            # ---- From here, everything is stacked (2d arrays of n rows = n_workers)
            states, dones = agent.interact_with_environment(states, mp_env)

            if dones.sum() or t_step - n_steps_start == max_n_steps:
                next_values = agent.ac_model.get_state_value(states).detach().numpy() * (1 - dones)
                agent.rewards.append(next_values)
                agent.values.append(torch.Tensor(next_values))
                agent.learn()
                agent.reset_metrics()
                n_steps_start = t_step
            
            if dones.sum() != 0.:  # if at least one worker is done
                mean_eval_score, _ = agent.evaluate_one_episode(env_eval, seed)
                evaluation_scores.append(mean_eval_score)
                mean_100_eval_score = np.mean(evaluation_scores)
                print(f"Episode {episode}\tAverage mean 100 eval score: {mean_100_eval_score}")

                if mean_100_eval_score >= goal_mean_100_reward:
                    torch.save(agent.ac_model.state_dict(), model_path)
                    break

                # reset state of done workers so they can restart collecting while others continue.
                for i in range(agent.n_workers):
                    if dones[i]:
                        states[i] = mp_env.reset(worker_id=i)
                        episode += 1

        mp_env.close()


    




