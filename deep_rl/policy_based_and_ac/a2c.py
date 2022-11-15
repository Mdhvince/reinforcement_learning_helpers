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
        new_states, rewards, is_terminals, _ = mp_env.step(actions)

        self.logpas.append(logpas)
        self.entropies.append(entropies)
        self.rewards.append(rewards)
        self.values.append(values)
        
        return new_states, is_terminals


    def train(self):
        # torch.manual_seed(self.seed)
        # np.random.seed(self.seed)
        # random.seed(self.seed)

        # mp_env = MultiprocessEnv(self.config, self.seed)

        # states = mp_env.reset()
        # episode, n_steps_start = 0, 0
        # self.logpas, self.entropies, self.rewards, self.values = [], [], [], []

        # for t_step in count(start=1):
        #     states, is_terminals = 
        pass
    

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
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

    mp_env = MultiprocessEnv(conf_a2c, seed)
    states = mp_env.reset()
    
    episode, n_steps_start = 0, 0

    agent.reset_metrics()
    for t_step in count(start=1):
        states, is_terminals = agent.interact_with_environment(states, mp_env)

        print(is_terminals)
        print(is_terminals.sum())

        if t_step == 100: break
    mp_env.close()
        
        # # some if conditio here then
        # next_value = agent.ac_model.get_state_value(states).detach().numpy() * (1 - is_terminals)



    




