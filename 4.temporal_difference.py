import numpy as np
from collections import namedtuple



"""
Here we combine exploration_exploitation with the balancing of short-long term rewards.
Algorithms for learning simultanously from Sequential and Evaluative feedbacks
"""


def temporal_difference(π, env, gamma=1.0, init_lr=0.5, min_lr=0.01, lr_decay_ratio=0.3, n_episodes=500):
    """
    Here we can update V as we go, no need to first generate a trajectory
    """
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    lrs = _decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)

    for e in range(n_episodes):
        state, done = env.reset(), False
        while not done:
            action = π(state)
            next_state, reward, done, _ = env.step(action)
            episode_continue = not done

            td_target = reward + gamma * V[next_state] * episode_continue
            td_error = td_target - V[state]

            V[state] += lrs[e] * td_error

            state = next_state
            
        V_track[e] = V
    return V, V_track

    
def n_step_td_learning(π, env, gamma=1.0, init_lr=0.5, min_lr=0.01, lr_decay_ratio=0.5, n_step=3, n_episodes=500):
    """
    Bootstrap after n steps to estimate the value function instead of after one step as 
    temporal_difference() do. This will allow reducing the bias. But the higher the n, the higher
    the variance, the lower the bias.
    """
    
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    lrs = _decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)
    discounts = np.logspace(0, n_step+1, num=n_step+1, base=gamma, endpoint=False)
    Experience = namedtuple('Experience', ['state', 'reward', 'next_state', 'done'])

    for e in range(n_episodes):
        state, done, path = env.reset(), False, []  # path holds the n most recent experience : a partial trajectory

        while not done or path is not None:
            path = path[1:]

            while not done and len(path) < n_step:
                # MC like: Interact and collect experiences (not until the end but either until done or until n interaction)
                action = π(state)
                next_state, reward, done, _ = env.step(action)
                experience = Experience(state, reward, next_state, done)
                path.append(experience)
                state = next_state

                if done: break



            n = len(path)
            est_state = path[0].state  # state to estimate
            rewards = np.array(path)[:,1] # get all the rewards
            partial_return = discounts[:n] * rewards  # discounted reward from est_state to n

            bs_val = discounts[-1] * V[next_state] * (not done)  # bootstraping value
            target = np.sum(np.append(partial_return, bs_val))
            error = target - V[est_state]

            V[est_state] +=  lrs[e] * error

            if len(path) == 1 and path[0].done:
                path = None
                
        V_track[e] = V
    return V, V_track
                



def _decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    """Learning rate decay algorithm for learning"""
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values




if __name__ == "__main__":
    Experience = namedtuple('Experience', ['state', 'reward', 'next_state', 'done'])
    path = []
    path.append(Experience(0, 20, 9, 0))
    path.append(Experience(1, 21, 9, 0))
    path.append(Experience(2, 22, 9, 0))
    path.append(Experience(3, 23, 9, 0))
    path.append(Experience(4, 24, 9, 0))
    path.append(Experience(5, 25, 9, 0))

    print(np.array(path)[:, 1])
