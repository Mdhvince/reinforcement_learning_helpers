import numpy as np

from itertools import count


"""
Here we combine exploration_exploitation with the balancing of short-long term rewards.
Algorithms for learning simultanously from Sequential and Evaluative feedbacks
"""


def mc_prediction(π, env, gamma=1., init_lr=.5, min_lr=.01, lr_decay_ratio=.3, n_episodes=500, max_steps=100, mode="FV"):
    """
    We start with init_lr and decay its value down to min_lr within the first lr_decay_ratio*100 %
    of n_episodes.
    The algorithm works for first and every visit monte carlo.

    Goal is to evaluate the value funtion.

    - mode : FV or EV for First-Visit or Every-Visit Monte carlo
    """
    nS = env.observation_space.n

    # Pre-compute the discount factors / learning rates
    discounts = np.logspace( 0, max_steps, num=max_steps, base=gamma, endpoint=False)
    lrs = _decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)

    # State-value function
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))

    # for each episode, collect experience
    for e in range(n_episodes):
        trajectory = _generate_trajectory(π, env, max_steps)
        visited = np.zeros(nS, dtype=np.bool)

        # for each experience
        for t, (state, _, reward, _, _) in enumerate(trajectory):
            state_has_been_visited = visited[state]  # true or false

            if state_has_been_visited and mode == "FV":
                # do not compute anything else if visited and FVMC is applied
                continue

            n_steps = len(trajectory[:t])

            future_rewards = trajectory[t:, 2]  # rewards from this point until the end
            G = np.sum(discounts[:n_steps] * future_rewards)

            error_estimate = (G - V[state])
            V[state] = V[state] + lrs[e] * error_estimate
        
        V_track[e] = V  # store the value of the entire episode for stats
        
    return V.copy(), V_track
                

    



def _decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    """Learning rate decay algorithm for learning"""
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def _generate_trajectory(π, env, max_steps=20):
    """Generate set of experience tuple"""
    done, trajectory = False, []
    while not done:
        state = env.reset()

        for t in count():  # equivalent of a while True : t+=1
            action = π(state)
            next_state, reward, done, _ = env.step(action)

            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)

            if done: break  # Episode ends and the trajectory has been successfully generated.

            if t >= max_steps - 1: # No trajectory generated in the required window so we RETRY.
                trajectory = []
                break

            state = next_state
    return np.array(trajectory, np.object)







if __name__ == "__main__":
    pass