import numpy as np

from itertools import count


"""
Here we combine exploration_exploitation with the balancing of short-long term rewards.
Algorithms for learning simultanously from Sequential and Evaluative feedbacks
"""


def _decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
    """Learning rate decay algorithm for learning"""
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def generate_trajectory(pi, env, max_steps=20):
    """Generate set of experience tuple"""
    done, trajectory = False, []
    while not done:
        state = env.reset()

        for t in count():  # equivalent of a while True : t+=1
            action = pi(state)
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