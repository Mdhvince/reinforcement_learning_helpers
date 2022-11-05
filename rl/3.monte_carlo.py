from collections import defaultdict

import numpy as np

import utils


def mc_control(
    π, env, gamma=1.,
    init_lr=.5, min_lr=.01, lr_decay_ratio=.3,
    init_eps=1., min_eps=.1, eps_decay_ratio=.9,
    n_episodes=500, max_steps=100, mode="FV"):
    """
    """
    nS = env.observation_space.n
    nA = env.action_space.n
    π_track = []                           # Hold the improved (greedy) policy per episode

    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)  # hold the estimated Q per episode

    # Pre-compute the discount factors / learning rates / epsilons
    discounts = np.logspace( 0, max_steps, num=max_steps, base=gamma, endpoint=False)
    lrs = utils.decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)
    epsilons = utils.decay_schedule(init_eps, min_eps, eps_decay_ratio, n_episodes)


    for e in range(n_episodes):
        eps = epsilons[e]
        lr = lrs[e]

        trajectory = utils.generate_trajectory(Q, eps, env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool)

        for t, (state, action, reward, _, _) in enumerate(trajectory):
            state_has_been_visited = visited[state][action]  # true or false

            if state_has_been_visited and mode == "FV":
                # do not compute anything else if visited and FVMC is applied
                continue
            visited[state][action] = True

            n_steps = len(trajectory[:t])

            future_rewards = trajectory[t:, 2]  # rewards from this point until the end
            G = np.sum(discounts[:n_steps] * future_rewards)
            error = G - Q[state][action]
            Q[state][action] = Q[state][action] + lr * error
        
        # episode completed
        Q_track[e] = Q
        π_track.append(np.argmax(Q, axis=1))
    
    final_Q = Q
    estimated_optimal_V = np.max(final_Q, axis=1)
    greedy_policy_π = lambda s: { s: a for s, a in enumerate(np.argmax(final_Q, axis=1)) }[s]

    return final_Q, estimated_optimal_V, greedy_policy_π, Q_track, π_track











if __name__ == "__main__":
    pass