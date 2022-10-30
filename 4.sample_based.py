from collections import namedtuple
from collections import defaultdict

import numpy as np

import utils



"""
Sample Based Methods
"""

# Solving Prediction problems
def temporal_difference(π, env, gamma=1.0, init_lr=0.5, min_lr=0.01, lr_decay_ratio=0.3, n_episodes=500):
    """
    Here we can update V as we go, no need to first generate a trajectory.
    Solving the prediction problem
    """
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    lrs = utils.decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)

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
    lrs = utils.decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)
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


# Solving control problems
def sarsa(
    π, env, gamma=1.,
    init_lr=.5, min_lr=.01, lr_decay_ratio=.3,
    init_eps=1., min_eps=.1, eps_decay_ratio=.9, n_episodes=500):
    """
    Solving the control problem
    """
    nA = env.action_space.n                # number of actions
    nS = env.action_space.n                # number of states
    π_track = []                           # Hold the improved (greedy) policy per episode

    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)  # hold the estimated Q per episode

    lrs = utils.decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)
    epsilons = utils.decay_schedule(init_eps, min_eps, eps_decay_ratio, n_episodes)
    
    
    for i_episode in range(1, n_episodes+1): 
        state, done = env.reset(), False                       # at this point, we have S _ _ _ _
        
        eps = epsilons[i_episode]
        lr = lrs[i_episode]
        action = utils.epsilon_greedy(state, Q, eps)                # at this point, we have S A _ _ _
        
        while not done:
            next_state, reward, done, _ = env.step(action)     # at this point, we have S A R S' _
            next_action = utils.epsilon_greedy(next_state, Q, eps)  # at this point, we have S A R S' A'

            # Update Q
            sarsa_experience = (state, action, reward, next_state, next_action)
            Q[state][action] = utils.update_sarsa(Q, sarsa_experience, gamma, lr, False)

            state = next_state
            action = next_action
        
        # episode completed
        Q_track[i_episode] = Q
        π_track.append(np.argmax(Q, axis=1))

    final_Q = Q
    estimated_optimal_V = np.max(final_Q, axis=1)
    greedy_policy_π = lambda s: { s: a for s, a in enumerate(np.argmax(final_Q, axis=1)) }[s]

    return final_Q, estimated_optimal_V, greedy_policy_π, Q_track, π_track


def q_learning(π, env, gamma=1.0, init_lr=0.5, min_lr=0.01, lr_decay_ratio=0.3, plot_every=100, n_episodes=500):
    """
    Value iteration alike. It directly learn the optimal value function, no need for policy
    improvement step as opposed to sarsa, because it is directly taking the max action of the
    target target policy instead of the next action.
    Because q learning if acting in a different way (greedily) over the next state than the 
    behavior policy (rhs of the error term), Q-learning is an Off Policy.
    Solving the control problem
    """
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    lrs = utils.decay_schedule(init_lr, min_lr, lr_decay_ratio, n_episodes)
    
    
    for i_episode in range(1, n_episodes+1): 
        state = env.reset()
        action = π(state)
        
        while True:
            next_state, reward, done, _ = env.step(action)

            if not done:
                next_action = π(next_state)

                # Update Q
                sarsa_experience = (state, action, reward, next_state, next_action)
                Q[state][action] = utils.update_sarsa(Q, sarsa_experience, gamma, lrs[i_episode], True)

                state = next_state
                action = next_action

            if done: break
    return Q







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
