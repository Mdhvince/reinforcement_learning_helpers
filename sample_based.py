import random
import numpy as np


def pure_exploitation(env, n_episodes=5000):
    """The pure exploitation strategy"""

    Q = np.zeros((env.action_space.n))  # Q-function
    N = np.zeros((env.action_space.n))  # Keep track of nb times an action has been selected
    
    # stats variables (optional)
    Q_episode = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)
    
    for e in range(n_episodes):
        action = np.argmax(Q)  # choose action with highest value
        _, reward, _, _ = env.step(action)  # apply action and observe the env transition (we do not know the p of the transition)
        
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]
        
        # stats
        Q_episode[e] = Q
        returns[e] = reward
        actions[e] = action

    return returns, Q_episode, actions


def epsilon_greedy(env, epsilon=.01, n_episodes=5000):
    """Greedy most of the time with prob p and explore with prop 1-p = epsilon"""

    Q = np.zeros((env.action_space.n))  # Q-function
    N = np.zeros((env.action_space.n))  # Keep track of nb times an action has been selected

    # stats variables (optional)
    Q_episode = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    for e in range(n_episodes):

        if np.random.random() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        _, reward, _, _ = env.step(action) 

        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]

        # stats
        Q_episode[e] = Q
        returns[e] = reward
        actions[e] = action
    
    return returns, Q_episode, actions



def linear_decay_epsilon_greedy(
    env, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.05, n_episodes=5000):
    """
    Start  exploring then decay epsilon at each step.
    Linearly means decaying linearly with the number of steps
    """

    Q = np.zeros((env.action_space.n))  # Q-function
    N = np.zeros((env.action_space.n))  # Keep track of nb times an action has been selected

    # stats variables (optional)
    Q_episode = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    for e in range(n_episodes):
        epsilon = _linear_decay(n_episodes, e, decay_ratio, init_epsilon, min_epsilon)

        if np.random.random() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        _, reward, _, _ = env.step(action) 

        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]

        # stats
        Q_episode[e] = Q
        returns[e] = reward
        actions[e] = action
    
    return returns, Q_episode, actions


def exponential_decay_epsilon_greedy(
    env, init_epsilon=1.0, min_epsilon=0.01, decay_ratio=0.05, n_episodes=5000):
    """
    Start  exploring then decay epsilon at each step.
    Linearly means decaying linearly with the number of steps
    """

    Q = np.zeros((env.action_space.n))  # Q-function
    N = np.zeros((env.action_space.n))  # Keep track of nb times an action has been selected

    # stats variables (optional)
    Q_episode = np.empty((n_episodes, env.action_space.n))
    returns = np.empty(n_episodes)
    actions = np.empty(n_episodes, dtype=np.int)

    epsilons = _exponential_decay(n_episodes, decay_ratio, init_epsilon, min_epsilon)
    
    for e in range(n_episodes):
        epsilon = epsilons[e]

        if np.random.random() > epsilon:
            action = np.argmax(Q)
        else:
            action = np.random.randint(len(Q))

        _, reward, _, _ = env.step(action) 

        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]

        # stats
        Q_episode[e] = Q
        returns[e] = reward
        actions[e] = action
    
    return returns, Q_episode, actions


def _exponential_decay(n_episodes, decay_ratio, init_epsilon, min_epsilon):
    """Precompute the list of epsilon we will use (decayed exponentially)"""
    decay_episodes = int(n_episodes * decay_ratio)
    rem_episodes = n_episodes - decay_episodes
    epsilons = 0.01
    epsilons /= np.logspace(-2, 0, decay_episodes)
    epsilons *= init_epsilon - min_epsilon
    epsilons += min_epsilon
    epsilons = np.pad(epsilons, (0, rem_episodes), 'edge')
    return epsilons


def _linear_decay(n_episodes, current_episode, decay_ratio, init_epsilon, min_epsilon):
    """Calculate epsilon value for the current episode"""
    decay_episodes = n_episodes * decay_ratio
    epsilon = 1 - current_episode / decay_episodes
    epsilon *= init_epsilon - min_epsilon 
    epsilon += min_epsilon
    epsilon = np.clip(epsilon, min_epsilon, init_epsilon)
    return epsilon



if __name__ == "__main__":
    pass