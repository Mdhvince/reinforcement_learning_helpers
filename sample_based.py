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


if __name__ == "__main__":
    pass