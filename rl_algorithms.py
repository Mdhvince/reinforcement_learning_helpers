from collections import namedtuple

import numpy as np


"""
Algorithms for finding optimal policy when we have access to the MDP
(the dynamic of the environment P)
"""

def policy_evaluation(π, P, gamma, theta=1e-10):
    """
    Evaluating an arbitrary policy π by computing its V-function (state-value function)

    Inputs:
        - Policy π
        - Dynamic of the environment P
        - Discount factor gamma
        - Treshold theta to stop iterating if not enough changes
    """
    old_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s, _ in P.items():
            action = π[s]

            for t in P[s][action]:  # iterate over transitions
                episode_continue = not t.episode_is_done
                V[s] += t.prob * (t.reward + gamma * old_V[t.next_state] * episode_continue)
            

        if np.max(np.abs(old_V - V)) < theta:
            break
        old_V = V.copy()
    return V


if __name__ == "__main__":
    
    """
    3 states s0, s1, s2 and 2 actions left, right
    p sucess move = .7 / p stay = 0.15 / p backward = 0.15
    """
    actions = ["left", "right"]
    states = [0, 1]
    Transition = namedtuple('Transition', ['prob', 'next_state', 'reward', 'episode_is_done'])

    
    π = {
        0: np.random.choice(actions, p=[1.0, 0.0]),  # always select action left
        1: np.random.choice(actions, p=[1.0, 0.0]),  # always select action left
        2: np.random.choice(actions, p=[1.0, 0.0])   # always select action left
    }
    
    dynamic_P = {
        0: {
            "left": [Transition(1.0, 0, 0, True)], "right": [Transition(1.0, 0, 0, 1)]
        },
        1: {
            "left": [
                Transition(.7, 0, 0, 1),
                Transition(.15, 1, 0, 0),
                Transition(.15, 2, 1, 1)
            ],
            "right": [
                Transition(.7, 2, 1, 1),
                Transition(.15, 1, 0, 0),
                Transition(.15, 0, 0, 1)
            ]
        },
        2: {
            "left": [Transition(1.0, 2, 0, 1)], "right": [Transition(1.0, 2, 0, 1)]
        }
    }
    
    print(policy_evaluation(π, dynamic_P, gamma=.99, theta=1e-10))


