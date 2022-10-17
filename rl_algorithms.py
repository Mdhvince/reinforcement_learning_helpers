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
    Output:
        - Value function
    """
    old_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s, _ in P.items():
            action = π[s]

            for t in P[s][action]:  # iterate over possible transitions of "action"
                episode_continue = not t.episode_is_done
                V[s] += t.prob * (t.reward + gamma * old_V[t.next_state] * episode_continue)
            

        if np.max(np.abs(old_V - V)) < theta: break
        old_V = V.copy()
    return V


def policy_improvement(V, P, gamma):
    """
    Compute the Q-function, once done,
    Select max action per state thanks to Q : This will be our new policy
    This new policy will be evaluated by the policy_evaluation().
    Inputs:
        - Value function
        - Dynamic of the environment P
        - Discount factor gamma
    Output:
        - New policy π
    """
    states_size = len(P)
    actions_size = len(P[0])
    Q = np.zeros((states_size, actions_size), dtype=np.float64)

    for s in range(states_size):
        for action, transitions_list in P[s].items():  # iterate over actions available in a state
            for t in transitions_list:  # iterate over possible transitions of "action"
                episode_continue = not t.episode_is_done
                Q[s][action] += t.prob * (t.reward + gamma * V[t.next_state] * episode_continue)
    
    new_π = {}
    greedy_action_per_state = np.argmax(Q, axis=1)
    for s in range(states_size):
        new_π[s] = greedy_action_per_state[s]
    
    return new_π










if __name__ == "__main__":
    
    """
    3 states s0, s1, s2 and 2 actions left, right
    p sucess move = .7 / p stay = 0.15 / p backward = 0.15
    """
    actions = [0, 1]  # ["left", "right"]
    states = [0, 1, 2]
    Transition = namedtuple('Transition', ['prob', 'next_state', 'reward', 'episode_is_done'])

    # Define the always (for every states) "left" policy
    π = {}
    for s in states:
        π[s] = np.random.choice(actions, p=[1.0, 0.0])
    
    
    dynamic_P = {
        0: {
            0: [Transition(1.0, 0, 0, True)], 1: [Transition(1.0, 0, 0, 1)]
        },
        1: {
            0: [
                Transition(.7, 0, 0, 1), Transition(.15, 1, 0, 0), Transition(.15, 2, 1, 1)
            ],
            1: [
                Transition(.7, 2, 1, 1), Transition(.15, 1, 0, 0), Transition(.15, 0, 0, 1)
            ]
        },
        2: {
            0: [Transition(1.0, 2, 0, 1)], 1: [Transition(1.0, 2, 0, 1)]
        }
    }
    
    V = policy_evaluation(π, dynamic_P, gamma=.99, theta=1e-10)
    new_π = policy_improvement(V, dynamic_P, gamma=.99)



