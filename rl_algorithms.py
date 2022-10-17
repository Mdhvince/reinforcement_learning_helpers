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
    states_size = len(P)
    old_V = np.zeros(states_size)
    while True:
        V = np.zeros(states_size)
        for s in range(states_size):  # one sweep is completed when this loop is completed
            action = π[s]

            for t in P[s][action]:  # iterate over possible transitions of "action"
                episode_continue = not t.episode_is_done
                V[s] += t.prob * (t.reward + gamma * old_V[t.next_state] * episode_continue)
            

        if np.max(np.abs(old_V - V)) < theta: break
        old_V = V.copy()
    return V


def policy_improvement(V, P, gamma):
    """
    Policy Improvement is known as : Control

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


def policy_iteration(π, P, gamma=0.99, theta=1e-10):
    """
    - Repeat Evaluation and Improvement over and over
    """
    while True:
        V = policy_evaluation(π, P, gamma, theta)
        new_π = policy_improvement(V, P, gamma)
        
        if π == new_π: break  # no further improvements (no changes in π)
        π = new_π
    
    return V, new_π


def value_iteration(P, gamma=.99, theta=1e-10):
    """
    Policy iteration but without waiting for multiple sweeps of V before improving Policy.
    Part of GPI (Generalized Policy Iteration)
    """
    states_size = len(P)
    actions_size = len(P[0])

    V = np.zeros(states_size)

    while True:
        Q = np.zeros((states_size, actions_size), dtype=np.float64)

        for s in range(states_size):  # one sweep is completed when this loop is completed
            for action, transitions_list in P[s].items():  # iterate over actions available in a state
                for t in transitions_list:  # iterate over possible transitions of "action"
                    episode_continue = not t.episode_is_done
                    Q[s][action] += t.prob * (t.reward + gamma * V[t.next_state] * episode_continue)

        highest_value_per_state = np.max(Q, axis=1)
        if np.max(np.abs(V - highest_value_per_state)) < theta: break

        V = highest_value_per_state
        
        new_π = {}
        greedy_action_per_state = np.argmax(Q, axis=1)
        for s in range(states_size):
            new_π[s] = greedy_action_per_state[s]
    
    return V, new_π







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
    
    V, new_π = policy_iteration(π, dynamic_P, gamma=0.99, theta=1e-10)
    V, new_π = value_iteration(dynamic_P, gamma=.99, theta=1e-10)

