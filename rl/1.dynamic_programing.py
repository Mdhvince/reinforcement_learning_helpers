from collections import namedtuple

import numpy as np

"""
Here are algorithms to balance short term and long term rewards.
Algorithms for learning from Sequential feedbacks.
"""


class DynamicProgramming:
    """
    Algorithms for finding optimal policy when we have access to the MDP
    (the dynamic of the environment P)
    """
    def __init__(self, π, P, gamma=0.99, theta=1e-10):
        self.π = π
        self.P = P
        self.gamma = gamma
        self.theta = theta

        self.states_size = len(P)
        self.actions_size = len(P[0])


    #### Estimating State-Value functions ####

    def _policy_evaluation(self):
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
        old_V = np.zeros(self.states_size)
        while True:
            V = np.zeros(self.states_size)
            for s in range(self.states_size):  # one sweep is completed when this loop is completed
                action = self.π[s]

                # iterate over possible transitions of "action"
                for t in self.P[s][action]:
                    episode_continue = not t.episode_is_done
                    V[s] += t.prob * (t.r + self.gamma * old_V[t.next_state] * episode_continue)

            if np.max(np.abs(old_V - V)) < self.theta: break
            old_V = V.copy()
        return V  #[v0, v1, v2] value of 3 states
    

    #### -------------------------------------------------------------------------------------- ####

    #### Estimating Action-Value functions ####

    def _policy_improvement(self, V):
        """
        Policy Improvement is known as : Control

        Compute the Q-function, once done,
        Select max action per state thanks to Q : This will be our new policy
        This new policy will be evaluated by the _policy_evaluation().
        Inputs:
            - Value function [v0, v1, v2] value of 3 states
            - Dynamic of the environment P
            - Discount factor gamma
        Output:
            - New policy π
        """
        Q = np.zeros((self.states_size, self.actions_size), dtype=np.float64)

        for s in range(self.states_size):
            for action, transitions_list in self.P[s].items():  # iterate over actions available in a state
                for t in transitions_list:  # iterate over possible transitions of "action"
                    episode_continue = not t.episode_is_done
                    Q[s][action] += t.prob * (t.r + self.gamma * V[t.next_state] * episode_continue)
        
        new_π = {}
        greedy_action_per_state = np.argmax(Q, axis=1)  # a*
        for s in range(self.states_size):
            new_π[s] = greedy_action_per_state[s]
        
        return new_π

    #### -------------------------------------------------------------------------------------- ####

    #### Find Optimal Policies ####

    def policy_iteration(self):
        """
        - Repeat Evaluation and Improvement over and over
        """
        while True:
            V = self._policy_evaluation()
            new_π = self._policy_improvement(V)
            
            if self.π == new_π: break  # no further improvements (no changes in π)
            self.π = new_π
        
        return V, self.π


    def value_iteration(self):
        """
        Policy iteration but without waiting for multiple sweeps of V before improving Policy.
        Part of GPI (Generalized Policy Iteration)
        """

        V = np.zeros(self.states_size)

        while True:
            Q = np.zeros((self.states_size, self.actions_size), dtype=np.float64)

            for s in range(self.states_size):  # one sweep is completed when this loop is completed
                for action, transitions_list in self.P[s].items():  # iterate over actions available in a state
                    for t in transitions_list:  # iterate over possible transitions of "action"
                        episode_continue = not t.episode_is_done
                        Q[s][action] += t.prob * (t.r + self.gamma * V[t.next_state] * episode_continue)

            highest_value_per_state = np.max(Q, axis=1)
            if np.max(np.abs(V - highest_value_per_state)) < self.theta: break

            V = highest_value_per_state
            
            new_π = {}
            greedy_action_per_state = np.argmax(Q, axis=1)
            for s in range(self.states_size):
                new_π[s] = greedy_action_per_state[s]
        
        return V, new_π

    #### -------------------------------------------------------------------------------------- ####





if __name__ == "__main__":
    
    """
    3 states s0, s1, s2 and 2 actions left, right
    p sucess move = .7 / p stay = 0.15 / p backward = 0.15
    """
    actions = [0, 1]  # ["left", "right"]
    states = [0, 1, 2]
    Transition = namedtuple('Transition', ['prob', 'next_state', 'r', 'episode_is_done'])

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

    agent = DynamicProgramming(π, dynamic_P)

    
    V, new_π = agent.policy_iteration()
    print(V, new_π)

    agent2 = DynamicProgramming(π, dynamic_P)

    V, new_π = agent2.value_iteration()
    print(V, new_π)

    

