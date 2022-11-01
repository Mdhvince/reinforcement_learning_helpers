import gc
from pathlib import Path
from itertools import count

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from action_selection import EGreedyStrategy, GreedyStrategy



class FCQ(nn.Module):
    """
    Fully connected Q-Function (To estimate action-values)
    We input the state and output the q(s, a) of each action in that state
    (state-in-values-out architechture)
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        - in_dim: state dimention as input (if state composed of [x, y, z] location, in_dim=3)
        - out_dim: number of action (will output the q(s, a) for all actions)
        - hidden_dims: (32, 32, 16) will create 3 hidden layers of 32, 32, 16 units.
        - activation: activation function
        """
        super(FCQ, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)

    def _format(self, x):
        """
        Convert state to tensor if not and shape it correctly for the training process
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))
        
        x = self.out_layer(x)
        return x
    
    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable
    
    def format_experiences(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals
        

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    env = None
    nS, nA = None, None
    lr = 0.0005
    batch_size = 1024
    epochs = 40
    gamma = 0.99
    model_dir = Path("models")

    net = FCQ(device, nS, nA, hidden_dims=(512, 128))
    optimizer = optim.RMSprop(net.parameters(), lr=lr)

    interaction_strategy = EGreedyStrategy(epsilon=.5)
    evaluation_strategy = GreedyStrategy()

    max_episodes = 500
    experiences = []
    episode_reward = []
    episode_timestep = []
    episode_exploration = []
    evaluation_scores = []

    for i_episode in range(1, max_episodes + 1):
        state, is_terminal = env.reset(), False
        episode_reward.append(0.0)
        episode_timestep.append(0.0)
        episode_exploration.append(0.0)

        for step in count(): # inifinte loop but with an index (step)

            # ---> interact with the env
            action = interaction_strategy.select_action(net, state)
            new_state, reward, is_terminal, info = env.step(action)
            is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
            is_failure = is_terminal and not is_truncated
            experience = (state, action, reward, new_state, float(is_failure))

            # Store experience
            experiences.append(experience)

            # For metrics
            episode_reward[-1] += reward
            episode_timestep[-1] += 1
            episode_exploration[-1] += int(interaction_strategy.exploratory_action_taken)

            if len(experiences) >= batch_size:
                experiences = np.array(experiences)
                batches = [np.vstack(sars) for sars in experiences.T]
                experiences = net.format_experiences(batches)

                # train
                for _ in range(epochs):
                    # working with batche of stateS / actionS / rewardS / is_terminalS
                    states, actions, rewards, next_states, is_terminals = experiences
                    batch_size = len(is_terminals)
                    
                    # target = expected discounted return at S+1
                    # here we do not want to compute gradient on the target obviously
                    # what we are updating is our current estimate
                    max_q_next = net(next_states).detach().max(1)[0].unsqueeze(1)
                    targets = rewards + gamma * max_q_next * (1 - is_terminals)

                    # get all the current value estimate of this state-action pair
                    q_hat_sa = net(states).gather(1, actions)

                    # the "true" value of q(s, a) is the expected return at s+1
                    # so we want q_hat_sa close to q(s, a) = Expected G_t+1 = targets
                    errors = q_hat_sa - targets
                    loss = errors.pow(2).mul(0.5).mean()  # mean squared error
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                experiences.clear()
            
            if is_terminal:
                gc.collect()  # garbage collection
                break
        
        # evaluate after one episode
        R = []
        n_episodes = 50
        for _ in range(n_episodes):
            state, done = env.reset(), False
            R.append(0)
            for _ in count():
                a = evaluation_strategy.select_action(net, state)
                state, reward, done, _ = env.step(a)
                R[-1] += reward
                if done: break

        mean_reward_eval = np.mean(R)
        ts = episode_timestep[-1]
        evaluation_scores.append(mean_reward_eval)

        print(f"Episode {i_episode}")
        print(f"#Timesteps done during training = {ts}")
        print(f"Total reward of the episode during training = {episode_reward[-1]}")
        print("*********************************************************************")
        print(f"Mean eval reward = {mean_reward_eval}\n")

        if mean_reward_eval >= max(evaluation_scores):
            torch.save(net.state_dict(), model_dir / f"model.{i_episode}.pt")
    
    print('\nTraining complete.')