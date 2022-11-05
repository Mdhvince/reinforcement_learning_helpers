import warnings ; warnings.filterwarnings('ignore')

import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        - in_dim: state dimention as input (if state composed of [x, y, z] location, in_dim=3)
        - out_dim: number of action (will output the q(s, a) for all actions)
        - hidden_dims: (32, 32, 16) will create 3 hidden layers of 32, 32, 16 units.
        - activation: activation function
        """
        super(DQN, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.out_layer = nn.Linear(hidden_dims[-1], out_dim)
        self.to(self.device)

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


class DDQN(nn.Module):
    """
    Dueling Deep Q-Network Architechture.
    Same as DQN but using 2 outputs : One for the State-value function (return a single number)
    and one for the Action-advantage function (return the advantage value of each actions)
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        super(DDQN, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.state_value_output = nn.Linear(hidden_dims[-1], 1)
        self.advantage_value_output = nn.Linear(hidden_dims[-1], out_dim)
        self.to(self.device)

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
        
        advantage = self.advantage_value_output(x)
        state_value = self.state_value_output(x)

        # we could stop here and then use an agent specific to DDQN. But here I'm gonna reconstruct
        # the Action-value function Q, so I can use the Agent from DQN, then I will recover A and V
        # from Q.

        # A = Q - V  --> Q = V + A
        # But once we have Q, we cannot recover uniquely V and A...
        # To address this we will substract the mean of A from Q, this will shift A and V by a
        # constant and stabilize the optim process

        # expand the scalar to the same size as advantage so we can add them up to recreate Q
        # because a = q - v so q = v + a
        state_value = state_value.expand_as(advantage)

        q = state_value + advantage - advantage.mean(1, keepdim=True).expand_as(advantage)
        
        return q



if __name__ == "__main__":
   pass
                
    
