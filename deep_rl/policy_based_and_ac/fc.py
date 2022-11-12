import warnings ; warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

"""Policy Based

Goal is to maximize the true value function of a parameterized policy from all initial states.
So maximize the true value function by changing the policy (without touching the V-Fucntion)

So we want to find the gradient which will help reaching this objective
From a trajectory τ, we obtain at each step:

>>>> Gt(τ) ▽θ log πθ(At|St)

For the entire trajectory, we just sum over each step from t=0 to T:
>>>> sum( Gt(τ) ▽θ log πθ(At|St) )

So the gradient we are trying to estimate is:

>>>> ▽θ J(θ) = sum( Gt(τ) ▽θ log πθ(At|St) )

In plain words:
- We sample a trajectory
- For each step, we calculate the return from that step
- We use the return from step t to weight the probability of the action taken at step t

That mean if the return is bad at time t, it is because action taken at time t was bad
so by multiplying the bad return with the probability of that action, we reduce the likelihood
of that action being selected at that step.
"""


class FCDAP(nn.Module):  # Fully connected discret action policy
    """
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        """
        super(FCDAP, self).__init__()

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
    
    def full_pass(self, state):
        logits = self.forward(state)  # preferences over actions

        # sample action from the probability distribution
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        log_p_action = dist.log_prob(action).unsqueeze(-1)
        
        # the entropy term encourage having evenly distributed actions
        entropy = dist.entropy().unsqueeze(-1)

        return action.item(), log_p_action, entropy
    
    def select_action(self, state):
        """Helper function for when we just need to sample an action"""
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())



class FCV(nn.Module):  # Fully connected value (state-value)
    """
    """

    def __init__(self, device, in_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        """
        super(FCV, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.out_layer = nn.Linear(hidden_dims[-1], 1)

    def _format(self, x):
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


class FCAC(nn.Module):  # Fully connected actor-critic
    """
    """

    def __init__(self, device, in_dim, out_dim, hidden_dims=(32, 32), activation=F.relu) -> None:
        """
        """
        super(FCAC, self).__init__()

        self.device = device
        self.activation = activation

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        self.value_out_layer = nn.Linear(hidden_dims[-1], 1)
        self.policy_out_layer = nn.Linear(hidden_dims[-1], out_dim)


    def _format(self, x):
        """
        Convert state to tensor if not and shape it correctly for the training process
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x


    def forward(self, state):
        x = self._format(state)
        x = self.activation(self.fc1(x))

        for fc_hidden in self.hidden_layers:
            x = self.activation(fc_hidden(x))
        
        return self.policy_out_layer(x), self.value_out_layer(x)
    

    def full_pass(self, state):
        logits, value = self.forward(state)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        
        # the entropy term encourage having evenly distributed actions
        entropy = dist.entropy().unsqueeze(-1)
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action, logpa, entropy, value


    def select_action(self, state):
        """Helper function for when we just need to sample an action"""
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        action = action.item() if len(action) == 1 else action.data.numpy()
        return action


if __name__ == "__main__":
    pass