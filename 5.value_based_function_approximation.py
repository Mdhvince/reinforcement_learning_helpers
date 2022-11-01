import torch
from torch import nn
import torch.nn.functional as F



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
        

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}\n")

    fcq = FCQ(device, 4, 3, (64, 32, 16), F.relu)
    print(fcq)