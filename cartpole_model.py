import torch.nn as nn


class CartPoleModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(CartPoleModel, self).__init__()

        self.fc = net_Qvalue = nn.Sequential(
            nn.Linear(input_shape[0], 32), 
            nn.ReLU(),
            nn.Linear(32, n_actions)
        )
    
    def forward(self, x):
        return self.fc(x)