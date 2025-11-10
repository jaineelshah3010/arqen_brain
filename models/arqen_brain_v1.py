import torch.nn as nn

class ArqenBrain(nn.Module):
    def __init__(self, input_dim, hidden_size=128, output_size=1):
        super(ArqenBrain, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )

    def forward(self, x):
        return self.layers(x)