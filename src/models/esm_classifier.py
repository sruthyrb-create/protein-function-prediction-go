import torch
import torch.nn as nn

class ESMClassifier(nn.Module):
    def __init__(self, embed_dim=320, num_labels=9494):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_labels)

    def forward(self, x):
        return self.fc(x)
