import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

    def forward(self, x):

        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        weights = F.softmax(scores, dim=-1)

        out = weights @ V

        return out