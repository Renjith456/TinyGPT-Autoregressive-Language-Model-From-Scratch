import torch
import torch.nn as nn

class TinyGPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            16
        )

    def forward(self, x):

        x = self.embedding(x)

        return x