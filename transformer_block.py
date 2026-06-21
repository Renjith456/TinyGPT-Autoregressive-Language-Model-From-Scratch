import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout=0.2):
        super().__init__()
        # Correctly project to head_size instead of full n_embd
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)    # (B, T, head_size)
        Q = self.query(x)  # (B, T, head_size)
        V = self.value(x)  # (B, T, head_size)

        # Compute attention scores (affinity)
        scores = Q @ K.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        scores = scores / (K.size(-1) ** 0.5)  # Scale by sqrt(head_size)

        # Causal Masking (strictly prevent looking into the future)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Convert to probabilities
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)  # Apply dropout to attention map

        # Weighted aggregation of values
        out = weights @ V  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, dropout=0.2):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be perfectly divisible by num_heads!"
        head_size = n_embd // num_heads

        # Pass the calculated head_size to each attention head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, dropout) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate along the channel/embedding dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # Project back and apply dropout
        out = self.proj_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Upgraded to GELU (Standard for GPT variants)
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)  # Regularization inside feed-forward block
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head=4, dropout=0.2):
        super().__init__()
        # Made n_head dynamic to align cleanly with upscaled topologies
        self.attention = MultiHeadAttention(n_embd, n_head, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-LN residual connections
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x