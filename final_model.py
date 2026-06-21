import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class TinyGPT(nn.Module):

    def __init__(self, vocab_size, n_embd=256, n_layer=6, max_position=512):
        super().__init__()
        
        # Save structural dimensions
        self.max_position = max_position

        # 1. Expanded Token Vector Space
        self.embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        # 2. Scaled Positional Memory Matrix
        self.position_embedding = nn.Embedding(
            max_position,
            n_embd
        )

        # 3. Deeper Stack: Generates 6 layers instead of 3
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd) for _ in range(n_layer)
        ])

        # 4. Layer Norm Layer: Crucial stabilizer before mapping to logits
        self.ln_f = nn.LayerNorm(n_embd)

        # 5. Language Model Language Mapping Head
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )
        
        # Parameter Initialization Hook for deeper networks
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x shape: (B, T)
        B, T = x.shape
        
        # Runtime Guardrail: Protect against sequence overflows
        if T > self.max_position:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max block size is {self.max_position}"
            )

        token_emb = self.embedding(x)  # Shape: (B, T, n_embd)

        # Generate positional indices dynamically relative to current device host
        positions = torch.arange(
            T,
            device=x.device
        )
        pos_emb = self.position_embedding(positions)  # Shape: (T, n_embd)

        # Combine embeddings via broadcasting
        x = token_emb + pos_emb  # Shape: (B, T, n_embd)

        # Process through the deep Transformer blocks stack
        x = self.blocks(x)
        
        # Normalize activations before projection
        x = self.ln_f(x)

        # Compute vocabulary score distributions
        logits = self.lm_head(x)  # Shape: (B, T, vocab_size)

        return logits