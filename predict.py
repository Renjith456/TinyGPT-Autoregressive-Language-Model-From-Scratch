import torch
import torch.nn.functional as F

from final_model import TinyGPT
from dataset import vocab_size, stoi, itos

# Load Model

model = TinyGPT(vocab_size)

model.load_state_dict(
    torch.load("tinygpt6.pth")
)

model.eval()

# Input Text

text = "machin"

tokens = [stoi[c] for c in text]

# Add batch dimension
# Shape: (1, T)
x = torch.tensor([tokens])

# Prediction

with torch.no_grad():
    logits = model(x)

# Get logits for:
# Batch 0
# Last position
last_logits = logits[0, -1]

probs = F.softmax(
    last_logits,
    dim=-1
)

top_probs, top_indices = torch.topk(
    probs,
    5
)

# Display Results

print("Input:", text)
print()

for p, idx in zip(top_probs, top_indices):

    print(
        f"{itos[idx.item()]} : {p.item():.4f}"
    )