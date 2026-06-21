import torch
import torch.nn.functional as F

from final_model import TinyGPT
from dataset import vocab_size
from tokenizer import BPETokenizer

# Hyperparameters
block_size = 32  
max_new_tokens = 60
temperature = 0.8  # Slight bump to push past repetitive loops
top_k = 25         # Expanded search space for cleaner word distributions

# Load Tokenizer
tokenizer = BPETokenizer()

# Hardware Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-instantiate model and load optimal weights
model = TinyGPT(vocab_size).to(device)
model.load_state_dict(torch.load("tinygpt_upscaled_best_10.pth", map_location=device))
model.eval()

# Prompt Processing
start_text = "Before we proceed"
tokens = tokenizer.encode(start_text)

print(f"Using Device: {device}")
print(f"Starting Prompt: '{start_text}'")
print("Generating extensions...\n")

# Seed the output with the initial prompt text
print(start_text, end="", flush=True)

# Cache index values to optimize structural generation
unk_id = tokenizer.token_to_id("<UNK>")

for _ in range(max_new_tokens):
    context_cond = tokens[-block_size:]
    x = torch.tensor([context_cond], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)

    logits = logits[0, -1, :]
    
    # 1. Defend against out-of-vocabulary crashes
    logits[unk_id] = -float('Inf')
    
    # Apply Temperature Scaling
    logits = logits / temperature

    # Top-K Filtering
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[-1]] = -float('Inf')

    # Softmax Sampling
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    
    # 2. Sequential print normalization
    # Decode the current fragment and check if it requires space formatting adjustments
    raw_token_str = tokenizer.itos.get(next_token, "")
    
    # Append token tracking metrics
    tokens.append(next_token)
    
    # Decode the total accumulated array up to this point cleanly to ensure internal markers match
    # This naturally allows your custom tokenizer logic to transform internal boundary symbols into clean spaces
    current_full_text = tokenizer.decode(tokens)
    
    # Print out just the incremental character additions dynamically
    printed_len = len(tokenizer.decode(tokens[:-1]))
    print(current_full_text[printed_len:], end="", flush=True)

print("\n\n" + "-" * 30)
print("Generation Complete.")
print("-" * 30)