import torch
import torch.nn.functional as F
import math

print("Step 1: Importing scaled dataset...")
from dataset import (
    train_data,
    val_data,
    vocab_size
)
print("Step 2: Dataset imported successfully")

print("Step 3: Importing model...")
from final_model import TinyGPT
print("Step 4: Model imported successfully")

# ==========================================
# STRATEGY 1: SCALE-UP HYPERPARAMETERS
# ==========================================
batch_size = 64        # Larger batch size to stabilize gradients over bigger data spans
block_size = 256       # Quadrupled window context size (from 64 to 256 words/tokens)
max_steps = 40000      # Extended training iterations for big data
learning_rate = 6e-4   
min_lr = 6e-5          
warmup_iters = 1000    # Gradual warmup ramp to handle large batch sizes smoothly

# Hardware Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n=== Hardware Diagnostics ===")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Details:   ", torch.cuda.get_device_name(0))
print("Using Device:  ", device)
print("============================\n")

print("Train Token Length:", len(train_data))
print("Val Token Length:  ", len(val_data))

# Dataset Guardrails
if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError("Dataset parsing error: Token counts are smaller than requested block_size!")

print("Vocab Size Target:", vocab_size)

print("Step 5: Creating model...")
# Passing explicitly upscaled layer arguments to override old defaults
model = TinyGPT(vocab_size, n_embd=256, n_layer=6, max_position=512).to(device)
print("Step 6: Model created successfully (Running in Eager Mode)")

# Regularized AdamW configuration
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.05   # Forces parameter weights down to inhibit rote memorization
)
print("Step 7: Optimizer created")


# Learning Rate Scheduler Logic (Cosine Decay with Warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_steps - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)


# Batched Vector Generator
def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))
    
    x_batch = torch.stack([source[i.item() : i.item() + block_size] for i in ix]).to(device)
    y_batch = torch.stack([source[i.item() + 1 : i.item() + block_size + 1] for i in ix]).to(device)
    return x_batch, y_batch


# Validation Evaluation
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        total_loss = 0
        eval_iters = 40  # Sample more iterations for accurate metrics across big data
        
        for _ in range(eval_iters):
            x, y = get_batch(split)
            logits = model(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), y.reshape(B * T))
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses


print("Step 8: Starting upscaled big-data training loop...")
best_val_loss = float('inf')

for step in range(max_steps):
    # Dynamically update the step learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x_batch, y_batch = get_batch("train")

    # Forward
    logits = model(x_batch)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, C), y_batch.reshape(B * T))

    # Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Clip gradients to ensure structural stability in deep layers
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Evaluation Periodic Checkpoint Trigger
    if step % 250 == 0:
        losses = estimate_loss()
        current_val_loss = losses['val']
        
        print(
            f"Step {step:5d} | LR: {lr:.6f} | "
            f"Train Loss = {losses['train']:.4f} | "
            f"Val Loss = {current_val_loss:.4f}"
        )
        
        # Save snapshot dynamically when validation hits a new record low
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), "tinygpt_scaled_best_11.pth")
            print(f"  --> [SAVED LOG] New peak performance achieved! (Val Loss: {best_val_loss:.4f})")

print("Step 9: Saving final run snapshot...")
torch.save(model.state_dict(), "tinygpt_final_11.pth")
print("Training Complete! Scaled weights are securely archived as 'tinygpt_scaled_best.pth'.")