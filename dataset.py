import torch
from tokenizer import BPETokenizer

# =========================
# Load Corpus
# =========================

corpus_path = r"D:\My_VLM\data\shakespeare_data.txt"

print("Loading corpus text...")
with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()

print(f"Corpus characters: {len(text)}")
print(f"Corpus words:      {len(text.split())}")
print(f"Corpus lines:      {len(lines)}")

# =========================
# Initialize BPE Tokenizer
# =========================

print("Loading BPE Tokenizer...")
tokenizer = BPETokenizer()
vocab_size = tokenizer.vocab_size

# =========================
# Safe Batched Encoding (Line-by-Line)
# =========================

print("Encoding corpus efficiently...")
all_tokens = []

for i, line in enumerate(lines):
    # Re-add the newline character so the model learns sentence structures, 
    # except optionally for the very last empty line block
    line_to_encode = line + "\n" if i < len(lines) - 1 else line
    
    # Encode the individual line chunk
    tokens = tokenizer.encode(line_to_encode)
    all_tokens.extend(tokens)
    
    # Progress indicator so you know it's working
    if (i + 1) % 5000 == 0 or (i + 1) == len(lines):
        print(f"Processed {i + 1}/{len(lines)} lines...")

# Convert the master list directly into a PyTorch LongTensor
all_tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)

# =========================
# Train / Validation Split (On Tokens)
# =========================

# Split the token array directly (80% train, 20% validation)
split_idx = int(0.8 * len(all_tokens_tensor))

train_data = all_tokens_tensor[:split_idx]
val_data = all_tokens_tensor[split_idx:]

# =========================
# Decode Helper Function
# =========================

def decode(ids):
    # Handles tensor inputs or standard lists
    if torch.is_tensor(ids):
        ids = ids.tolist()
    return tokenizer.decode(ids)

# =========================
# Final Debug Summary
# =========================

print("\n=== Data Preparation Complete ===")
print(f"Vocabulary Size:     {vocab_size}")
print(f"Total Tokens:        {len(all_tokens_tensor)}")
print(f"Train Dataset Size:  {len(train_data)} tokens")
print(f"Val Dataset Size:    {len(val_data)} tokens")
print(f"Sample First 20 IDs: {train_data[:20].tolist()}")
print("=================================\n")