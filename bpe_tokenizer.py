import json
import re

# =========================
# Load Corpus & Apply Space Hack
# =========================

with open(
    r"D:\My_VLM\data\shakespeare_data.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

# CRITICAL FIX: The Space Hack
# Replace spaces and newlines with a special character '_' attached to the start of words.
# This teaches the tokenizer where words begin and preserves spacing!
text = text.replace(" ", " _").replace("\n", " _\n")

# Ensure the very first word also gets the space marker if it doesn't have one
if not text.startswith("_") and not text.startswith(" _"):
    text = "_" + text

words = text.split()

# =========================
# Build Initial Vocabulary
# =========================

vocab = {}

for word in words:
    chars = " ".join(list(word))
    vocab[chars] = (
        vocab.get(chars, 0)
        + 1
    )

print(
    "Initial Unique Word Layouts:",
    len(vocab)
)

# =========================
# Count Pair Frequencies
# =========================

def get_pair_counts(vocab):
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (
                symbols[i],
                symbols[i + 1]
            )
            pairs[pair] = (
                pairs.get(pair, 0)
                + freq
            )
    return pairs


# =========================
# Merge Pair
# =========================

def merge_pair(pair, vocab):
    new_vocab = {}
    
    p0 = re.escape(pair[0])
    p1 = re.escape(pair[1])
    
    # Regex to safely merge independent tokens
    pattern = re.compile(
        r'(?<!\S)'
        + p0
        + r'\s+'
        + p1
        + r'(?!\S)'
    )
    
    replacement = (
        pair[0]
        + pair[1]
    )
    
    for word, freq in vocab.items():
        new_word = pattern.sub(
            replacement,
            word
        )
        new_vocab[new_word] = freq
        
    return new_vocab


# =========================
# BPE Training Loop
# =========================

# CRITICAL FIX: Increased merges to 2000 to build full words
num_merges = 2000
merge_rules = []

print(f"\nStarting BPE Training with {num_merges} targeted merges...")

for i in range(num_merges):
    pairs = get_pair_counts(vocab)
    
    if not pairs:
        print(
            f"No more pairs at merge {i}"
        )
        break
        
    best_pair = max(
        pairs,
        key=pairs.get
    )
    
    # Clean console: Print the first 10, then every 100th merge
    if i < 10 or (i + 1) % 100 == 0:
        print(
            f"Merge {i+1:04d}: "
            f"{best_pair} "
            f"(Count={pairs[best_pair]})"
        )
        
    merge_rules.append(
        list(best_pair)
    )
    
    vocab = merge_pair(
        best_pair,
        vocab
    )

# =========================
# Extract Final Vocabulary
# =========================

tokens = set()

for word in vocab:
    tokens.update(
        word.split()
    )

# Add UNK token
tokens.add("<UNK>")

tokens = sorted(
    list(tokens)
)

print()
print(
    "Final BPE Vocabulary Size:",
    len(tokens)
)

# =========================
# Save Merge Rules
# =========================

with open(
    "bpe_rules.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        merge_rules,
        f,
        indent=4
    )

print(
    "Saved: bpe_rules.json"
)

# =========================
# Save Vocabulary
# =========================

with open(
    "bpe_vocab.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        tokens,
        f,
        indent=4
    )

print(
    "Saved: bpe_vocab.json"
)

# =========================
# Preview
# =========================

print()
print(
    "First 50 Tokens:"
)
print(
    tokens[:50]
)