import json
import re
import torch


class BPETokenizer:

    def __init__(self, rules_file="bpe_rules.json", vocab_file="bpe_vocab.json"):
        print("Loading BPE Tokenizer...")

        # =========================
        # Load Merge Rules
        # =========================
        with open(rules_file, "r", encoding="utf-8") as f:
            raw_rules = json.load(f)

        # Map each pair to its priority rank (lower index = higher priority)
        self.merge_rules = {tuple(rule): i for i, rule in enumerate(raw_rules)}

        # =========================
        # Load Vocabulary
        # =========================
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_tokens = json.load(f)

        if "<UNK>" not in vocab_tokens:
            vocab_tokens.append("<UNK>")

        self.stoi = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.itos = {idx: token for idx, token in enumerate(vocab_tokens)}
        self.vocab_size = len(self.stoi)

        # Global Word Cache
        self.word_cache = {}
        print(f"Vocabulary Size: {self.vocab_size}")

    # ==================================
    # Apply Learned BPE Merges (Optimized Array Approach)
    # ==================================
    def _tokenize_word(self, word):
        cached = self.word_cache.get(word)
        if cached is not None:
            return cached

        # Start with individual characters
        tokens = list(word)

        while len(tokens) > 1:
            # 1. Find all adjacent pairs currently in the word
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            # 2. Find the pair with the highest priority (lowest rank) in our merge rules
            best_pair = None
            best_rank = float('inf')
            
            for pair in pairs:
                rank = self.merge_rules.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            # If none of the adjacent pairs can be merged, we are done
            if best_pair is None:
                break

            # 3. Merge the chosen pair in the token list
            new_tokens = []
            i = 0
            p0, p1 = best_pair
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                    new_tokens.append(p0 + p1)
                    i += 2  # Skip next since it's merged
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        self.word_cache[word] = tokens
        return tokens

    # ==================================
    # Text -> Token IDs
    # ==================================
    def encode(self, text):
        text = text.replace(" ", " _").replace("\n", " _\n")
        if not text.startswith(" _"):
            text = " _" + text

        token_ids = []
        words = re.findall(r'\S+|\n', text)  # Keep newlines intact explicitly

        for i, word in enumerate(words):
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i:,}/{len(words):,} words | Cache: {len(self.word_cache):,}")

            subwords = self._tokenize_word(word)

            for token in subwords:
                if token in self.stoi:
                    token_ids.append(self.stoi[token])
                else:
                    for char in token:
                        token_ids.append(self.stoi.get(char, self.stoi["<UNK>"]))

        return token_ids  # Returning a list here lets dataset.py extend it flawlessly

    # ==================================
    # Token IDs -> Text
    # ==================================
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = [self.itos[idx] for idx in token_ids if idx in self.itos]
        raw_text = "".join(tokens)
        clean_text = raw_text.replace("_", " ")
        return clean_text.lstrip()

    def token_to_id(self, token):
        return self.stoi.get(token, self.stoi["<UNK>"])

    def id_to_token(self, idx):
        return self.itos.get(idx, "<UNK>")


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    text = "Before we proceed any further"
    print("\nOriginal:", text)
    encoded = tokenizer.encode(text)
    print("\nEncoded IDs:", encoded)
    print("\nDecoded Text:", tokenizer.decode(encoded))