"""
setup_data.py
──────────────
Run this ONCE before training. It:
  1. Loads 75% of TinyStories from HuggingFace (+ 5% held-out val shard)
  2. Trains a BPE tokeniser (vocab=32K) and saves it
  3. Tokenises all stories → packed .bin shards ready for training
     Each story is wrapped as:  [BOS] [tokens...] [EOS]

Usage:
    python setup_data.py

Outputs:
    tokenizer/tokenizer.json        ← saved tokeniser (needed at inference too)
    data/bin/train_shard_XXXX.bin   ← uint16 token ID train shards
    data/bin/val_shard_XXXX.bin     ← uint16 token ID val shards
"""

import os
import numpy as np
from datasets   import load_dataset
from tokenizers import Tokenizer
from tokenizers.models          import BPE
from tokenizers.trainers        import BpeTrainer
from tokenizers.pre_tokenizers  import ByteLevel
from tokenizers.normalizers     import NFC
from tokenizers.decoders        import ByteLevel as ByteLevelDecoder   # FIX 1: add decoder
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_SIZE     = 32_000
SHARD_SIZE     = 10_000_000           # tokens per shard (~20 MB at uint16)
TOKENIZER_OUT  = "tokenizer/tokenizer.json"
DATA_OUT_DIR   = "data/bin"

# FIX 2: pad=0 so it's framework-friendly; order is intentional
SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
#                   ID 0        ID 1       ID 2       ID 3


# ── Step 1: Load dataset ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading TinyStories (75% train + 5% val)")
print("=" * 60)

ds      = load_dataset("roneneldan/TinyStories", split="train", trust_remote_code=True)
n_total = len(ds)
n_train = int(0.75 * n_total)
n_val   = int(0.05 * n_total)          # FIX 3: held-out validation split

train_texts = ds.select(range(n_train))["text"]
val_texts   = ds.select(range(n_train, n_train + n_val))["text"]

print(f"  Train stories : {len(train_texts):,}")
print(f"  Val   stories : {len(val_texts):,}")


# ── Step 2: Train BPE tokeniser ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Training BPE tokeniser")
print("=" * 60)

os.makedirs(os.path.dirname(TOKENIZER_OUT), exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer.normalizer    = NFC()
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder       = ByteLevelDecoder()   # FIX 1: enables correct decoding

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,
    show_progress=True,
)
# Train only on train split — no val data leakage
tokenizer.train_from_iterator(train_texts, trainer=trainer, length=len(train_texts))
tokenizer.save(TOKENIZER_OUT)

print(f"\n  Tokeniser saved → {TOKENIZER_OUT}")
print(f"  Actual vocab size : {tokenizer.get_vocab_size():,}")

# Sanity check
sample = tokenizer.encode("Once upon a time there was a little girl.")
print(f"  Sample tokens : {sample.tokens}")

PAD_ID = tokenizer.token_to_id("<|pad|>")
BOS_ID = tokenizer.token_to_id("<|bos|>")
EOS_ID = tokenizer.token_to_id("<|eos|>")
UNK_ID = tokenizer.token_to_id("<|unk|>")
print(f"  PAD={PAD_ID}  BOS={BOS_ID}  EOS={EOS_ID}  UNK={UNK_ID}")

# Verify decode round-trip
decoded = tokenizer.decode(sample.ids)
print(f"  Decoded       : {decoded}")


# ── Step 3: Tokenise → .bin shards ───────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Tokenising stories → .bin shards")
print("=" * 60)

os.makedirs(DATA_OUT_DIR, exist_ok=True)

def flush_shard(buf: list, idx: int, prefix: str) -> tuple[int, list]:
    arr  = np.array(buf, dtype=np.uint16)
    path = os.path.join(DATA_OUT_DIR, f"{prefix}_shard_{idx:04d}.bin")
    arr.tofile(path)
    print(f"  [{prefix}] Shard {idx:04d}: {len(arr):,} tokens → {path}")
    return idx + 1, []


def tokenise_and_shard(texts: list[str], prefix: str) -> dict:
    """Encode texts into packed BOS+tokens+EOS shards."""
    shard_idx    = 0
    token_buf    = []
    total_tokens = 0
    BATCH        = 5_000

    for i in tqdm(range(0, len(texts), BATCH), desc=f"Encoding [{prefix}]"):
        batch     = texts[i : i + BATCH]
        encodings = tokenizer.encode_batch(batch)

        for enc in encodings:
            # FIX 4: wrap every story with BOS ... EOS
            story_tokens = [BOS_ID] + enc.ids + [EOS_ID]
            token_buf.extend(story_tokens)
            total_tokens += len(story_tokens)

            if len(token_buf) >= SHARD_SIZE:
                shard_idx, token_buf = flush_shard(token_buf, shard_idx, prefix)

    # Flush remaining tokens
    if token_buf:
        shard_idx, _ = flush_shard(token_buf, shard_idx, prefix)

    return {"shards": shard_idx, "tokens": total_tokens}


print("\n── Train shards ──")
train_stats = tokenise_and_shard(train_texts, prefix="train")

print("\n── Val shards ──")
val_stats   = tokenise_and_shard(val_texts,   prefix="val")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  Train shards : {train_stats['shards']}   "
      f"({train_stats['tokens']/1e6:.1f}M tokens)")
print(f"  Val   shards : {val_stats['shards']}   "
      f"({val_stats['tokens']/1e6:.1f}M tokens)")
print(f"  Data dir     : {DATA_OUT_DIR}/")
print(f"  Tokeniser    : {TOKENIZER_OUT}")
print(f"\nAll set! Run  python train.py  to start pretraining.")