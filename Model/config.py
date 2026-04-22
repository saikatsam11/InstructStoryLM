from dataclasses import dataclass

@dataclass
class ModelConfig:
    # ── Architecture (fixed) ──────────────────────────────────────────────────
    vocab_size   : int   = 32_000
    d_model      : int   = 448        # head_dim = 448 / 7 = 64
    n_heads      : int   = 7
    n_layers     : int   = 7
    d_ff         : int   = 1792       # 4 × d_model
    context_len  : int   = 256
    dropout      : float = 0.1

    # ── 4060 Ti 16GB tuned settings ───────────────────────────────────────────
    #
    #   VRAM budget:
    #     Model weights (bf16)   ~0.06 GB
    #     Optimizer states(fp32) ~0.50 GB   (AdamW m + v)
    #     Activations + grads    ~1.50 GB   (batch=128, ctx=256, 7 layers)
    #     PyTorch overhead       ~0.50 GB
    #     ─────────────────────  ────────
    #     Total estimated        ~2.60 GB   (out of 16 GB — very safe)
    #
    #   Effective batch = 128 x 4 = 512 sequences = 131,072 tokens/step
    #
    batch_size   : int   = 128
    grad_accum   : int   = 4          # effective batch = 512 seqs
    num_workers  : int   = 4

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr           : float = 6e-4
    weight_decay : float = 0.1
    beta1        : float = 0.9
    beta2        : float = 0.95
    grad_clip    : float = 1.0

    # ── LR Schedule ───────────────────────────────────────────────────────────
    #   330M tokens / 131K per step = 2517 steps/epoch
    #   2 epochs = 5034 steps → rounded to 5100
    warmup_steps : int   = 200
    max_steps    : int   = 5_100      # 2 full epochs

    # ── Logging & saving ──────────────────────────────────────────────────────
    log_every    : int   = 50
    val_every    : int   = 500
    save_every   : int   = 500

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir      : str  = "data/bin"
    ckpt_dir      : str  = "checkpoints"
    tokenizer_path: str  = "tokenizer/tokenizer.json"

    # ── torch.compile: ~20% faster on Ada Lovelace (4060 Ti) ─────────────────
    compile_model : bool = True
