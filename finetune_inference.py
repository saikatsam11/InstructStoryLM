"""
inference.py — Interactive generation with the fine-tuned GPT model
────────────────────────────────────────────────────────────────────
Usage:
    python inference.py
    
"""

import argparse, sys, time, re
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from Model.config import ModelConfig
from Model.gpt import GPT


DEFAULT_CKPT = "ft_ckpt_best.pt"


def extract_name(instruction: str, inp: str) -> str | None:
    """Extract a character name from 'named X' or 'name is X' in instruction or input."""
    for text in (instruction, inp):
        m = re.search(r'\bnamed\s+([A-Za-z]+)', text, re.IGNORECASE) or \
            re.search(r'\bname\s+is\s+([A-Za-z]+)', text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
    return None


def build_prompt(instruction: str, inp: str = "") -> tuple[str, str]:
    """
    Returns (prompt, primer).
    primer is pre-filled response text to force the model to use the correct name.
    """
    if inp.strip():
        base   = instruction.rstrip(".!?")
        merged = f"{base}. Details: {inp}"
        prompt = f"### Instruction:\n{merged}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    # Prime the response with the character name so the model doesn't substitute its own
    name   = extract_name(instruction, inp)
    primer = f"Once upon a time, there was a character named {name}. " if name else ""
    return prompt, primer


def load_model(ckpt_path: str, device: str) -> GPT:
    cfg   = ModelConfig()
    model = GPT(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded ← {ckpt_path}")
    if "val_loss" in ckpt:
        print(f"  val_loss : {ckpt['val_loss']:.4f}  |  step : {ckpt.get('step', '?')}")
    return model


@torch.no_grad()
def stream_generate(model: GPT, tokenizer: Tokenizer, prompt: str,
                    max_new_tokens: int, temperature: float, top_p: float,
                    device: str, primer: str = ""):
    """Yield decoded text one token at a time for streaming output."""
    cfg    = model.cfg
    eos_id = tokenizer.token_to_id("<|eos|>")
    bos_id = tokenizer.token_to_id("<|bos|>")

    # Encode prompt + primer together so the model continues from the primed text
    full_prompt = prompt + primer
    prompt_ids  = [bos_id] + tokenizer.encode(full_prompt).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # Yield the primer immediately so the user sees it before generation starts
    if primer:
        yield primer

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -cfg.context_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature          # [1, V]

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs     = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        mask      = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0]  = False
        sorted_logits[mask] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(1, sorted_indices, sorted_logits)
        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)   # [1, 1]

        if next_tok.item() == eos_id:
            break

        idx = torch.cat([idx, next_tok], dim=1)

        # Decode just this one token
        token_str = tokenizer.decode([next_tok.item()])
        yield token_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        default=DEFAULT_CKPT)
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p",       type=float, default=0.9)
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
    model     = load_model(args.ckpt, device)

    print(f"\n  Device      : {device}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Temperature : {args.temperature}  |  top_p : {args.top_p}")
    print("\nEnter Instruction, Input (optional), then adjust generation params.")
    print("Press Enter on any param to keep the shown default. Type :quit to exit.\n")

    temperature = args.temperature
    top_p       = args.top_p
    max_tokens  = args.max_tokens

    while True:
        print("─" * 50)

        def read_lines(label):
            """Read multi-line input until blank line. Returns stripped string."""
            lines = []
            try:
                while True:
                    line = input(f"{label} >>> " if not lines else f"{label} ... ")
                    if line.strip() == "":
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                raise KeyboardInterrupt
            return "\n".join(lines).strip()

        try:
            instruction = read_lines("Instruction")
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if not instruction:
            continue

        # Commands
        if instruction.startswith(":"):
            parts = instruction.split()
            cmd   = parts[0]
            if cmd == ":quit":
                print("Bye."); break
            elif cmd == ":temp" and len(parts) == 2:
                temperature = float(parts[1])
                print(f"  temperature → {temperature}")
            elif cmd == ":top_p" and len(parts) == 2:
                top_p = float(parts[1])
                print(f"  top_p → {top_p}")
            elif cmd == ":tokens" and len(parts) == 2:
                max_tokens = int(parts[1])
                print(f"  max_tokens → {max_tokens}")
            else:
                print(f"  Unknown command: {cmd}")
            continue

        try:
            inp = read_lines("Input    ")   # optional — leave blank to skip
        except KeyboardInterrupt:
            print("\nBye.")
            break

        def read_param(label, default, cast):
            """Read a single-line parameter with a shown default. Empty = keep default."""
            try:
                raw = input(f"{label} [{default}] >>> ").strip()
                return cast(raw) if raw else default
            except (ValueError, EOFError, KeyboardInterrupt):
                return default

        temperature = read_param("Temperature", temperature, float)
        top_p       = read_param("Top-p      ", top_p,       float)
        max_tokens  = read_param("Max tokens ", max_tokens,  int)

        prompt, primer = build_prompt(instruction=instruction, inp=inp)
        print()
        t0    = time.perf_counter()
        n_tok = 0
        for token_str in stream_generate(model, tokenizer, prompt,
                                         max_new_tokens=max_tokens,
                                         temperature=temperature,
                                         top_p=top_p,
                                         primer=primer,
                                         device=device):
            print(token_str, end="", flush=True)
            n_tok += 1

        elapsed = time.perf_counter() - t0
        print(f"\n\n  [{n_tok} tokens | {n_tok/elapsed:.0f} tok/s]\n")


if __name__ == "__main__":
    main()
