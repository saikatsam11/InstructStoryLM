"""
Evaluates the trained GPT model on N generated stories vs real TinyStories.

Metrics computed:
    - Perplexity (PPL)
    - BLEU-1, BLEU-2, BLEU-3, BLEU-4
    - ROUGE-1, ROUGE-2, ROUGE-L
    - BERTScore (Precision, Recall, F1)
    - MAUVE
    - Distinct-1, Distinct-2
    - Repetition rate (4-gram)
    - Average story length (tokens)
    - Vocabulary coverage

Usage:
    python evaluate.py --ckpt_best.pt --n_stories 500

"""

import os
import re
import csv
import math
import argparse
import torch
import numpy as np
from collections        import Counter
from tokenizers         import Tokenizer
from datasets           import load_dataset
from nltk.translate.bleu_score  import corpus_bleu, SmoothingFunction
from rouge_score        import rouge_scorer as rouge_scorer_lib

from Model.config import ModelConfig
from Model.gpt    import GPT


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                          if k in ModelConfig.__dataclass_fields__})
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step     = ckpt.get("step", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"  Loaded checkpoint: step={step}  val_loss={val_loss:.4f}  ppl={math.exp(val_loss):.2f}")
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Generate one story
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_story(model, tokenizer, prompt, cfg, device,
                   max_tokens=200, temperature=0.8, top_p=0.9):
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    enc      = tokenizer.encode(prompt)
    seed_ids = [bos_id] + enc.ids
    idx      = torch.tensor([seed_ids], device=device)

    out = model.generate(idx, max_new_tokens=max_tokens,
                         temperature=temperature, top_p=top_p, eos_id=eos_id)

    token_ids = [t for t in out[0].tolist() if t not in (bos_id, eos_id)]

    # Decode and clean ByteLevel BPE artifacts
    text = tokenizer.decode(token_ids)
    text = text.replace("Ġ", " ").replace("Ċ", "\n").strip()
    return text, token_ids


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def tokenize_words(text):
    """Simple word tokenisation for metric computation."""
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())


def compute_bleu(generated_texts, reference_texts):
    """
    Corpus-level BLEU-1,2,3,4.
    Each generated story is compared against its corresponding reference.
    """
    smoothie = SmoothingFunction().method1
    refs, hyps = [], []
    for gen, ref in zip(generated_texts, reference_texts):
        hyps.append(tokenize_words(gen))
        refs.append([tokenize_words(ref)])      # list of references per hypothesis

    scores = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        scores[f"BLEU-{n}"] = corpus_bleu(refs, hyps,
                                           weights=weights,
                                           smoothing_function=smoothie)
    return scores


def compute_rouge(generated_texts, reference_texts):
    """ROUGE-1, ROUGE-2, ROUGE-L averaged across all story pairs."""
    scorer  = rouge_scorer_lib.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                            use_stemmer=True)
    r1, r2, rl = [], [], []
    for gen, ref in zip(generated_texts, reference_texts):
        s = scorer.score(ref, gen)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return {
        "ROUGE-1": np.mean(r1),
        "ROUGE-2": np.mean(r2),
        "ROUGE-L": np.mean(rl),
    }


def compute_bertscore(generated_texts, reference_texts, device):
    """
    BERTScore Precision, Recall, F1 averaged across all story pairs.

    Uses microsoft/deberta-xlarge-mnli by default (state-of-the-art rescaled
    variant). Falls back to bert-base-uncased if the large model cannot be
    loaded (e.g., no internet access or GPU memory constraints).

    Args:
        generated_texts : list[str]  – model outputs
        reference_texts : list[str]  – ground-truth stories
        device          : str        – "cuda" or "cpu"

    Returns:
        dict with keys "BERTScore-P", "BERTScore-R", "BERTScore-F1"
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError(
            "bert-score is not installed. Run:  pip install bert-score"
        )

    print("  Computing BERTScore (this may take a while on CPU) ...")

    P, R, F1 = bert_score_fn(
        cands=generated_texts,
        refs=reference_texts,
        lang="en",
        rescale_with_baseline=False,
        device=device,
        verbose=False,
    )

    return {
        "BERTScore-P":  P.mean().item(),
        "BERTScore-R":  R.mean().item(),
        "BERTScore-F1": F1.mean().item(),
    }


def compute_mauve(generated_texts, reference_texts, device, max_text_length=256):
    """
    MAUVE score – measures the distribution-level similarity between
    generated text and reference text using divergence frontiers in the
    feature space of a pre-trained LM (GPT-2 by default).

    A higher MAUVE score (closer to 1) indicates the generated distribution
    is closer to the human-written reference distribution.

    Args:
        generated_texts : list[str]  – model outputs
        reference_texts : list[str]  – ground-truth stories
        device          : str        – "cuda" or "cpu"
        max_text_length : int        – truncation length for the featuriser

    Returns:
        dict with key "MAUVE"
    """
    try:
        import mauve as mauve_lib
    except ImportError:
        raise ImportError(
            "mauve-text is not installed. Run:  pip install mauve-text"
        )

    print("  Computing MAUVE (featurising texts with GPT-2, may take a while) ...")

    device_id = 0 if (device == "cuda" and torch.cuda.is_available()) else -1

    out = mauve_lib.compute_mauve(
        p_text=reference_texts,       # "p" = reference / human distribution
        q_text=generated_texts,       # "q" = model distribution
        device_id=device_id,
        max_text_length=max_text_length,
        verbose=False,
        featurize_model_name="gpt2",  # lightweight; swap for "gpt2-large" if GPU available
    )

    return {"MAUVE": out.mauve}


def compute_distinct(generated_texts):
    """
    Distinct-1 and Distinct-2 across all generated stories combined.
    Measures diversity: fraction of unique unigrams / bigrams.
    Higher = more diverse vocabulary.
    """
    all_words  = []
    all_bigrams = []
    for text in generated_texts:
        words = tokenize_words(text)
        all_words.extend(words)
        all_bigrams.extend(zip(words[:-1], words[1:]))

    d1 = len(set(all_words))  / max(len(all_words),  1)
    d2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    return {"Distinct-1": d1, "Distinct-2": d2}


def compute_repetition(token_ids_list, n=4):
    """
    Average 4-gram repetition rate across all generated stories.
    Lower = less repetition.
    """
    rates = []
    for token_ids in token_ids_list:
        if len(token_ids) < n:
            rates.append(0.0); continue
        ngrams  = [tuple(token_ids[i:i+n]) for i in range(len(token_ids) - n + 1)]
        counts  = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        rates.append(repeated / len(ngrams))
    return np.mean(rates)


def compute_vocab_coverage(generated_texts, tokenizer):
    """
    Fraction of the full BPE vocabulary used across all generated stories.
    """
    vocab_size  = tokenizer.get_vocab_size()
    used_tokens = set()
    for text in generated_texts:
        enc = tokenizer.encode(text)
        used_tokens.update(enc.ids)
    return len(used_tokens) / vocab_size


def avg_story_length(token_ids_list):
    return np.mean([len(t) for t in token_ids_list])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs("logs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  Evaluation Script")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  N stories  : {args.n_stories}")
    print("=" * 60 + "\n")

    # ── Load model + tokeniser ────────────────────────────────────────────────
    model, cfg = load_model(args.ckpt, device)
    tokenizer  = Tokenizer.from_file(cfg.tokenizer_path)
    print(f"  Tokeniser  : {cfg.tokenizer_path}  (vocab={tokenizer.get_vocab_size():,})\n")

    # ── Load real TinyStories as reference ────────────────────────────────────
    print("  Loading TinyStories references ...")
    ds         = load_dataset("roneneldan/TinyStories", split="train",
                               trust_remote_code=True)
    # Use stories from the held-out 25% (indices after 75%)
    start_idx  = int(0.75 * len(ds))
    ref_pool   = ds.select(range(start_idx, min(start_idx + args.n_stories * 2, len(ds))))
    ref_texts  = ref_pool["text"][:args.n_stories]
    print(f"  References : {len(ref_texts)} stories from held-out 25%\n")

    # ── Extract prompts from reference stories (first sentence as prompt) ─────
    def extract_prompt(text):
        # Use first few words as the prompt
        words = text.strip().split()[:6]
        return " ".join(words)

    prompts = [extract_prompt(r) for r in ref_texts]

    # ── Generate N stories ────────────────────────────────────────────────────
    print(f"  Generating {args.n_stories} stories ...")
    generated_texts   = []
    generated_tok_ids = []

    for i, prompt in enumerate(prompts):
        text, tok_ids = generate_story(
            model, tokenizer, prompt, cfg, device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generated_texts.append(text)
        generated_tok_ids.append(tok_ids)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/{args.n_stories} stories ...", end="\r")

    print(f"  Generated {args.n_stories}/{args.n_stories} stories ✓        \n")

    # ── Print sample ──────────────────────────────────────────────────────────
    print("  Sample generated story:")
    print("  " + "─" * 50)
    print(f"  Prompt : {prompts[0]}")
    print(f"  Story  : {generated_texts[0][:300]}...")
    print("  " + "─" * 50 + "\n")

    # ── Compute all metrics ───────────────────────────────────────────────────
    print("  Computing metrics ...")

    val_loss = torch.load(args.ckpt, map_location="cpu").get("val_loss", float("nan"))
    ppl      = math.exp(val_loss)

    bleu_scores   = compute_bleu(generated_texts, ref_texts)
    rouge_scores  = compute_rouge(generated_texts, ref_texts)
    distinct      = compute_distinct(generated_texts)
    rep_rate      = compute_repetition(generated_tok_ids, n=4)
    avg_len       = avg_story_length(generated_tok_ids)
    vocab_cov     = compute_vocab_coverage(generated_texts, tokenizer)

    # BERTScore — gracefully skip if library missing
    if args.skip_bertscore:
        bert_scores = {"BERTScore-P": float("nan"),
                       "BERTScore-R": float("nan"),
                       "BERTScore-F1": float("nan")}
        print("  BERTScore : skipped (--skip_bertscore flag set)")
    else:
        try:
            bert_scores = compute_bertscore(generated_texts, ref_texts, device)
        except ImportError as e:
            bert_scores = {"BERTScore-P": float("nan"),
                           "BERTScore-R": float("nan"),
                           "BERTScore-F1": float("nan")}
            print(f"  BERTScore : skipped ({e})")

    # MAUVE — gracefully skip if library missing
    if args.skip_mauve:
        mauve_scores = {"MAUVE": float("nan")}
        print("  MAUVE     : skipped (--skip_mauve flag set)")
    else:
        try:
            mauve_scores = compute_mauve(
                generated_texts, ref_texts, device,
                max_text_length=args.mauve_max_len,
            )
        except ImportError as e:
            mauve_scores = {"MAUVE": float("nan")}
            print(f"  MAUVE     : skipped ({e})")

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Evaluation Report  ({args.n_stories} stories)")
    print("=" * 55)
    print(f"  {'Perplexity (PPL)':<28} {ppl:>10.4f}")
    print("  " + "─" * 40)
    print(f"  {'BLEU-1':<28} {bleu_scores['BLEU-1']:>10.4f}")
    print(f"  {'BLEU-2':<28} {bleu_scores['BLEU-2']:>10.4f}")
    print(f"  {'BLEU-3':<28} {bleu_scores['BLEU-3']:>10.4f}")
    print(f"  {'BLEU-4':<28} {bleu_scores['BLEU-4']:>10.4f}")
    print("  " + "─" * 40)
    print(f"  {'ROUGE-1':<28} {rouge_scores['ROUGE-1']:>10.4f}")
    print(f"  {'ROUGE-2':<28} {rouge_scores['ROUGE-2']:>10.4f}")
    print(f"  {'ROUGE-L':<28} {rouge_scores['ROUGE-L']:>10.4f}")
    print("  " + "─" * 40)

    # BERTScore block
    bs_p  = bert_scores["BERTScore-P"]
    bs_r  = bert_scores["BERTScore-R"]
    bs_f1 = bert_scores["BERTScore-F1"]
    fmt   = lambda v: f"{v:>10.4f}" if not math.isnan(v) else f"{'N/A':>10}"
    print(f"  {'BERTScore-Precision':<28} {fmt(bs_p)}")
    print(f"  {'BERTScore-Recall':<28} {fmt(bs_r)}")
    print(f"  {'BERTScore-F1':<28} {fmt(bs_f1)}")
    print("  " + "─" * 40)

    # MAUVE block
    mv = mauve_scores["MAUVE"]
    print(f"  {'MAUVE':<28} {fmt(mv)}")
    print("  " + "─" * 40)

    print(f"  {'Distinct-1':<28} {distinct['Distinct-1']:>10.4f}")
    print(f"  {'Distinct-2':<28} {distinct['Distinct-2']:>10.4f}")
    print("  " + "─" * 40)
    print(f"  {'Repetition rate (4-gram)':<28} {rep_rate:>10.4f}")
    print(f"  {'Avg story length (tokens)':<28} {avg_len:>10.1f}")
    print(f"  {'Vocabulary coverage':<28} {vocab_cov*100:>9.2f}%")
    print("=" * 55)

    # ── Save to CSV ───────────────────────────────────────────────────────────
    csv_path = "logs/eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Perplexity",              round(ppl, 4)])
        writer.writerow(["BLEU-1",                  round(bleu_scores["BLEU-1"], 4)])
        writer.writerow(["BLEU-2",                  round(bleu_scores["BLEU-2"], 4)])
        writer.writerow(["BLEU-3",                  round(bleu_scores["BLEU-3"], 4)])
        writer.writerow(["BLEU-4",                  round(bleu_scores["BLEU-4"], 4)])
        writer.writerow(["ROUGE-1",                 round(rouge_scores["ROUGE-1"], 4)])
        writer.writerow(["ROUGE-2",                 round(rouge_scores["ROUGE-2"], 4)])
        writer.writerow(["ROUGE-L",                 round(rouge_scores["ROUGE-L"], 4)])
        writer.writerow(["BERTScore-P",             round(bs_p,  4) if not math.isnan(bs_p)  else "N/A"])
        writer.writerow(["BERTScore-R",             round(bs_r,  4) if not math.isnan(bs_r)  else "N/A"])
        writer.writerow(["BERTScore-F1",            round(bs_f1, 4) if not math.isnan(bs_f1) else "N/A"])
        writer.writerow(["MAUVE",                   round(mv,    4) if not math.isnan(mv)    else "N/A"])
        writer.writerow(["Distinct-1",              round(distinct["Distinct-1"], 4)])
        writer.writerow(["Distinct-2",              round(distinct["Distinct-2"], 4)])
        writer.writerow(["Repetition_rate_4gram",   round(float(rep_rate), 4)])
        writer.writerow(["Avg_story_length_tokens", round(float(avg_len), 1)])
        writer.writerow(["Vocab_coverage_pct",      round(vocab_cov * 100, 2)])

    print(f"\n  Results saved → {csv_path}")

    # ── Save all generated stories ────────────────────────────────────────────
    stories_path = "logs/generated_stories.txt"
    with open(stories_path, "w") as f:
        for i, (prompt, story) in enumerate(zip(prompts, generated_texts)):
            f.write(f"=== Story {i+1} ===\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"{story}\n\n")

    print(f"  Stories saved  → {stories_path}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",           required=True,            help="Path to checkpoint")
    p.add_argument("--n_stories",      type=int,   default=100,  help="Number of stories to evaluate")
    p.add_argument("--max_tokens",     type=int,   default=200,  help="Max tokens per story")
    p.add_argument("--temperature",    type=float, default=0.8,  help="Sampling temperature")
    p.add_argument("--top_p",          type=float, default=0.9,  help="Nucleus sampling top-p")
    # BERTScore / MAUVE toggles — handy when the libraries aren't installed
    p.add_argument("--skip_bertscore", action="store_true",      help="Skip BERTScore computation")
    p.add_argument("--skip_mauve",     action="store_true",      help="Skip MAUVE computation")
    p.add_argument("--mauve_max_len",  type=int,   default=256,  help="Max token length for MAUVE featuriser")
    main(p.parse_args())