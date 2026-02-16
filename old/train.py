#!/usr/bin/env python3
"""
train.py — Train a CARD (Causal Autoregressive Diffusion) Language Model
=========================================================================

Usage:
    # Train a small model on WikiText-2 (good for testing / single GPU):
    python train.py --preset small

    # Train with paper defaults (1B params — needs multi-GPU or large GPU):
    python train.py --preset paper

    # Custom configuration:
    python train.py \
        --n_layers 12 --d_model 768 --n_heads 12 --d_ff 3072 \
        --batch_size 32 --max_steps 50000 --seq_len 128

    # Resume from checkpoint:
    python train.py --resume checkpoints/checkpoint_step_10000.pt

Dataset:
    Uses WikiText-2 or WikiText-103 from HuggingFace datasets.
    WikiText-2:   ~2M tokens   (good for testing)
    WikiText-103: ~100M tokens (more realistic training)

Paper reference:
    Hyperparameter defaults match Table 6 of "Causal Autoregressive Diffusion
    Language Model" (Ruan et al., 2026). The paper trains on 300B tokens from
    FineWeb; WikiText is much smaller but sufficient for validating the approach.
"""

import argparse
import logging
import sys
import os

import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WikiTextDataset(Dataset):
    """
    Loads WikiText from HuggingFace and chunks it into fixed-length sequences.

    The dataset is tokenized with the specified tokenizer and then split into
    contiguous chunks of `seq_len` tokens. This is standard practice for LM
    training — each chunk is an independent training example.

    We add a [MASK] token to the tokenizer's vocabulary (required by CARD).
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 128,
        dataset_name: str = "wikitext-2-raw-v1",
        tokenizer_name: str = "gpt2",
    ):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len

        # Load tokenizer and add [MASK] token
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add [MASK] as a special token — this becomes the last token id
        # which matches CARDModel's convention: mask_token_id = vocab_size - 1
        num_added = self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self.vocab_size = len(self.tokenizer)
        self.mask_token_id = self.tokenizer.mask_token_id

        logging.info(
            f"Tokenizer vocab size: {self.vocab_size} "
            f"(added {num_added} special tokens, [MASK] id = {self.mask_token_id})"
        )

        # Load and tokenize the dataset
        logging.info(f"Loading {dataset_name} ({split} split)...")
        raw_dataset = load_dataset("wikitext", dataset_name, split=split)

        # Concatenate all text and tokenize in one pass
        all_text = "\n".join(
            line for line in raw_dataset["text"] if line.strip()
        )
        tokens = self.tokenizer.encode(all_text)
        logging.info(f"Total tokens in {split}: {len(tokens):,}")

        # Chunk into fixed-length sequences
        # Drop the last incomplete chunk
        n_chunks = len(tokens) // seq_len
        tokens = tokens[: n_chunks * seq_len]
        self.data = torch.tensor(tokens, dtype=torch.long).reshape(n_chunks, seq_len)

        logging.info(f"Created {len(self.data):,} sequences of length {seq_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Train a CARD (Causal Autoregressive Diffusion) Language Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Presets ---
    parser.add_argument(
        "--preset",
        choices=["paper", "medium", "small", "tiny"],
        default=None,
        help=(
            "Model size preset. Overrides architecture args. "
            "'paper' = Table 6 (1B params, needs >>24GB VRAM). "
            "'medium' = 350M params. "
            "'small' = 125M params (fits on 8GB GPU). "
            "'tiny' = 10M params (for debugging)."
        ),
    )

    # --- Model architecture (defaults = Table 6 of the paper) ---
    arch = parser.add_argument_group("Model Architecture")
    arch.add_argument("--n_layers", type=int, default=33, help="Number of transformer layers")
    arch.add_argument("--d_model", type=int, default=1536, help="Hidden dimension")
    arch.add_argument("--n_heads", type=int, default=24, help="Number of attention heads")
    arch.add_argument("--d_ff", type=int, default=4096, help="FFN intermediate dimension")
    arch.add_argument("--max_len", type=int, default=8192, help="Max position embeddings")
    arch.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")

    # --- Training hyperparameters (defaults = Table 6) ---
    train_grp = parser.add_argument_group("Training")
    train_grp.add_argument("--batch_size", type=int, default=64, help="Batch size per step")
    train_grp.add_argument("--seq_len", type=int, default=128, help="Sequence length (Table 6: 128)")
    train_grp.add_argument("--max_steps", type=int, default=100_000, help="Total training steps")
    train_grp.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate")
    train_grp.add_argument("--warmup_steps", type=int, default=2500, help="LR warmup steps")
    train_grp.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    train_grp.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    train_grp.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    train_grp.add_argument("--no_amp", action="store_true", help="Disable mixed precision")

    # --- CARD-specific diffusion hyperparameters ---
    diff_grp = parser.add_argument_group("CARD Diffusion")
    diff_grp.add_argument(
        "--tail_factor", type=float, default=1.5,
        help="λ: soft tail masking window width factor (1.0=strict tail, >1.0=relaxed)"
    )
    diff_grp.add_argument(
        "--reweight_beta", type=float, default=1.0,
        help="β: smoothing constant for context-aware reweighting"
    )
    diff_grp.add_argument(
        "--reweight_decay", type=float, default=0.5,
        help="p: decay factor for noise distance in reweighting"
    )

    # --- Dataset ---
    data_grp = parser.add_argument_group("Dataset")
    data_grp.add_argument(
        "--dataset", type=str, default="wikitext-2-raw-v1",
        choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
        help="WikiText variant to train on"
    )
    data_grp.add_argument(
        "--tokenizer", type=str, default="gpt2",
        help="HuggingFace tokenizer name"
    )
    data_grp.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # --- Logging & checkpointing ---
    io_grp = parser.add_argument_group("Logging & Checkpointing")
    io_grp.add_argument("--output_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    io_grp.add_argument("--log_interval", type=int, default=100, help="Steps between log messages")
    io_grp.add_argument("--eval_interval", type=int, default=1000, help="Steps between evaluations")
    io_grp.add_argument("--save_interval", type=int, default=5000, help="Steps between checkpoints")
    io_grp.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # --- Apply presets (override architecture args) ---
    presets = {
        "paper": dict(n_layers=33, d_model=1536, n_heads=24, d_ff=4096),
        "medium": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096),
        "small": dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072),
        "tiny": dict(n_layers=6, d_model=384, n_heads=6, d_ff=1536),
    }
    if args.preset:
        for key, val in presets[args.preset].items():
            setattr(args, key, val)
        logging.info(f"Applied '{args.preset}' preset: {presets[args.preset]}")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = get_args()

    # Log all arguments
    logging.info("=" * 60)
    logging.info("CARD Training Configuration")
    logging.info("=" * 60)
    for key, val in sorted(vars(args).items()):
        logging.info(f"  {key}: {val}")
    logging.info("=" * 60)

    # --- Load datasets ---
    logging.info("Preparing datasets...")
    train_dataset = WikiTextDataset(
        split="train",
        seq_len=args.seq_len,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
    )
    eval_dataset = WikiTextDataset(
        split="validation",
        seq_len=args.seq_len,
        dataset_name=args.dataset,
        tokenizer_name=args.tokenizer,
    )

    # Verify vocab sizes match
    vocab_size = train_dataset.vocab_size
    assert vocab_size == eval_dataset.vocab_size, "Train/eval vocab mismatch"
    assert train_dataset.mask_token_id == vocab_size - 1, (
        f"[MASK] id ({train_dataset.mask_token_id}) must be vocab_size-1 ({vocab_size-1})"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid partial batches
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logging.info(
        f"Train: {len(train_dataset):,} sequences, "
        f"{len(train_loader):,} batches/epoch"
    )
    logging.info(
        f"Eval:  {len(eval_dataset):,} sequences, "
        f"{len(eval_loader):,} batches/epoch"
    )

    # --- Build model ---
    logging.info("Building CARD model...")
    model = CARDModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    )

    # --- Build trainer ---
    from card_trainer import TrainerConfig, CARDTrainer

    config = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=not args.no_amp,
        tail_factor=args.tail_factor,
        reweight_beta=args.reweight_beta,
        reweight_decay=args.reweight_decay,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
    )

    trainer = CARDTrainer(model, train_loader, eval_loader, config)

    # --- Resume if requested ---
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # --- Train ---
    logging.info("Starting CARD training!")
    trainer.train()

    logging.info("Done.")


if __name__ == "__main__":
    from card_model import CARDModel

    main()
