#!/usr/bin/env python3
"""
train.py — Train CARD or AR language models on WikiText
========================================================

Both models use the EXACT SAME CausalLM architecture. The only difference
is the training procedure:
  AR:   clean input → next-token prediction
  CARD: noisy input → next-token prediction + context-aware reweighting

Usage:
    # Train CARD (small, for testing):
    python train.py --mode card --preset small

    # Train AR baseline (same architecture):
    python train.py --mode ar --preset small

    # Paper config (needs large GPU):
    python train.py --mode card --preset paper --dataset wikitext-103-raw-v1

    # Custom:
    python train.py --mode card --n_layers 12 --d_model 768 --batch_size 32

    # Resume:
    python train.py --mode card --preset small --resume checkpoints_card/card_best.pt

Presets:
    tiny   —  ~10M params   (debug, any GPU)
    small  — ~124M params   (8GB GPU)
    medium — ~350M params   (24GB GPU)
    paper  —  ~1B params    (40GB+ GPU, Table 6)
"""

import argparse
import logging
import sys
import os

import torch
from torch.utils.data import DataLoader, Dataset
import neptune


# =============================================================================
# Dataset
# =============================================================================

class WikiTextDataset(Dataset):
    """
    Load WikiText from HuggingFace, tokenize, chunk into fixed-length sequences.

    The tokenizer vocabulary is extended with a [MASK] token as the last id.
    AR training ignores it; CARD training uses it for corruption.
    """

    def __init__(self, split: str, seq_len: int, dataset_name: str, tokenizer_name: str):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add [MASK] as last token — must be vocab_size - 1 for model convention
        self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self.vocab_size = len(self.tokenizer)

        logging.info(f"Vocab: {self.vocab_size} tokens ([MASK] = {self.tokenizer.mask_token_id})")

        # Load and tokenize
        raw = load_dataset("wikitext", dataset_name, split=split)
        text = "\n".join(line for line in raw["text"] if line.strip())
        tokens = self.tokenizer.encode(text)
        logging.info(f"{split}: {len(tokens):,} tokens total")

        # Chunk into sequences (drop remainder)
        n = len(tokens) // seq_len
        self.data = torch.tensor(tokens[:n * seq_len], dtype=torch.long).reshape(n, seq_len)
        logging.info(f"{split}: {len(self.data):,} sequences of length {seq_len}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# =============================================================================
# CLI
# =============================================================================

PRESETS = {
    "paper":  dict(n_layers=33, d_model=1536, n_heads=24, d_ff=4096),  # ~1B (Table 6)
    "medium": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096),  # ~350M
    "small":  dict(n_layers=12, d_model=768,  n_heads=12, d_ff=3072),  # ~124M
    "tiny":   dict(n_layers=6,  d_model=384,  n_heads=6,  d_ff=1536),  # ~10M
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Train CARD or AR language model on WikiText",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--mode", choices=["card", "ar"], required=True,
                   help="Training mode: 'card' (diffusion) or 'ar' (autoregressive)")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Model size preset (overrides architecture args)")

    # Architecture
    g = p.add_argument_group("Architecture")
    g.add_argument("--n_layers", type=int, default=12)
    g.add_argument("--d_model", type=int, default=768)
    g.add_argument("--n_heads", type=int, default=12)
    g.add_argument("--d_ff", type=int, default=3072)
    g.add_argument("--max_len", type=int, default=8192)
    g.add_argument("--dropout", type=float, default=0.0)

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--seq_len", type=int, default=128, help="Training sequence length (Table 6: 128)")
    g.add_argument("--max_steps", type=int, default=100_000)
    g.add_argument("--learning_rate", type=float, default=3e-4)
    g.add_argument("--warmup_steps", type=int, default=2500)
    g.add_argument("--weight_decay", type=float, default=0.1)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    g.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    g.add_argument("--compile", action="store_true", help="torch.compile the model")

    # CARD-specific
    g = p.add_argument_group("CARD Diffusion")
    g.add_argument("--tail_factor", type=float, default=1.5, help="λ: tail masking window")
    g.add_argument("--reweight_beta", type=float, default=1.0, help="β: reweighting smoothing")
    g.add_argument("--reweight_decay", type=float, default=0.5, help="p: distance decay")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--dataset", default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    g.add_argument("--tokenizer", default="gpt2")
    g.add_argument("--num_workers", type=int, default=4)

    # I/O
    g = p.add_argument_group("I/O")
    g.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory (default: checkpoints_{mode})")
    g.add_argument("--log_interval", type=int, default=100)
    g.add_argument("--eval_interval", type=int, default=1000)
    g.add_argument("--save_interval", type=int, default=5000)
    g.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device to use for training (cuda, cpu, or specific device like cuda:0)")
    g.add_argument("--no-log", action="store_true", help="Disable Neptune logging")

    args = p.parse_args()

    # Apply preset
    if args.preset:
        for k, v in PRESETS[args.preset].items():
            setattr(args, k, v)

    # Default output dir
    if args.output_dir is None:
        args.output_dir = f"./checkpoints_{args.mode}"

    return args


# =============================================================================
# Neptune Logging
# =============================================================================

def init_neptune_run(no_log: bool = False, project: str = "halcyon/card", token_file: str = ".neptune_tok"):
    """
    Initialize a Neptune run for experiment tracking.

    Args:
        no_log: If True, skip Neptune initialization
        project: Neptune project name (workspace/project)
        token_file: Path to file containing Neptune API token (first line)

    Returns:
        Neptune run object, or None if initialization fails or disabled
    """
    if no_log:
        logging.info("Neptune logging disabled via --no-log flag")
        return None

    try:
        # Read API token from file
        if not os.path.exists(token_file):
            logging.warning(f"Neptune token file '{token_file}' not found. Skipping Neptune logging.")
            return None

        with open(token_file, 'r') as f:
            api_token = f.readline().strip()

        # Initialize Neptune run
        run = neptune.init_run(
            project=project,
            api_token=api_token,
        )
        logging.info(f"Neptune run initialized: {run.get_url()}")
        return run

    except Exception as e:
        logging.warning(f"Failed to initialize Neptune: {e}. Continuing without Neptune logging.")
        return None


def log_parameters_to_neptune(run, args):
    """
    Log training parameters to Neptune.

    Args:
        run: Neptune run object (can be None)
        args: Parsed arguments from argparse
    """
    if run is None:
        return

    try:
        # Convert args to dict and log all parameters
        params = vars(args)
        run["parameters"] = params
        logging.info("Parameters logged to Neptune")
    except Exception as e:
        logging.warning(f"Failed to log parameters to Neptune: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()

    # Initialize Neptune for experiment tracking
    neptune_run = init_neptune_run(no_log=args.no_log)
    log_parameters_to_neptune(neptune_run, args)

    logging.info("=" * 60)
    logging.info(f"{'CARD' if args.mode == 'card' else 'AR'} Training")
    logging.info("=" * 60)
    for k, v in sorted(vars(args).items()):
        logging.info(f"  {k}: {v}")
    logging.info("=" * 60)

    # --- Data ---
    train_ds = WikiTextDataset("train", args.seq_len, args.dataset, args.tokenizer)
    eval_ds = WikiTextDataset("validation", args.seq_len, args.dataset, args.tokenizer)
    vocab_size = train_ds.vocab_size

    # pin_memory should only be True when using CUDA
    use_pin_memory = args.device.startswith("cuda")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin_memory, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin_memory,
    )

    # --- Model (shared architecture) ---
    from model import CausalLM, ModelConfig

    model_config = ModelConfig(
        vocab_size=vocab_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
    )
    model = CausalLM(model_config)
    logging.info(f"Model: {model.count_parameters():,} parameters")

    # Optional torch.compile for faster training (PyTorch 2.0+)
    if args.compile:
        logging.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # --- Trainer ---
    if args.mode == "card":
        from card_trainer import CARDTrainer, CARDTrainerConfig
        trainer_config = CARDTrainerConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            use_amp=not args.no_amp,
            tail_factor=args.tail_factor,
            reweight_beta=args.reweight_beta,
            reweight_decay=args.reweight_decay,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            device=args.device,
        )
        trainer = CARDTrainer(model, train_loader, eval_loader, trainer_config, neptune_run=neptune_run)
    else:
        from ar_trainer import ARTrainer, ARTrainerConfig
        trainer_config = ARTrainerConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            use_amp=not args.no_amp,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            device=args.device,
        )
        trainer = ARTrainer(model, train_loader, eval_loader, trainer_config, neptune_run=neptune_run)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    # Stop Neptune run
    if neptune_run is not None:
        neptune_run.stop()
        logging.info("Neptune run stopped")

    logging.info("Done.")


if __name__ == "__main__":
    main()
