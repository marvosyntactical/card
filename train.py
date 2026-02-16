#!/usr/bin/env python3
"""
train.py — Train CARD, AR, or DREAM language models
=====================================================

Three training modes sharing the SAME CausalLM architecture:
  AR:    clean input → next-token prediction
  CARD:  noisy input → next-token prediction + context-aware reweighting
  DREAM: continuous thought in embedding space (CoCoNut-style), on AR or CARD

Usage:
    # Train CARD with cosine noise schedule:
    python train.py --mode card --preset small --noise_schedule cosine

    # Train AR baseline:
    python train.py --mode ar --preset small

    # Train DREAM on AR base with synthetic reasoning data:
    python train.py --mode dream --dream_base ar --preset tiny \\
        --dream_dataset coin_flip --dream_thought_steps 3

    # Train DREAM on CARD base:
    python train.py --mode dream --dream_base card --preset small \\
        --dream_dataset last_letter --noise_schedule cosine

    # Paper config:
    python train.py --mode card --preset paper --dataset wikitext-103-raw-v1

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

        # Add [MASK] as last token
        self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self.vocab_size = len(self.tokenizer)

        logging.info(f"Vocab: {self.vocab_size} tokens ([MASK] = {self.tokenizer.mask_token_id})")

        raw = load_dataset("wikitext", dataset_name, split=split)
        text = "\n".join(line for line in raw["text"] if line.strip())
        tokens = self.tokenizer.encode(text)
        logging.info(f"{split}: {len(tokens):,} tokens total")

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
    "paper":  dict(n_layers=33, d_model=1536, n_heads=24, d_ff=4096),
    "medium": dict(n_layers=24, d_model=1024, n_heads=16, d_ff=4096),
    "small":  dict(n_layers=12, d_model=768,  n_heads=12, d_ff=3072),
    "tiny":   dict(n_layers=6,  d_model=384,  n_heads=6,  d_ff=1536),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Train CARD, AR, or DREAM language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument("--mode", choices=["card", "ar", "dream"], required=True,
                   help="Training mode: 'card' (diffusion), 'ar' (autoregressive), 'dream' (continuous thought)")
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
    g.add_argument("--seq_len", type=int, default=128)
    g.add_argument("--max_steps", type=int, default=100_000)
    g.add_argument("--learning_rate", type=float, default=3e-4)
    g.add_argument("--warmup_steps", type=int, default=2500)
    g.add_argument("--weight_decay", type=float, default=0.1)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--grad_accum", type=int, default=1)
    g.add_argument("--no_amp", action="store_true")
    g.add_argument("--compile", action="store_true")

    # CARD-specific
    g = p.add_argument_group("CARD Diffusion")
    g.add_argument("--tail_factor", type=float, default=1.5, help="λ: tail masking window")
    g.add_argument("--reweight_beta", type=float, default=1.0, help="β: reweighting smoothing")
    g.add_argument("--reweight_decay", type=float, default=0.5, help="p: distance decay")
    g.add_argument("--noise_schedule", type=str, default="linear",
                   choices=["linear", "cosine"],
                   help="Noise schedule: 'linear' (paper default) or 'cosine' (DLM literature)")

    # CARD decoding strategy (for eval/generation)
    g.add_argument("--decoding_strategy", type=str, default="threshold",
                   choices=["threshold", "entropy", "adaptive", "speculative"],
                   help="CARD decoding strategy for generation")

    # CARD denoising visualization
    g.add_argument("--denoise_vis_samples", type=int, default=3,
                   help="Number of examples to visualize during eval (0 to disable)")
    g.add_argument("--denoise_vis_steps", type=int, default=8,
                   help="Denoising steps for visualization")
    g.add_argument("--denoise_vis_block_size", type=int, default=16,
                   help="Block size for denoising visualization")

    # DREAM-specific
    g = p.add_argument_group("DREAM (Continuous Thought)")
    g.add_argument("--dream_base", type=str, default="ar", choices=["ar", "card"],
                   help="Base training mode for DREAM: 'ar' or 'card'")
    g.add_argument("--dream_thought_steps", type=int, default=3,
                   help="Number of continuous thought iterations per position")
    g.add_argument("--dream_thought_loss_weight", type=float, default=0.0,
                   help="Weight for intermediate thought losses (0=pure, 0.1-0.5=curriculum)")
    g.add_argument("--dream_projection", type=str, default="linear",
                   choices=["none", "linear", "mlp"],
                   help="DREAM hidden→embed projection type")
    g.add_argument("--dream_dataset", type=str, default=None,
                   choices=["wikitext", "coin_flip", "last_letter", "gsm8k"],
                   help="Dataset for DREAM training. 'coin_flip' and 'last_letter' are "
                        "synthetic reasoning tasks. 'gsm8k' requires HuggingFace download. "
                        "None defaults to wikitext.")
    g.add_argument("--dream_num_samples", type=int, default=10000,
                   help="Number of synthetic samples for coin_flip/last_letter datasets")
    g.add_argument("--dream_max_flips", type=int, default=8,
                   help="Max flip operations for coin_flip dataset")
    g.add_argument("--dream_max_words", type=int, default=6,
                   help="Max words for last_letter dataset")

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--dataset", default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    g.add_argument("--tokenizer", default="gpt2")
    g.add_argument("--num_workers", type=int, default=4)

    # I/O
    g = p.add_argument_group("I/O")
    g.add_argument("--output_dir", type=str, default=None)
    g.add_argument("--log_interval", type=int, default=100)
    g.add_argument("--eval_interval", type=int, default=1000)
    g.add_argument("--save_interval", type=int, default=5000)
    g.add_argument("--resume", type=str, default=None)
    g.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--no-log", action="store_true")

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
    if no_log:
        logging.info("Neptune logging disabled via --no-log flag")
        return None
    try:
        if not os.path.exists(token_file):
            logging.warning(f"Neptune token file '{token_file}' not found. Skipping.")
            return None
        with open(token_file, 'r') as f:
            api_token = f.readline().strip()
        run = neptune.init_run(project=project, api_token=api_token)
        logging.info(f"Neptune run initialized: {run.get_url()}")
        return run
    except Exception as e:
        logging.warning(f"Failed to initialize Neptune: {e}. Continuing without.")
        return None


def log_parameters_to_neptune(run, args):
    if run is None:
        return
    try:
        run["parameters"] = vars(args)
    except Exception as e:
        logging.warning(f"Failed to log parameters to Neptune: {e}")


# =============================================================================
# DREAM Dataset Loading
# =============================================================================

def load_dream_datasets(args, tokenizer):
    """
    Load training and evaluation datasets for DREAM mode.

    Returns (train_ds, eval_ds) with the appropriate dataset type.

    Recommended datasets for evaluating continuous thought:
      - coin_flip:    Synthetic, tests state tracking (easy to diagnose)
      - last_letter:  Synthetic, tests extraction + concatenation
      - gsm8k:        Real math word problems (harder, more realistic)
      - wikitext:     Standard LM data (tests general benefit of thought)

    NOTE: For quick evaluation without heavy pretraining, coin_flip and
    last_letter are ideal because they have known difficulty curves (just
    increase max_flips or max_words). gsm8k is better for realistic eval
    but needs more training data to see benefits.
    """
    from transformers import AutoTokenizer

    dataset_name = args.dream_dataset or "wikitext"

    if dataset_name == "wikitext":
        # Use standard WikiText
        train_ds = WikiTextDataset("train", args.seq_len, args.dataset, args.tokenizer)
        eval_ds = WikiTextDataset("validation", args.seq_len, args.dataset, args.tokenizer)
        return train_ds, eval_ds

    elif dataset_name == "coin_flip":
        from dream_trainer import CoinFlipDataset
        train_ds = CoinFlipDataset(
            args.dream_num_samples, args.dream_max_flips, args.seq_len, tokenizer
        )
        eval_ds = CoinFlipDataset(
            args.dream_num_samples // 5, args.dream_max_flips, args.seq_len, tokenizer
        )
        return train_ds, eval_ds

    elif dataset_name == "last_letter":
        from dream_trainer import LastLetterDataset
        train_ds = LastLetterDataset(
            args.dream_num_samples, args.dream_max_words, args.seq_len, tokenizer
        )
        eval_ds = LastLetterDataset(
            args.dream_num_samples // 5, args.dream_max_words, args.seq_len, tokenizer
        )
        return train_ds, eval_ds

    elif dataset_name == "gsm8k":
        # Load GSM8K from HuggingFace
        from datasets import load_dataset as hf_load

        raw = hf_load("openai/gsm8k", "main")
        train_data, eval_data = [], []

        for split, storage in [("train", train_data), ("test", eval_data)]:
            for item in raw[split]:
                text = f"Question: {item['question']}\nAnswer: {item['answer']}"
                tokens = tokenizer.encode(text)
                if len(tokens) <= args.seq_len:
                    tokens = tokens + [tokenizer.pad_token_id] * (args.seq_len - len(tokens))
                else:
                    tokens = tokens[:args.seq_len]
                storage.append(torch.tensor(tokens, dtype=torch.long))

        class ListDataset(Dataset):
            def __init__(self, data): self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return self.data[idx]

        return ListDataset(train_data), ListDataset(eval_data)

    else:
        raise ValueError(f"Unknown DREAM dataset: {dataset_name}")


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

    # Initialize Neptune
    neptune_run = init_neptune_run(no_log=args.no_log)
    log_parameters_to_neptune(neptune_run, args)

    mode_label = {
        "card": "CARD",
        "ar": "AR",
        "dream": f"DREAM ({args.dream_base.upper()} base)",
    }[args.mode]

    logging.info("=" * 60)
    logging.info(f"{mode_label} Training")
    logging.info("=" * 60)
    for k, v in sorted(vars(args).items()):
        logging.info(f"  {k}: {v}")
    logging.info("=" * 60)

    # --- Tokenizer (needed early for DREAM synthetic datasets) ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    # --- Data ---
    if args.mode == "dream" and args.dream_dataset and args.dream_dataset != "wikitext":
        train_ds, eval_ds = load_dream_datasets(args, tokenizer)
        vocab_size = len(tokenizer)
    else:
        train_ds = WikiTextDataset("train", args.seq_len, args.dataset, args.tokenizer)
        eval_ds = WikiTextDataset("validation", args.seq_len, args.dataset, args.tokenizer)
        vocab_size = train_ds.vocab_size
        tokenizer = train_ds.tokenizer  # use the one with [MASK] added

    use_pin_memory = args.device.startswith("cuda")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin_memory, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin_memory,
    )

    # --- Model ---
    from model import CausalLM, ModelConfig

    dream_enabled = (args.mode == "dream")

    model_config = ModelConfig(
        vocab_size=vocab_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_len,
        dropout=args.dropout,
        dream_enabled=dream_enabled,
        dream_max_thoughts=args.dream_thought_steps if dream_enabled else 4,
        dream_projection=args.dream_projection if dream_enabled else "linear",
    )
    model = CausalLM(model_config)
    logging.info(f"Model: {model.count_parameters():,} parameters")
    if dream_enabled:
        logging.info(f"  DREAM projection: {args.dream_projection}")

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
            noise_schedule=args.noise_schedule,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            device=args.device,
            denoise_vis_samples=args.denoise_vis_samples,
            denoise_vis_steps=args.denoise_vis_steps,
            denoise_vis_block_size=args.denoise_vis_block_size,
        )
        trainer = CARDTrainer(
            model, train_loader, eval_loader, trainer_config,
            neptune_run=neptune_run, tokenizer=tokenizer,
        )

    elif args.mode == "ar":
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

    elif args.mode == "dream":
        from dream_trainer import DREAMTrainer, DREAMTrainerConfig
        trainer_config = DREAMTrainerConfig(
            base_mode=args.dream_base,
            thought_steps=args.dream_thought_steps,
            thought_loss_weight=args.dream_thought_loss_weight,
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
            noise_schedule=args.noise_schedule,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            device=args.device,
        )
        trainer = DREAMTrainer(model, train_loader, eval_loader, trainer_config, neptune_run=neptune_run)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    # Stop Neptune
    if neptune_run is not None:
        neptune_run.stop()
        logging.info("Neptune run stopped")

    logging.info("Done.")


if __name__ == "__main__":
    main()
