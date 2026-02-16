"""
dream_trainer.py — DREAM: Diffusion-Refined Embedding-space Autoregressive Model
==================================================================================

Implements CoCoNut-style continuous thought training where the model learns to
"think" in embedding space before committing to discrete tokens.

Core idea:
  Instead of:  input_ids → embed → transformer → logits → loss
  We do:       input_ids → embed → transformer → hidden → project_back_to_embed
               → transformer → ... (K thought steps) ... → logits → loss

The key insight: by not projecting to logits at intermediate steps, we keep the
representation continuous and differentiable. The model can refine its internal
representation over multiple "thought" steps before committing to a token.

DREAM interacts with CARD by treating thought embeddings as additional tokens
in the causal sequence. During CARD training, [MASK] tokens in the corrupted
input trigger the same masking/reweighting logic — thought steps happen on top
of the (possibly corrupted) input, so the model learns to "think through" noise.

Training:
  1. Standard forward pass on input tokens → hidden states
  2. For each thought step: project hidden → embedding space → forward pass
  3. Final logits from last thought step → cross-entropy loss
  4. Optionally: auxiliary loss from intermediate thought logits (curriculum)

Evaluation datasets for CoT-like reasoning without heavy pretraining:
  - gsm8k:     Grade school math (structured reasoning, 8.5K train)
  - coin_flip: Synthetic coin flip tracking (deterministic, easy to generate)
  - last_letter: Last letter concatenation (synthetic, scales arbitrarily)
  - reversal:  String reversal (synthetic, tests sequential reasoning)

We recommend gsm8k for realistic evaluation and coin_flip/reversal for
controlled ablations since they're synthetic and have known difficulty curves.
"""

import os
import time
import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import CausalLM, ModelConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DREAM Configuration
# =============================================================================

@dataclass
class DREAMTrainerConfig:
    """DREAM training hyperparameters."""

    # Base mode: train on top of AR or CARD
    base_mode: str = "ar"  # 'ar' or 'card'

    # Thought steps
    thought_steps: int = 3            # K: continuous thought iterations per position
    thought_loss_weight: float = 0.0  # weight for intermediate thought logit losses
    # NOTE: thought_loss_weight > 0 adds auxiliary supervision at each thought
    # step, which can help early training but may constrain the model's ability
    # to use non-interpretable intermediate representations. Set to 0.0 for
    # "pure" continuous thought, or 0.1-0.5 for curriculum-guided training.

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2500
    max_steps: int = 100_000

    # CARD-specific (only used when base_mode='card')
    tail_factor: float = 1.5
    reweight_beta: float = 1.0
    reweight_decay: float = 0.5
    noise_schedule: str = "linear"

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    use_amp: bool = True

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./checkpoints_dream"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Synthetic Reasoning Datasets
# =============================================================================

class CoinFlipDataset(Dataset):
    """
    Synthetic coin flip tracking dataset for evaluating reasoning.

    Format: "Person A flips. Person B doesn't flip. ... Is the coin heads up? Yes/No"

    This tests the model's ability to track state through a sequence of operations,
    which requires exactly the kind of sequential reasoning that continuous
    thought should help with.

    NOTE: Alternative synthetic tasks:
      - String reversal: "Reverse: hello → olleh" (pure sequential)
      - Parity: count 1s in binary string, output parity (accumulation)
      - Multi-step arithmetic: "2+3=5, 5*2=10, result?" (chained)
    We use coin_flip because it's the simplest task that requires multi-step
    state tracking while still being naturally expressible in text.
    """

    def __init__(self, num_samples: int, max_flips: int, seq_len: int, tokenizer):
        import random
        self.samples = []
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

        for _ in range(num_samples):
            n_flips = random.randint(2, max_flips)
            heads_up = True  # start heads up
            parts = ["A coin is heads up."]
            chosen = random.sample(names, min(n_flips, len(names)))
            if n_flips > len(names):
                chosen = chosen + random.choices(names, k=n_flips - len(names))

            for i in range(n_flips):
                if random.random() > 0.5:
                    parts.append(f" {chosen[i]} flips the coin.")
                    heads_up = not heads_up
                else:
                    parts.append(f" {chosen[i]} does not flip the coin.")

            answer = "Yes" if heads_up else "No"
            parts.append(f" Is the coin still heads up? {answer}")
            text = "".join(parts)

            tokens = tokenizer.encode(text)
            if len(tokens) <= seq_len:
                # Pad to seq_len
                tokens = tokens + [tokenizer.pad_token_id] * (seq_len - len(tokens))
            else:
                tokens = tokens[:seq_len]

            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class LastLetterDataset(Dataset):
    """
    Last letter concatenation: given a list of words, concatenate last letters.

    Format: "Take the last letter of each: cat dog fox → txg"

    This tests the model's ability to extract and concatenate information
    across multiple positions — a natural fit for continuous thought.
    """

    def __init__(self, num_samples: int, max_words: int, seq_len: int, tokenizer):
        import random
        self.samples = []
        self.tokenizer = tokenizer

        # Simple word list — common short words
        words = [
            "cat", "dog", "fox", "hat", "cup", "pen", "box", "red", "sun",
            "map", "run", "top", "big", "old", "new", "hot", "wet", "dry",
            "bus", "car", "bag", "bed", "fan", "key", "leg", "arm", "lip",
        ]

        for _ in range(num_samples):
            n_words = random.randint(2, max_words)
            chosen = random.choices(words, k=n_words)
            answer = "".join(w[-1] for w in chosen)
            text = f"Take the last letter of each word: {' '.join(chosen)} → {answer}"

            tokens = tokenizer.encode(text)
            if len(tokens) <= seq_len:
                tokens = tokens + [tokenizer.pad_token_id] * (seq_len - len(tokens))
            else:
                tokens = tokens[:seq_len]

            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# DREAM Trainer
# =============================================================================

class DREAMTrainer:
    """
    Trainer for DREAM (Diffusion-Refined Embedding-space Autoregressive Model).

    Supports two base modes:
      - 'ar':   standard AR + continuous thought
      - 'card': CARD diffusion + continuous thought

    The thought mechanism works as follows:
      1. Encode input tokens normally → hidden states h₀
      2. For k = 1..K thought steps:
         - Project h_{k-1} to embedding space via dream_proj
         - Run forward pass with these embeddings → h_k
      3. Compute logits from h_K → loss against targets

    With CARD mode, the input tokens may contain [MASK], and the context-aware
    reweighting still applies. The thought steps happen on top of the corrupted
    input, so the model learns to "think through" masked context.
    """

    def __init__(
        self,
        model: CausalLM,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: DREAMTrainerConfig,
        neptune_run=None,
    ):
        assert model.dream_proj is not None, (
            "Model must have dream_enabled=True in config"
        )
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.mask_token_id = model.config.mask_token_id
        self.neptune_run = neptune_run

        # Import CARD utilities if needed
        if config.base_mode == "card":
            from card_trainer import (
                soft_tail_masking, compute_context_aware_weights, NOISE_SCHEDULES
            )
            self.soft_tail_masking = soft_tail_masking
            self.compute_context_aware_weights = compute_context_aware_weights
            self.noise_schedule_fn = NOISE_SCHEDULES[config.noise_schedule]

        fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
            fused=fused and config.device.startswith("cuda"),
        )

        self.scheduler = self._build_cosine_schedule()
        self.device_type = "cuda" if config.device.startswith("cuda") else "cpu"
        self.use_amp = config.use_amp and self.device_type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        self.global_step = 0
        self.best_eval_loss = float("inf")
        os.makedirs(config.output_dir, exist_ok=True)

        logger.info(f"DREAM Trainer | {model.count_parameters():,} params")
        logger.info(f"  Base mode: {config.base_mode}")
        logger.info(f"  Thought steps: {config.thought_steps}")
        logger.info(f"  Thought loss weight: {config.thought_loss_weight}")

    def _build_cosine_schedule(self):
        cfg = self.config
        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def training_step(self, batch: torch.Tensor) -> float:
        """
        DREAM training step.

        The key difference from standard training: after the initial forward
        pass, we loop K thought steps where hidden states are projected back
        to embedding space and re-fed through the transformer.

        For CARD mode, the input is first corrupted with soft tail masking,
        and context-aware reweighting is applied to the loss.
        """
        cfg = self.config
        x0 = batch.to(cfg.device)
        B, L = x0.shape

        # Prepare input (CARD: corrupt, AR: clean)
        if cfg.base_mode == "card":
            t = torch.rand(B, device=cfg.device)
            x_t = self.soft_tail_masking(
                x0, t, self.mask_token_id, cfg.tail_factor,
                noise_schedule_fn=self.noise_schedule_fn,
            )
            input_ids = x_t[:, :-1]
            targets = x0[:, 1:]
        else:
            input_ids = x0[:, :-1]
            targets = x0[:, 1:]

        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            # Initial forward pass
            logits, _, hidden = self.model(input_ids, return_hidden=True)

            total_loss = torch.tensor(0.0, device=cfg.device)
            thought_losses = []

            # Auxiliary loss from initial logits (optional)
            if cfg.thought_loss_weight > 0:
                init_ce = F.cross_entropy(
                    logits.reshape(-1, self.model.config.vocab_size),
                    targets.reshape(-1),
                    reduction="mean",
                )
                thought_losses.append(init_ce)

            # Continuous thought steps
            for k in range(cfg.thought_steps):
                # Project hidden → embedding space
                thought_embed = self.model.dream_forward_thought(hidden)

                # Forward pass with continuous embeddings
                # NOTE: We don't use KV cache during training — full sequence
                # recomputation is needed because the embeddings change each step.
                # This means training cost scales as O(K * forward_pass_cost).
                # Alternative: cache the first pass and only re-run from the
                # thought injection point, but this requires careful bookkeeping.
                logits, _, hidden = self.model(
                    input_embeds=thought_embed, return_hidden=True
                )

                # Auxiliary loss at intermediate thoughts
                if cfg.thought_loss_weight > 0 and k < cfg.thought_steps - 1:
                    inter_ce = F.cross_entropy(
                        logits.reshape(-1, self.model.config.vocab_size),
                        targets.reshape(-1),
                        reduction="mean",
                    )
                    thought_losses.append(inter_ce)

            # Final loss (always computed)
            final_ce = F.cross_entropy(
                logits.reshape(-1, self.model.config.vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, -1)

            # Apply CARD reweighting if in card mode
            if cfg.base_mode == "card":
                weights = self.compute_context_aware_weights(
                    input_ids, self.mask_token_id,
                    cfg.reweight_beta, cfg.reweight_decay
                )
                loss = (weights * final_ce).sum() / weights.sum()
            else:
                loss = final_ce.mean()

            # Add auxiliary thought losses
            if thought_losses:
                aux_loss = cfg.thought_loss_weight * sum(thought_losses) / len(thought_losses)
                loss = loss + aux_loss

            loss = loss / cfg.gradient_accumulation_steps

        loss.backward()
        return loss.item() * cfg.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate with thought steps applied."""
        if self.eval_loader is None:
            return float("nan"), float("nan"), float("nan")

        cfg = self.config
        self.model.eval()
        total_loss, total_tokens, correct_tokens = 0.0, 0, 0
        start_time = time.time()

        for batch in self.eval_loader:
            x0 = batch.to(cfg.device)
            input_ids, targets = x0[:, :-1], x0[:, 1:]

            with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                # Forward with thought steps
                logits, _, hidden = self.model(input_ids, return_hidden=True)
                for _ in range(cfg.thought_steps):
                    thought_embed = self.model.dream_forward_thought(hidden)
                    logits, _, hidden = self.model(
                        input_embeds=thought_embed, return_hidden=True
                    )

                loss = F.cross_entropy(
                    logits.reshape(-1, self.model.config.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                )
                predictions = logits.argmax(dim=-1)
                correct_tokens += (predictions == targets).sum().item()

            total_loss += loss.item()
            total_tokens += targets.numel()

        elapsed = time.time() - start_time
        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        accuracy = correct_tokens / max(total_tokens, 1)
        tps = total_tokens / max(elapsed, 1e-6)
        return avg_loss, accuracy, tps

    def save_checkpoint(self, path: Optional[str] = None):
        path = path or os.path.join(self.config.output_dir, f"dream_step_{self.global_step}.pt")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "model_config": self.model.config,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["step"]
        self.best_eval_loss = ckpt.get("best_eval_loss", float("inf"))
        logger.info(f"Resumed from step {self.global_step}")

    def train(self):
        cfg = self.config
        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        t0 = time.time()
        tokens = 0

        logger.info(f"DREAM training ({cfg.base_mode}): steps {self.global_step} → {cfg.max_steps}")

        while self.global_step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                step_loss = self.training_step(batch)
                accum_loss += step_loss
                tokens += batch.numel()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            if self.global_step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                avg = accum_loss / cfg.log_interval
                lr = self.scheduler.get_last_lr()[0]
                tps = tokens / elapsed
                logger.info(
                    f"[DREAM-{cfg.base_mode.upper()}] step {self.global_step:>7d} | "
                    f"loss {avg:.4f} | lr {lr:.2e} | {tps:.0f} tok/s"
                )

                if self.neptune_run is not None:
                    try:
                        self.neptune_run["train/loss"].append(avg, step=self.global_step)
                        self.neptune_run["train/learning_rate"].append(lr, step=self.global_step)
                        self.neptune_run["train/tokens_per_sec"].append(tps, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log to Neptune: {e}")

                accum_loss = 0.0

            if self.global_step % cfg.eval_interval == 0:
                val_loss, val_acc, val_tps = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20))
                logger.info(f"  eval loss={val_loss:.4f} ppl={val_ppl:.2f} acc={val_acc:.4f} | {val_tps:.0f} tok/s")

                if self.neptune_run is not None:
                    try:
                        self.neptune_run["val/loss"].append(val_loss, step=self.global_step)
                        self.neptune_run["val/ppl"].append(val_ppl, step=self.global_step)
                        self.neptune_run["val/acc"].append(val_acc, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log eval metrics to Neptune: {e}")

                if val_loss < self.best_eval_loss:
                    self.best_eval_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(cfg.output_dir, "dream_best.pt")
                    )

            if self.global_step % cfg.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        logger.info("DREAM training complete.")
