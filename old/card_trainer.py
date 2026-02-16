"""
CARD Trainer: Training loop for Causal Autoregressive Diffusion LM
===================================================================
Implements the full Algorithm 1 from the paper:
  1. Noise scheduling:        t ~ U[0, 1]
  2. Soft tail masking:       concentrate corruption at sequence tail
  3. Context-aware reweighting: downweight predictions in ambiguous contexts
  4. Optimization:             weighted cross-entropy with cosine LR schedule

The key insight: this training loop has almost the same cost as standard
autoregressive LM training. The overhead is just the masking (O(L)) and
weight computation (O(L)), both negligible compared to the transformer forward.
"""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from card_model import CARDModel

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """
    Training hyperparameters. Defaults match Table 6 of the CARD paper.

    The paper trains a 1B-param model on 300B tokens from FineWeb.
    For WikiText training (much smaller), you'll want to reduce model size
    and training steps — see the CLI defaults in train.py.
    """
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2500
    max_steps: int = 1_000_000
    lr_scheduler: str = "cosine"  # Paper says "Cosine w/ Warmup"

    # CARD-specific diffusion hyperparameters
    tail_factor: float = 1.5     # λ: controls soft tail masking window width
    reweight_beta: float = 1.0   # β: smoothing constant for context-aware weights
    reweight_decay: float = 0.5  # p: exponential decay factor for noise distance

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    use_amp: bool = True         # BF16 mixed precision (paper uses BF16)

    # Logging & checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./checkpoints"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CARDTrainer:
    """
    Trainer for the CARD model.

    Implements the complete training procedure from Algorithm 1:
      - Soft Tail Masking (Section 3.2)
      - Context-Aware Reweighting (Section 3.3)
      - Cosine LR schedule with warmup (Table 6)
      - BF16 mixed-precision training

    Usage:
        model = CARDModel(...)
        trainer = CARDTrainer(model, train_loader, eval_loader, config)
        trainer.train()
    """

    def __init__(
        self,
        model: CARDModel,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: TrainerConfig,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config

        # Optimizer: AdamW as specified in Table 6
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
        )

        # LR scheduler: cosine annealing with linear warmup
        self.scheduler = self._build_scheduler()

        # Mixed-precision scaler (only used with float16, not bfloat16)
        # For bfloat16 we use autocast without a scaler since bf16 doesn't need
        # loss scaling (it has the same exponent range as float32)
        self.use_bf16 = config.use_amp and torch.cuda.is_bf16_supported()
        self.use_fp16 = config.use_amp and not self.use_bf16
        self.scaler = GradScaler(enabled=self.use_fp16)

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")

        os.makedirs(config.output_dir, exist_ok=True)

        # Log configuration
        n_params = model.count_parameters()
        logger.info(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        logger.info(f"Device: {config.device}")
        logger.info(f"Precision: {'bf16' if self.use_bf16 else 'fp16' if self.use_fp16 else 'fp32'}")
        logger.info(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Cosine LR schedule with linear warmup.

        - Linearly increase from 0 to peak_lr over warmup_steps
        - Cosine decay from peak_lr to 0 over remaining steps

        This matches the "Cosine w/ Warmup" specification in Table 6.
        """
        cfg = self.config

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                # Linear warmup: 0 → 1 over warmup_steps
                return step / max(1, cfg.warmup_steps)
            else:
                # Cosine decay: 1 → 0 over remaining steps
                progress = (step - cfg.warmup_steps) / max(
                    1, cfg.max_steps - cfg.warmup_steps
                )
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # =========================================================================
    # SOFT TAIL MASKING (Section 3.2, Algorithm 1 lines 6-11)
    # =========================================================================

    def soft_tail_masking(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Apply soft tail masking to a batch of clean sequences.

        Standard MDLM masks tokens UNIFORMLY across all positions. This is
        disastrous under causal attention because early positions have no
        future context to fall back on — masking position 2 forces the model
        to predict from essentially nothing.

        CARD's solution: concentrate masks at the TAIL of the sequence.
        The prefix stays clean, providing solid signal for the model.

        The "soft" part: the masking window is WIDER than the number of masks
        (controlled by tail_factor λ). This means within the tail region,
        you get a MIX of clean and masked tokens, preserving local context.

        Visual comparison for a 10-token sequence at t=0.3 (3 masks):
            Uniform:    [clean] [MASK] [clean] [clean] [MASK] [clean] [clean] [MASK] [clean] [clean]
            Strict tail: [clean] [clean] [clean] [clean] [clean] [clean] [clean] [MASK] [MASK] [MASK]
            Soft tail:   [clean] [clean] [clean] [clean] [clean] [clean] [MASK] [clean] [MASK] [MASK]
                                                                         ^--- window extends here

        The paper proves (Proposition 2, Appendix A) that soft tail masking
        preserves higher mutual information than uniform masking.

        Args:
            x0: (B, L) clean token ids
            t:  (B,)   noise levels in [0, 1]
        Returns:
            x_t: (B, L) corrupted sequences with tail-biased masking
        """
        B, L = x0.shape
        device = x0.device
        mask_id = self.model.mask_token_id
        lam = self.config.tail_factor

        x_t = x0.clone()

        # Vectorized implementation for the batch
        for b in range(B):
            t_val = t[b].item()

            # Number of tokens to mask: N = floor(L * t)
            N = max(1, int(L * t_val))

            # Window size: W = floor(N * λ), clamped to sequence length
            # λ > 1 means the window is wider than the mask count
            # λ = 1 would be strict tail masking (solid block of noise)
            W = min(L, int(N * lam))

            # The window covers the last W positions
            # We randomly select N positions WITHIN this window to mask
            window_start = L - W

            # Random permutation within the window → take first N indices
            perm = torch.randperm(W, device=device)[:N]
            mask_indices = perm + window_start

            x_t[b, mask_indices] = mask_id

        return x_t

    # =========================================================================
    # CONTEXT-AWARE REWEIGHTING (Section 3.3, Algorithm 1 lines 13-17)
    # =========================================================================

    def compute_context_aware_weights(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token loss weights based on context ambiguity.

        In standard ARM training, every position sees a clean prefix, so
        uniform weighting is fine. In CARD, each position sees a CORRUPTED
        prefix with varying amounts of noise. Predicting from heavy noise
        produces high-variance gradients that destabilize training.

        This function computes a "local ambiguity score" S^local_n for each
        position based on three factors (Eq. 6-7):

        1. QUANTITY: How many masks are in the context?
           → More masks = higher score = lower weight

        2. DISTANCE: How close are the masks to the current position?
           → Nearby masks hurt more (exponential decay with distance)
           → Follows the finding that LM context relevance decays
             exponentially (Khandelwal et al., 2018)

        3. DENSITY: Are masks consecutive?
           → Consecutive masks (spans) are worse than isolated masks
           → They completely sever local dependencies
           → Cost C_i = 1 for isolated mask, 2 for consecutive mask

        These combine into:
            C_i = 𝟙[x_i is MASK] · (1 + 𝟙[x_{i-1} is MASK])
            S^local_n = Σ_{i<n} C_i · (1-p)^{n-i}
            w_n = 1 / (β + S^local_n)

        The paper proves (Proposition 1) this is an inverse-variance
        weighting that minimizes gradient variance.

        Args:
            x_t: (B, L) corrupted token ids
        Returns:
            weights: (B, L) per-token loss weights, higher = more trustworthy
        """
        B, L = x_t.shape
        device = x_t.device
        beta = self.config.reweight_beta
        decay = self.config.reweight_decay

        is_mask = (x_t == self.model.mask_token_id).float()  # (B, L)

        # Density term: C_i = is_mask[i] * (1 + is_mask[i-1])
        # Consecutive masks → cost 2; isolated mask → cost 1; clean → cost 0
        prev_mask = F.pad(is_mask[:, :-1], (1, 0), value=0.0)
        C = is_mask * (1.0 + prev_mask)  # (B, L)

        # Compute S^local via the recurrence: S_n = C_n + (1-p) * S_{n-1}
        # This is an exponentially-decayed cumulative sum (causal EMA)
        retain = 1.0 - decay
        S_local = torch.zeros(B, L, device=device)
        running = torch.zeros(B, device=device)
        for n in range(L):
            running = C[:, n] + retain * running
            S_local[:, n] = running

        # Inverse weighting: ambiguous contexts get low weight
        weights = 1.0 / (beta + S_local)

        return weights

    # =========================================================================
    # SINGLE TRAINING STEP
    # =========================================================================

    def training_step(self, batch: torch.Tensor) -> float:
        """
        Execute one CARD training step (Algorithm 1).

        The complete procedure:
          1. Sample t ~ U[0,1] per sequence in the batch
          2. Corrupt via soft tail masking: x0 → x_t
          3. Shift for next-token prediction: input = x_t[:-1], target = x0[1:]
          4. Forward pass through causal transformer (same cost as ARM)
          5. Compute context-aware weights for the input
          6. Weighted cross-entropy loss
          7. Backprop

        At t=0 (no noise): this is EXACTLY autoregressive training.
        At t=1 (all noise): the model must generate from pure masks.
        By sampling t uniformly, the model learns the full spectrum.

        Args:
            batch: (B, L) clean token sequences
        Returns:
            loss value (float, detached)
        """
        cfg = self.config
        x0 = batch.to(cfg.device)
        B, L = x0.shape

        # Step 1: Sample noise level for each sequence
        # Linear schedule σ(t) = t as stated in Section 2.3
        t = torch.rand(B, device=cfg.device)

        # Step 2: Corrupt input via soft tail masking
        x_t = self.soft_tail_masking(x0, t)

        # Step 3: Shifted causal setup (standard GPT-style)
        # The model at position n sees x^t_{≤n} and predicts x_{n+1}
        input_ids = x_t[:, :-1]   # (B, L-1) — corrupted context
        targets = x0[:, 1:]       # (B, L-1) — clean next tokens

        # Step 4: Forward pass with mixed precision
        amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.use_amp):
            logits = self.model(input_ids, t)  # (B, L-1, V)

            # Step 5: Context-aware weights
            weights = self.compute_context_aware_weights(input_ids)  # (B, L-1)

            # Step 6: Weighted cross-entropy
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.model.vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, -1)  # (B, L-1)

            loss = (weights * ce_loss).sum() / weights.sum()

            # Scale for gradient accumulation
            loss = loss / cfg.gradient_accumulation_steps

        # Step 7: Backward pass
        if self.use_fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * cfg.gradient_accumulation_steps

    # =========================================================================
    # EVALUATION
    # =========================================================================

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Compute average loss on the eval set.

        Uses t=0 (no noise) for evaluation, which makes this equivalent
        to standard autoregressive perplexity. This gives a clean comparison
        with ARM baselines.
        """
        if self.eval_loader is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.eval_loader:
            x0 = batch.to(self.config.device)
            B, L = x0.shape

            input_ids = x0[:, :-1]
            targets = x0[:, 1:]

            # Evaluate with t=0 (clean input, standard AR evaluation)
            t = torch.zeros(B, device=self.config.device)
            logits = self.model(input_ids, t)

            loss = F.cross_entropy(
                logits.reshape(-1, self.model.vocab_size),
                targets.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += targets.numel()

        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss

    # =========================================================================
    # CHECKPOINTING
    # =========================================================================

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model, optimizer, scheduler state and training progress."""
        if path is None:
            path = os.path.join(
                self.config.output_dir, f"checkpoint_step_{self.global_step}.pt"
            )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))

        logger.info(f"Resumed from {path} at step {self.global_step}")

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================

    def train(self):
        """
        Main training loop.

        Iterates over the training data, calling training_step() for each batch,
        with gradient accumulation, LR scheduling, periodic evaluation, and
        checkpointing.

        The loop structure is standard PyTorch training — the CARD-specific
        logic (masking, reweighting) is all inside training_step().
        """
        cfg = self.config
        self.model.train()

        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        start_time = time.time()
        tokens_processed = 0

        logger.info(f"Starting training from step {self.global_step}")
        logger.info(f"Total steps: {cfg.max_steps}")

        while self.global_step < cfg.max_steps:
            # --- Gradient accumulation loop ---
            self.optimizer.zero_grad()

            for micro_step in range(cfg.gradient_accumulation_steps):
                # Get next batch, cycling through the dataset
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                loss = self.training_step(batch)
                accum_loss += loss

                tokens_processed += batch.numel()

            # --- Gradient clipping + optimizer step ---
            if self.use_fp16:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.max_grad_norm
            )

            if self.use_fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            # --- Logging ---
            if self.global_step % cfg.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed
                current_lr = self.scheduler.get_last_lr()[0]
                avg_loss = accum_loss / cfg.log_interval

                logger.info(
                    f"Step {self.global_step:>7d} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tok/s: {tokens_per_sec:.0f} | "
                    f"Elapsed: {elapsed:.0f}s"
                )

                accum_loss = 0.0

            # --- Evaluation ---
            if self.global_step % cfg.eval_interval == 0:
                eval_loss = self.evaluate()
                eval_ppl = math.exp(min(eval_loss, 20))  # Clamp to avoid overflow

                logger.info(
                    f"  Eval @ step {self.global_step}: "
                    f"Loss = {eval_loss:.4f}, PPL = {eval_ppl:.2f}"
                )

                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    best_path = os.path.join(cfg.output_dir, "best_model.pt")
                    self.save_checkpoint(best_path)
                    logger.info(f"  New best model (loss={eval_loss:.4f})")

            # --- Periodic checkpointing ---
            if self.global_step % cfg.save_interval == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()
        logger.info(f"Training complete after {self.global_step} steps")
