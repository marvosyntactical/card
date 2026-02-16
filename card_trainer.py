"""
card_trainer.py — CARD Training Loop (Algorithm 1)
====================================================

Implements the full training procedure from the paper:
  1. Noise scheduling:             t ~ U[0, 1], σ(t) = t (linear)
  2. Soft tail masking:            Section 3.2, Algorithm 1 lines 6-11
  3. Context-aware reweighting:    Section 3.3, Algorithm 1 lines 13-17
  4. Dense supervision:            Loss on ALL positions (100% token utilization)
  5. Cosine LR with warmup:        Table 6

Key corrections from naive implementation:
  - NO timestep conditioning in the model (Eq. 5 has no explicit t)
  - Reweighting exponent is (n-1-i) not (n-i) (Algorithm 1, line 15)
  - Loss is on ALL positions, not just masked ones (Section 3.1)
  - The model is a standard GPT — all CARD logic is in this trainer
"""

import os
import time
import math
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CausalLM, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class CARDTrainerConfig:
    """Training hyperparameters. Defaults from Table 6 + Algorithm 1."""

    # Optimizer (Table 6)
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # Schedule (Table 6: "Cosine w/ Warmup")
    warmup_steps: int = 2500
    max_steps: int = 100_000

    # CARD diffusion hyperparameters (Algorithm 1, line 2)
    tail_factor: float = 1.5     # λ: window width multiplier (λ=1 → strict tail)
    reweight_beta: float = 1.0   # β: smoothing constant (Algorithm 1, line 16)
    reweight_decay: float = 0.5  # p: distance decay factor (Algorithm 1, line 15)

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    use_amp: bool = True

    # Logging & saving
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./checkpoints_card"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SOFT TAIL MASKING — Section 3.2, Algorithm 1 lines 6-11
# =============================================================================

def soft_tail_masking(
    x0: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    tail_factor: float = 1.5,
) -> torch.Tensor:
    """
    Corrupt clean sequences by concentrating [MASK] tokens at the tail.

    WHY NOT UNIFORM MASKING?
    Under causal attention, position n can only see positions < n.
    If early positions are masked, the model has no signal to predict from.
    Example: if positions 0-3 are all [MASK], predicting position 4 is
    pure guessing because the entire causal context is noise.

    SOFT TAIL MASKING SOLUTION:
    1. Compute N = ⌊L·t⌋ tokens to mask
    2. Define a window of size W = ⌊N·λ⌋ at the sequence tail
    3. Randomly place N masks within this window of W positions

    The window is WIDER than N (since λ > 1), creating a mix of clean
    and masked tokens in the tail. This preserves local context even
    in the corrupted region (Proposition 2: higher MI than uniform).

    Example (L=10, t=0.3, λ=1.5):
        N = 3 masks, W = 4 positions
        Clean prefix:  [tok] [tok] [tok] [tok] [tok] [tok]
        Tail window:                                       [tok] [MASK] [MASK] [MASK]
                                                           ^--- window of 4, 3 masked

    Args:
        x0:            (B, L) clean token ids
        t:             (B,)   noise levels in [0, 1]
        mask_token_id: id for [MASK]
        tail_factor:   λ from Algorithm 1
    Returns:
        x_t: (B, L) corrupted sequences
    """
    B, L = x0.shape
    device = x0.device
    x_t = x0.clone()

    # Algorithm 1, line 6
    N = torch.clamp((L * t).floor().long(), min=1)   # (B,) number to mask
    W = torch.clamp((N.float() * tail_factor).floor().long(), max=L)  # (B,) window size

    for b in range(B):
        n, w = N[b].item(), W[b].item()
        # Algorithm 1, line 7: sample n indices from the last w positions
        window_start = L - w
        perm = torch.randperm(w, device=device)[:n]
        x_t[b, window_start + perm] = mask_token_id

    return x_t


# =============================================================================
# CONTEXT-AWARE REWEIGHTING — Section 3.3, Algorithm 1 lines 13-17
# =============================================================================

def compute_context_aware_weights(
    x_t: torch.Tensor,
    mask_token_id: int,
    beta: float = 1.0,
    decay: float = 0.5,
) -> torch.Tensor:
    """
    Per-token loss weights that downweight predictions from ambiguous contexts.

    The weight for position n depends on three properties of its causal
    context x^t_{<n} (Section 3.3):

    QUANTITY — total masked tokens in context:
        Accumulated via the summation Σ_{i=1}^{n}.

    DISTANCE — proximity of masks to position n:
        Nearby masks matter more. Decay factor (1-p)^{n-1-i} gives
        exponential decay. With p=0.5: a mask 1 position away has 2×
        the impact of one 2 positions away. This follows the finding
        that LM context relevance decays exponentially (Khandelwal 2018).

    DENSITY — consecutive mask spans:
        C_i = 𝟙[x_i = MASK] · (1 + 𝟙[x_{i-1} = MASK])
        Isolated mask → cost 1. Consecutive masks → cost 2.
        Spans sever ALL local dependencies, making prediction harder.

    Combined into (Algorithm 1, lines 14-16):
        S^local_n = Σ_{i=1}^{n} C_i · (1-p)^{n-1-i}
        w_n = (β + S^local_n)^{-1}

    Proposition 1 proves this is inverse-variance weighting that minimizes
    the variance of stochastic gradients, stabilizing optimization without
    requiring aggressive EMA.

    Args:
        x_t:           (B, L) corrupted tokens
        mask_token_id: [MASK] id
        beta:          β smoothing (prevents div by zero, default 1.0)
        decay:         p decay factor (default 0.5 as in paper)
    Returns:
        weights: (B, L) per-position loss weights
    """
    B, L = x_t.shape
    device = x_t.device
    retain = 1.0 - decay  # (1-p), the retention factor

    is_mask = (x_t == mask_token_id).float()  # (B, L)

    # Algorithm 1, line 14: C_n = 𝟙[x_n is MASK] · (1 + 𝟙[x_{n-1} is MASK])
    prev_mask = F.pad(is_mask[:, :-1], (1, 0), value=0.0)  # shifted right
    C = is_mask * (1.0 + prev_mask)  # (B, L)

    # Algorithm 1, line 15: S^local_n = Σ_{i=1}^{n} C_i · (1-p)^{n-1-i}
    # This is a first-order IIR filter: S_n = C_n + retain · S_{n-1}
    # Note the exponent is (n-1-i), which means:
    #   S_1 = C_1 · (1-p)^0 = C_1
    #   S_2 = C_2 · (1-p)^0 + C_1 · (1-p)^1 = C_2 + retain·C_1 = C_2 + retain·S_1
    # So the recurrence is: S_n = C_n + retain · S_{n-1}  ✓
    S_local = torch.zeros(B, L, device=device)
    s = torch.zeros(B, device=device)
    for n in range(L):
        s = C[:, n] + retain * s
        S_local[:, n] = s

    # Algorithm 1, line 16: w_n = (β + S^local_n)^{-1}
    weights = 1.0 / (beta + S_local)

    return weights


# =============================================================================
# Trainer Class
# =============================================================================

class CARDTrainer:
    """
    Trainer for CARD (Causal Autoregressive Diffusion).

    The model is a standard CausalLM (GPT). All CARD-specific logic —
    masking, reweighting, noise scheduling — lives here in the trainer,
    not in the model architecture.

    This separation means:
      - Same model class for CARD and AR (fair comparison)
      - CARD training cost ≈ AR training cost (same forward pass)
      - The model can be used for AR generation after CARD training
    """

    def __init__(
        self,
        model: CausalLM,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: CARDTrainerConfig,
        neptune_run=None,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.mask_token_id = model.config.mask_token_id
        self.neptune_run = neptune_run

        # AdamW with fused kernel if available (faster on CUDA)
        fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
            fused=fused and config.device.startswith("cuda"),
        )

        self.scheduler = self._build_cosine_schedule()
        # Determine device type for autocast (cuda or cpu)
        self.device_type = "cuda" if config.device.startswith("cuda") else "cpu"
        self.use_amp = config.use_amp and self.device_type == "cuda"
        self.amp_dtype = torch.bfloat16 if (self.device_type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        self.global_step = 0
        self.best_eval_loss = float("inf")
        os.makedirs(config.output_dir, exist_ok=True)

        logger.info(f"CARD Trainer | {model.count_parameters():,} params")
        logger.info(f"  Device: {config.device} | AMP: {self.use_amp} ({self.amp_dtype})")
        logger.info(f"  Tail factor λ={config.tail_factor}, β={config.reweight_beta}, "
                     f"decay p={config.reweight_decay}")

    def _build_cosine_schedule(self):
        """Cosine annealing with linear warmup (Table 6)."""
        cfg = self.config
        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # =========================================================================
    # Training Step — Algorithm 1
    # =========================================================================

    def training_step(self, batch: torch.Tensor) -> float:
        """
        One CARD training step implementing Algorithm 1.

        Steps:
          1. Sample t ~ U[0,1] per sequence               (line 4)
          2. Apply soft tail masking: x0 → x_t             (lines 6-11)
          3. Shift for next-token prediction               (standard GPT)
          4. Forward pass (same cost as AR)                (line 19)
          5. Compute context-aware weights                 (lines 13-17)
          6. Weighted cross-entropy, all positions         (line 19)
          7. Backprop                                      (line 20)

        CRITICAL: the loss is computed at ALL positions (100% token utilization,
        Section 3.1), not just masked ones. This is what gives CARD the same
        data efficiency as ARM. Clean-context positions get high weight
        (strong signal), noisy-context positions get low weight (weak signal).
        """
        cfg = self.config
        x0 = batch.to(cfg.device)  # (B, L) clean tokens
        B, L = x0.shape

        # Step 1: sample noise level per sequence
        t = torch.rand(B, device=cfg.device)

        # Step 2: corrupt via soft tail masking
        x_t = soft_tail_masking(x0, t, self.mask_token_id, cfg.tail_factor)

        # Step 3: shifted causal setup (standard GPT next-token prediction)
        # Position n in input → predict position n+1 in targets
        input_ids = x_t[:, :-1]   # (B, L-1) corrupted prefix
        targets   = x0[:, 1:]     # (B, L-1) CLEAN next tokens

        # Steps 4-6 under mixed precision
        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            logits, _ = self.model(input_ids)  # (B, L-1, V)

            # Step 5: context-aware weights on the input (the corrupted context)
            weights = compute_context_aware_weights(
                input_ids, self.mask_token_id, cfg.reweight_beta, cfg.reweight_decay
            )  # (B, L-1)

            # Step 6: weighted cross-entropy over ALL positions
            ce = F.cross_entropy(
                logits.reshape(-1, self.model.config.vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, -1)  # (B, L-1)

            loss = (weights * ce).sum() / weights.sum()
            loss = loss / cfg.gradient_accumulation_steps

        loss.backward()
        return loss.item() * cfg.gradient_accumulation_steps

    # =========================================================================
    # Evaluation
    # =========================================================================

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """
        Evaluate at t=0 (clean input) → standard AR perplexity.
        This gives a clean comparison with the AR baseline.

        Returns:
            (loss, accuracy, tokens_per_sec): average loss per token, accuracy, and throughput
        """
        if self.eval_loader is None:
            return float("nan"), float("nan"), float("nan")

        self.model.eval()
        total_loss, total_tokens, correct_tokens = 0.0, 0, 0
        start_time = time.time()

        for batch in self.eval_loader:
            x0 = batch.to(self.config.device)
            input_ids, targets = x0[:, :-1], x0[:, 1:]

            with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
                logits, _ = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.model.config.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                )

                # Compute accuracy
                predictions = logits.argmax(dim=-1)
                correct_tokens += (predictions == targets).sum().item()

            total_loss += loss.item()
            total_tokens += targets.numel()

        elapsed_time = time.time() - start_time
        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        accuracy = correct_tokens / max(total_tokens, 1)
        tokens_per_sec = total_tokens / max(elapsed_time, 1e-6)
        return avg_loss, accuracy, tokens_per_sec

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def save_checkpoint(self, path: Optional[str] = None):
        path = path or os.path.join(self.config.output_dir, f"card_step_{self.global_step}.pt")
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

    # =========================================================================
    # Main loop
    # =========================================================================

    def train(self):
        cfg = self.config
        self.model.train()
        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        t0 = time.time()
        tokens = 0

        logger.info(f"CARD training: steps {self.global_step} → {cfg.max_steps}")

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

            # Gradient clipping + optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Logging
            if self.global_step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                avg = accum_loss / cfg.log_interval
                lr = self.scheduler.get_last_lr()[0]
                tps = tokens / elapsed
                logger.info(
                    f"[CARD] step {self.global_step:>7d} | loss {avg:.4f} | "
                    f"lr {lr:.2e} | {tps:.0f} tok/s"
                )

                # Log to Neptune
                if self.neptune_run is not None:
                    try:
                        self.neptune_run["train/loss"].append(avg, step=self.global_step)
                        self.neptune_run["train/learning_rate"].append(lr, step=self.global_step)
                        self.neptune_run["train/tokens_per_sec"].append(tps, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log to Neptune: {e}")

                accum_loss = 0.0

            # Eval
            if self.global_step % cfg.eval_interval == 0:
                val_loss, val_acc, val_tps = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20))
                logger.info(f"  eval loss={val_loss:.4f} ppl={val_ppl:.2f} acc={val_acc:.4f} | {val_tps:.0f} tok/s")

                # Log to Neptune
                if self.neptune_run is not None:
                    try:
                        self.neptune_run["val/loss"].append(val_loss, step=self.global_step)
                        self.neptune_run["val/ppl"].append(val_ppl, step=self.global_step)
                        self.neptune_run["val/acc"].append(val_acc, step=self.global_step)
                        self.neptune_run["val/tokens_per_sec"].append(val_tps, step=self.global_step)
                    except Exception as e:
                        logger.warning(f"Failed to log eval metrics to Neptune: {e}")

                if val_loss < self.best_eval_loss:
                    self.best_eval_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(cfg.output_dir, "card_best.pt")
                    )

            # Periodic save
            if self.global_step % cfg.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        logger.info("CARD training complete.")
