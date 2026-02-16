"""
card_trainer.py — CARD Training Loop (Algorithm 1)
====================================================

Implements the full training procedure from the paper:
  1. Noise scheduling:             t ~ U[0, 1], σ(t) configurable
  2. Soft tail masking:            Section 3.2, Algorithm 1 lines 6-11
  3. Context-aware reweighting:    Section 3.3, Algorithm 1 lines 13-17
  4. Dense supervision:            Loss on ALL positions (100% token utilization)
  5. Cosine LR with warmup:        Table 6

Noise schedule options:
  - 'linear': σ(t) = t                          (paper default)
  - 'cosine': σ(t) = (1 - cos(πt)) / 2         (DLM literature standard)

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


# =============================================================================
# NOISE SCHEDULES
# =============================================================================

def linear_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """Linear schedule: σ(t) = t. Paper default."""
    return t


def cosine_noise_schedule(t: torch.Tensor) -> torch.Tensor:
    """
    Cosine schedule: σ(t) = (1 - cos(πt)) / 2.

    Commonly used in the DLM literature (Nichol & Dhariwal, 2021; Sahoo et al., 2024).
    Provides a gentler ramp-up near t=0 and t=1 compared to linear, which can
    improve training stability by reducing the frequency of extreme noise levels.

    NOTE: Alternative cosine variants exist:
      - Shifted cosine: σ(t) = (1 - cos(π(t+s)/(1+s))) / 2 with offset s
      - Squared cosine: σ(t) = sin²(πt/2) (equivalent to this formulation)
      - Log-linear:     σ(t) = 1 - (1-t)^α for adjustable curvature
    We use the standard cosine as it's the most common in MDLM/LLaDA.
    """
    return (1.0 - torch.cos(math.pi * t)) / 2.0


NOISE_SCHEDULES = {
    "linear": linear_noise_schedule,
    "cosine": cosine_noise_schedule,
}


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

    # Noise schedule
    noise_schedule: str = "linear"  # 'linear' or 'cosine'

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    use_amp: bool = True

    # Logging & saving
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./checkpoints_card"

    # Denoising visualization during eval
    denoise_vis_samples: int = 3      # number of examples to visualize
    denoise_vis_steps: int = 8        # denoising steps for visualization
    denoise_vis_block_size: int = 16  # block size for visualization

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# SOFT TAIL MASKING — Section 3.2, Algorithm 1 lines 6-11
# =============================================================================

def soft_tail_masking(
    x0: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    tail_factor: float = 1.5,
    noise_schedule_fn=None,
) -> torch.Tensor:
    """
    Corrupt clean sequences by concentrating [MASK] tokens at the tail.

    The noise_schedule_fn maps t ∈ [0,1] to σ(t) ∈ [0,1], which controls
    the fraction of tokens to mask. With a linear schedule, σ(t)=t and we
    mask L·t tokens. With a cosine schedule, the mapping is nonlinear and
    provides gentler ramp-up at the extremes.

    Args:
        x0:            (B, L) clean token ids
        t:             (B,)   raw noise levels in [0, 1]
        mask_token_id: id for [MASK]
        tail_factor:   λ from Algorithm 1
        noise_schedule_fn: maps t → σ(t), the effective masking rate
    Returns:
        x_t: (B, L) corrupted sequences
    """
    B, L = x0.shape
    device = x0.device
    x_t = x0.clone()

    # Apply noise schedule to get effective masking rate
    if noise_schedule_fn is not None:
        sigma = noise_schedule_fn(t)  # (B,)
    else:
        sigma = t  # linear default

    # Algorithm 1, line 6
    N = torch.clamp((L * sigma).floor().long(), min=1)   # (B,) number to mask
    W = torch.clamp((N.float() * tail_factor).floor().long(), max=L)  # (B,) window size

    for b in range(B):
        n, w = N[b].item(), W[b].item()
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

    See Section 3.3 and Algorithm 1 lines 13-17 for the full derivation.
    S^local_n = Σ_{i=1}^{n} C_i · (1-p)^{n-1-i}
    w_n = (β + S^local_n)^{-1}

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
    retain = 1.0 - decay

    is_mask = (x_t == mask_token_id).float()
    prev_mask = F.pad(is_mask[:, :-1], (1, 0), value=0.0)
    C = is_mask * (1.0 + prev_mask)

    # IIR filter: S_n = C_n + retain · S_{n-1}
    S_local = torch.zeros(B, L, device=device)
    s = torch.zeros(B, device=device)
    for n in range(L):
        s = C[:, n] + retain * s
        S_local[:, n] = s

    weights = 1.0 / (beta + S_local)
    return weights


# =============================================================================
# Trainer Class
# =============================================================================

class CARDTrainer:
    """
    Trainer for CARD (Causal Autoregressive Diffusion).

    The model is a standard CausalLM (GPT). All CARD-specific logic —
    masking, reweighting, noise scheduling — lives here in the trainer.
    """

    def __init__(
        self,
        model: CausalLM,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: CARDTrainerConfig,
        neptune_run=None,
        tokenizer=None,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.mask_token_id = model.config.mask_token_id
        self.neptune_run = neptune_run
        self.tokenizer = tokenizer  # needed for denoising visualization

        # Noise schedule function
        self.noise_schedule_fn = NOISE_SCHEDULES[config.noise_schedule]

        # AdamW with fused kernel if available
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

        logger.info(f"CARD Trainer | {model.count_parameters():,} params")
        logger.info(f"  Device: {config.device} | AMP: {self.use_amp} ({self.amp_dtype})")
        logger.info(f"  Tail factor λ={config.tail_factor}, β={config.reweight_beta}, "
                     f"decay p={config.reweight_decay}")
        logger.info(f"  Noise schedule: {config.noise_schedule}")

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
        """
        cfg = self.config
        x0 = batch.to(cfg.device)
        B, L = x0.shape

        # Step 1: sample noise level per sequence
        t = torch.rand(B, device=cfg.device)

        # Step 2: corrupt via soft tail masking (with noise schedule)
        x_t = soft_tail_masking(
            x0, t, self.mask_token_id, cfg.tail_factor,
            noise_schedule_fn=self.noise_schedule_fn,
        )

        # Step 3: shifted causal setup
        input_ids = x_t[:, :-1]
        targets   = x0[:, 1:]

        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            logits, _ = self.model(input_ids)

            weights = compute_context_aware_weights(
                input_ids, self.mask_token_id, cfg.reweight_beta, cfg.reweight_decay
            )

            ce = F.cross_entropy(
                logits.reshape(-1, self.model.config.vocab_size),
                targets.reshape(-1),
                reduction="none",
            ).reshape(B, -1)

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

        Returns:
            (loss, accuracy, tokens_per_sec)
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
    # Denoising Visualization
    # =========================================================================

    @torch.no_grad()
    def visualize_denoising(self):
        """
        Log the progressive denoising process for a few validation examples.

        For each sample, we:
          1. Take a clean prefix
          2. Append block_size [MASK] tokens
          3. Iteratively denoise, logging the state at each step
          4. Show the clean target for comparison

        Output is logged both to console and Neptune (if available).
        This provides interpretable insight into how CARD decodes.
        """
        if self.eval_loader is None:
            return

        cfg = self.config
        self.model.eval()
        mask_id = self.mask_token_id

        # Grab a batch from eval
        batch = next(iter(self.eval_loader))
        x0 = batch[:cfg.denoise_vis_samples].to(cfg.device)  # (N, L)
        B, L = x0.shape

        # Split: use first half as prefix, second half as target
        prefix_len = L // 2
        block_size = min(cfg.denoise_vis_block_size, L - prefix_len)
        prefix = x0[:, :prefix_len]
        target = x0[:, prefix_len:prefix_len + block_size]

        # Encode prefix
        kv_caches = self.model.init_kv_caches(B, cfg.device, self.amp_dtype)
        _, kv_caches = self.model(prefix, pos_offset=0, kv_caches=kv_caches)

        # Initialize block with all [MASK]
        block = torch.full((B, block_size), mask_id, device=cfg.device, dtype=torch.long)

        all_steps = []
        all_steps.append(("init", block.clone()))

        # Iterative denoising
        for step in range(cfg.denoise_vis_steps):
            is_masked = (block == mask_id)
            if not is_masked.any():
                break

            temp_caches = [c.clone() for c in kv_caches]
            logits, _ = self.model(block, pos_offset=prefix_len, kv_caches=temp_caches)
            probs = logits.softmax(dim=-1)
            max_probs, predicted = probs.max(dim=-1)

            # Use adaptive threshold for visualization
            progress = step / max(cfg.denoise_vis_steps - 1, 1)
            tau = 0.9 * (1 - progress) + 0.1 * progress
            confident = is_masked & (max_probs > tau)
            block[confident] = predicted[confident]

            # Force-decode on last step
            if step == cfg.denoise_vis_steps - 1:
                still_masked = (block == mask_id)
                if still_masked.any():
                    for b in range(B):
                        mp = still_masked[b].nonzero(as_tuple=True)[0]
                        if mp.numel() > 0:
                            block[b, mp] = predicted[b, mp]

            all_steps.append((f"step_{step+1}", block.clone()))

        # Format and log
        vis_text = self._format_denoising_vis(prefix, target, all_steps)
        logger.info(f"\n{'='*70}\nDENOISING VISUALIZATION (step {self.global_step})\n{'='*70}\n{vis_text}")

        # Log to Neptune
        if self.neptune_run is not None:
            try:
                self.neptune_run[f"denoising_vis/step_{self.global_step}"].append(vis_text)
            except Exception as e:
                logger.warning(f"Failed to log denoising vis to Neptune: {e}")

        self.model.train()

    def _format_denoising_vis(self, prefix, target, steps):
        """Format denoising steps for human-readable logging."""
        lines = []
        B = prefix.shape[0]

        for b in range(B):
            lines.append(f"\n--- Sample {b+1} ---")

            # Decode tokens to text if tokenizer available
            if self.tokenizer is not None:
                prefix_text = self.tokenizer.decode(prefix[b], skip_special_tokens=False)
                target_text = self.tokenizer.decode(target[b], skip_special_tokens=False)
                lines.append(f"  PREFIX: ...{prefix_text[-80:]}")
                lines.append(f"  TARGET: {target_text[:80]}")
                lines.append(f"  DENOISING:")
                for step_name, block in steps:
                    tokens = []
                    for tok_id in block[b]:
                        if tok_id.item() == self.mask_token_id:
                            tokens.append("[M]")
                        else:
                            tokens.append(self.tokenizer.decode([tok_id.item()]))
                    line = "".join(tokens)
                    lines.append(f"    {step_name:>8}: {line[:100]}")
            else:
                # Fallback: show token ids with M for masks
                lines.append(f"  PREFIX (last 10): {prefix[b, -10:].tolist()}")
                lines.append(f"  TARGET:           {target[b].tolist()}")
                lines.append(f"  DENOISING:")
                for step_name, block in steps:
                    tok_strs = []
                    for tok_id in block[b]:
                        if tok_id.item() == self.mask_token_id:
                            tok_strs.append(" [M]")
                        else:
                            tok_strs.append(f" {tok_id.item()}")
                    lines.append(f"    {step_name:>8}:{' '.join(tok_strs[:20])}")

        return "\n".join(lines)

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

                # Denoising visualization (every eval)
                if cfg.denoise_vis_samples > 0:
                    self.visualize_denoising()

            # Periodic save
            if self.global_step % cfg.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        logger.info("CARD training complete.")
