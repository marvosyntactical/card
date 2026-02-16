"""
ar_trainer.py — Autoregressive Baseline Trainer
=================================================

Standard next-token prediction training using the SAME CausalLM architecture
as CARD. This provides a fair comparison: identical model, identical data,
only the training objective differs.

AR training:  L = Σ_n log p(x_n | x_{<n})         — clean prefix, no noise
CARD training: L = Σ_n w_n · log p(x_n | x^t_{<n}) — noisy prefix, weighted

The paper reports that on 300B tokens (Table 1):
  ARM:  56.39% average accuracy
  CARD: 53.23% average accuracy (but lower PPL on 6/8 domains, Table 2)
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

from model import CausalLM

logger = logging.getLogger(__name__)


@dataclass
class ARTrainerConfig:
    """AR training hyperparameters — same optimizer settings as CARD for fairness."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_steps: int = 2500
    max_steps: int = 100_000
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    output_dir: str = "./checkpoints_ar"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ARTrainer:
    """
    Standard autoregressive LM trainer.

    Nothing fancy — this is vanilla GPT training:
      input:  x_0, x_1, ..., x_{L-2}
      target: x_1, x_2, ..., x_{L-1}
      loss:   cross-entropy, uniform weight
    """

    def __init__(
        self,
        model: CausalLM,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: ARTrainerConfig,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config

        fused = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay,
            fused=fused and config.device == "cuda",
        )

        self.scheduler = self._build_cosine_schedule()
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.global_step = 0
        self.best_eval_loss = float("inf")
        os.makedirs(config.output_dir, exist_ok=True)

        logger.info(f"AR Trainer | {model.count_parameters():,} params")

    def _build_cosine_schedule(self):
        cfg = self.config
        def lr_lambda(step):
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def training_step(self, batch: torch.Tensor) -> float:
        """Standard next-token prediction: CE(logits, shifted targets)."""
        cfg = self.config
        x = batch.to(cfg.device)
        input_ids, targets = x[:, :-1], x[:, 1:]

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            logits, _ = self.model(input_ids)  # (B, L-1, V)
            loss = F.cross_entropy(
                logits.reshape(-1, self.model.config.vocab_size),
                targets.reshape(-1),
            )
            loss = loss / cfg.gradient_accumulation_steps

        loss.backward()
        return loss.item() * cfg.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.eval_loader is None:
            return float("nan")
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for batch in self.eval_loader:
            x = batch.to(self.config.device)
            input_ids, targets = x[:, :-1], x[:, 1:]
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                logits, _ = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.model.config.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                )
            total_loss += loss.item()
            total_tokens += targets.numel()
        self.model.train()
        return total_loss / max(total_tokens, 1)

    def save_checkpoint(self, path: Optional[str] = None):
        path = path or os.path.join(self.config.output_dir, f"ar_step_{self.global_step}.pt")
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

        logger.info(f"AR training: steps {self.global_step} → {cfg.max_steps}")

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
                    f"[AR]   step {self.global_step:>7d} | loss {avg:.4f} | "
                    f"lr {lr:.2e} | {tps:.0f} tok/s"
                )
                accum_loss = 0.0

            if self.global_step % cfg.eval_interval == 0:
                val_loss = self.evaluate()
                val_ppl = math.exp(min(val_loss, 20))
                logger.info(f"  eval loss={val_loss:.4f} ppl={val_ppl:.2f}")
                if val_loss < self.best_eval_loss:
                    self.best_eval_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(cfg.output_dir, "ar_best.pt")
                    )

            if self.global_step % cfg.save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        logger.info("AR training complete.")
