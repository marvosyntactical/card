"""
CARD Model Architecture: Causal Autoregressive Diffusion Language Model
=========================================================================
Paper: "Causal Autoregressive Diffusion Language Model" (Ruan et al., 2026)

The architecture is almost identical to a standard GPT — causal (unidirectional)
attention with a next-token prediction head. The only additions for diffusion:
  1. A timestep embedding (sinusoidal + MLP) injected via Adaptive LayerNorm
  2. A [MASK] token in the vocabulary for the absorbing-state diffusion process

This means:
  - Training cost is identical to a standard autoregressive model
  - KV caching works natively (unlike bidirectional MDLM)
  - You could even initialize from a pretrained GPT checkpoint
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Maps the continuous diffusion timestep t ∈ [0, 1] to a dense vector.

    Uses the same sinusoidal encoding as positional embeddings (Vaswani et al.),
    followed by a 2-layer MLP. This is standard across all diffusion models
    (DDPM, DiT, etc.) and contributes <0.1% of total parameters.

    The timestep tells the model "how noisy is the input right now?" so it
    can calibrate its denoising behavior — at t≈0 it acts like a standard LM,
    at t≈1 it must generate from scratch.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float tensor in [0, 1]
        Returns:
            (B, d_model) timestep embedding
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        args = t[:, None] * freqs[None, :]  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class CausalAdaLNBlock(nn.Module):
    """
    Transformer block with causal self-attention and Adaptive LayerNorm.

    Two key design choices:
      1. CAUSAL attention mask — standard GPT triangular mask. This is what
         distinguishes CARD from MDLM (which uses full/bidirectional attention).
      2. Adaptive LayerNorm (AdaLN) — the timestep embedding modulates the
         LayerNorm via learned scale (γ) and shift (β) parameters. This is
         the DiT approach and is more expressive than simply adding the timestep
         to the input embeddings.

    The causal mask is the entire reason KV caching works: past key-value
    pairs never need to be recomputed when new tokens are appended.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Fused QKV projection for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # SiLU activation as specified in Table 6 of the paper
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

        # LayerNorm WITHOUT learnable affine params — scale/shift come from AdaLN
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # AdaLN projection: timestep → (γ1, β1, γ2, β2) for the two norms
        self.adaln_proj = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, L, D) token representations
            t_emb: (B, D)   timestep embedding (shared across all positions)
        Returns:
            (B, L, D) updated representations
        """
        B, L, D = x.shape

        # AdaLN: compute per-layer scale and shift from timestep
        adaln_params = self.adaln_proj(t_emb).unsqueeze(1)  # (B, 1, 4*D)
        gamma1, beta1, gamma2, beta2 = adaln_params.chunk(4, dim=-1)

        # --- Causal Self-Attention ---
        h = self.norm1(x) * (1 + gamma1) + beta1

        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch's native SDPA with is_causal=True uses Flash Attention
        # under the hood when available, matching the paper's use of Flash Attn 2
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.dropout(self.out_proj(attn_out))

        # --- Feedforward ---
        h = self.norm2(x) * (1 + gamma2) + beta2
        x = x + self.dropout(self.ff(h))

        return x


class CARDModel(nn.Module):
    """
    CARD: Causal Autoregressive Diffusion Language Model.

    Architecture = GPT + timestep conditioning via AdaLN.

    Paper configuration (Table 6, ~1B parameters):
        layers=33, d_model=1536, d_ff=4096, heads=24, vocab=50368

    The "diffusion" aspect lives entirely in:
      - How training data is prepared (soft tail masking)
      - How the loss is weighted (context-aware reweighting)
    NOT in the architecture. This is why training cost = ARM cost.
    """

    def __init__(
        self,
        vocab_size: int = 50368,
        d_model: int = 1536,
        n_layers: int = 33,
        n_heads: int = 24,
        d_ff: int = 4096,
        max_len: int = 8192,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Convention: the last token id in the vocabulary is [MASK]
        self.mask_token_id = vocab_size - 1

        # --- Embeddings ---
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = SinusoidalTimestepEmbedding(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # --- Transformer backbone ---
        self.blocks = nn.ModuleList([
            CausalAdaLNBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # --- LM head (weight-tied with input embeddings) ---
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Standard GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict clean tokens from noisy causal context.

        This is the standard GPT forward pass plus timestep conditioning.
        The shifted prediction (position n predicts token n+1) is handled
        externally in the training loop, not inside the model.

        Args:
            input_ids: (B, L) token ids, some may be mask_token_id
            t:         (B,)   diffusion timestep in [0, 1]
        Returns:
            logits: (B, L, V) unnormalized prediction scores for each position
        """
        B, L = input_ids.shape
        device = input_ids.device

        positions = torch.arange(L, device=device)
        h = self.tok_emb(input_ids) + self.pos_emb(positions)
        h = self.emb_dropout(h)

        # Timestep conditioning: computed once, injected into every block
        t_emb = self.time_emb(t)  # (B, d_model)

        for block in self.blocks:
            h = block(h, t_emb)

        h = self.final_norm(h)
        logits = self.lm_head(h)  # (B, L, V)

        return logits

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
