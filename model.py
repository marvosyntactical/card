"""
model.py — Causal Transformer Language Model (shared by CARD and AR)
====================================================================

CRITICAL DESIGN NOTE:
    CARD does NOT modify the transformer architecture. There is no timestep
    embedding, no Adaptive LayerNorm, nothing diffusion-specific in the model.
    CARD's Eq. 5 conditions on x^t_{<n} only — the model infers noise level
    from the pattern of [MASK] tokens in the input. This is confirmed by the
    RADD result that "explicit time dependency in the input was not strictly
    necessary for mathematical validity" (Section 2.1).

    Therefore, CARD and AR use THE EXACT SAME MODEL. The only difference
    is the training procedure (soft tail masking + context-aware reweighting).

Architecture (Table 6):
    - Pre-norm causal transformer (GPT-style)
    - SiLU activation in FFN
    - Rotary Position Embeddings (RoPE)
    - Weight-tied LM head
    - Flash Attention via F.scaled_dot_product_attention
    - KV caching for efficient generation

Paper config (1B params): layers=33, d=1536, d_ff=4096, heads=24, vocab=50368
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Model hyperparameters. Defaults match Table 6 of the CARD paper."""
    vocab_size: int = 50368
    n_layers: int = 33
    d_model: int = 1536
    n_heads: int = 24
    d_ff: int = 4096
    max_len: int = 8192
    dropout: float = 0.0
    # The last token in the vocabulary is reserved for [MASK].
    # AR training never uses it; CARD training masks tokens with it.
    # Keeping it in both models ensures identical architectures for fair comparison.

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def mask_token_id(self) -> int:
        return self.vocab_size - 1


# =============================================================================
# KV Cache
# =============================================================================

class KVCache:
    """
    Pre-allocated key-value cache for autoregressive generation.

    Avoids repeated torch.cat() calls (which are O(n) copies each step)
    by pre-allocating a buffer of max_len and tracking the current length.

    Two usage modes:
      1. AR generation: call update() each step to append one token's KV.
      2. CARD denoising: use the prefix cache as read-only during denoising
         iterations, then call update() once with the finalized block.
    """

    def __init__(
        self,
        batch_size: int,
        n_heads: int,
        max_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.max_len = max_len
        self.seq_len = 0
        # Pre-allocate buffers: (B, H, max_len, head_dim)
        self.k = torch.zeros(batch_size, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch_size, n_heads, max_len, head_dim, device=device, dtype=dtype)

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple:
        """
        Append new key-value pairs and return the full cached KV.

        Args:
            k_new: (B, H, new_len, head_dim)
            v_new: (B, H, new_len, head_dim)
        Returns:
            (k_full, v_full): each (B, H, cached_len + new_len, head_dim)
        """
        new_len = k_new.size(2)
        end = self.seq_len + new_len
        assert end <= self.max_len, f"Cache overflow: {end} > {self.max_len}"

        self.k[:, :, self.seq_len:end] = k_new
        self.v[:, :, self.seq_len:end] = v_new
        self.seq_len = end

        return self.k[:, :, :end], self.v[:, :, :end]

    def get(self) -> tuple:
        """Return current cached KV without modification."""
        return self.k[:, :, :self.seq_len], self.v[:, :, :self.seq_len]

    def clone(self) -> "KVCache":
        """Create a deep copy (useful for CARD: snapshot prefix before denoising)."""
        new = KVCache.__new__(KVCache)
        new.max_len = self.max_len
        new.seq_len = self.seq_len
        new.k = self.k.clone()
        new.v = self.v.clone()
        return new

    @property
    def device(self):
        return self.k.device

    @property
    def dtype(self):
        return self.k.dtype


# =============================================================================
# Rotary Position Embeddings (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2021).

    Encodes position information by rotating pairs of dimensions in Q and K.
    Advantages over absolute position embeddings for KV caching:
      - Cached K vectors already have their rotations baked in, no recomputation
      - Position is determined by the rotation applied at encoding time
      - Better length generalization than learned absolute embeddings
    """

    def __init__(self, head_dim: int, max_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # Precompute the frequency bands: θ_i = base^{-2i/d}
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin tables for all positions up to max_len
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t = torch.arange(max_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (max_len, head_dim/2)
        # Duplicate for paired dimensions: [θ0, θ1, ..., θ0, θ1, ...]
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.

        Args:
            x: (B, H, L, head_dim) query or key tensor
            offset: position offset (= length of KV cache for generation)
        Returns:
            Rotated tensor of the same shape.
        """
        seq_len = x.size(2)
        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs of dimensions: [x0, x1, x2, x3, ...] → [-x_{d/2}, ..., x0, ...]"""
        d = x.size(-1)
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)


# =============================================================================
# Transformer Components
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE and KV caching.

    Uses F.scaled_dot_product_attention which dispatches to:
      - Flash Attention 2 (when available, CUDA)
      - Memory-efficient attention (fallback)
      - Math attention (CPU)

    For KV caching: is_causal=True with different Q/K lengths creates the
    correct mask where Q[i] attends to K[j] for j <= i + past_len.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        # Fused QKV projection — single matmul is faster than three separate ones
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.max_len)
        self.attn_dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple:
        """
        Args:
            x:          (B, L, D) hidden states for current tokens
            pos_offset: absolute position of the first token in x
            kv_cache:   optional KVCache to use/update
        Returns:
            output:  (B, L, D) attention output
            kv_cache: updated KVCache (or None if no cache provided)
        """
        B, L, D = x.shape

        # Compute Q, K, V for current tokens
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, L, H, head_dim)
        q = q.transpose(1, 2)  # (B, H, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K (position-aware rotation)
        q = self.rotary(q, offset=pos_offset)
        k = self.rotary(k, offset=pos_offset)

        # KV caching: prepend past keys/values
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)
            # k, v now have shape (B, H, past_len + L, head_dim)

        # Scaled dot-product attention with causal mask
        # When Q.size(2) != K.size(2), is_causal=True correctly masks such that
        # Q[i] attends to K[j] for j <= i + (K_len - Q_len), i.e. j <= i + past_len
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=dropout_p
        )

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_out), kv_cache


class FeedForward(nn.Module):
    """
    Position-wise feedforward with SiLU activation (Table 6).

    Standard 2-layer FFN: up_proj → SiLU → down_proj.
    The paper specifies "Activation Function: SiLU" and
    "Intermediate Size: 4096", indicating a standard (non-gated) FFN.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block: LN → Attention → residual → LN → FFN → residual.

    Pre-norm (as opposed to post-norm) is standard in modern LLMs and
    provides more stable training dynamics.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_offset: int = 0,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple:
        # Attention with pre-norm and residual
        h, kv_cache = self.attn(self.norm1(x), pos_offset, kv_cache)
        x = x + self.dropout(h)
        # FFN with pre-norm and residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, kv_cache


# =============================================================================
# Full Model
# =============================================================================

class CausalLM(nn.Module):
    """
    Causal Language Model shared by both CARD and AR training.

    This is a standard GPT-style model. There is NOTHING diffusion-specific
    in the architecture. CARD's Eq. 5 conditions only on x^t_{<n}, with no
    explicit timestep — the model infers noise level from the [MASK] pattern.

    The ONLY vocabulary difference: the last token (vocab_size - 1) is reserved
    for [MASK]. AR training never produces this token; CARD training injects it.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (no absolute position embedding — we use RoPE)
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head, weight-tied with token embedding
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2*n_layers) for stable deep training
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            nn.init.normal_(block.ff.down_proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_offset: int = 0,
        kv_caches: Optional[list] = None,
    ) -> tuple:
        """
        Forward pass through the causal transformer.

        Args:
            input_ids: (B, L) token ids
            pos_offset: starting position for RoPE (= cache length for generation)
            kv_caches: optional list of KVCache objects (one per layer)
        Returns:
            logits: (B, L, vocab_size)
            kv_caches: list of updated KVCache objects (or None)
        """
        B, L = input_ids.shape
        h = self.emb_dropout(self.tok_emb(input_ids))

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, cache_i = block(h, pos_offset, cache_i)
            new_caches.append(cache_i)

        logits = self.lm_head(self.final_norm(h))
        out_caches = new_caches if kv_caches is not None else None
        return logits, out_caches

    # =========================================================================
    # Generation utilities
    # =========================================================================

    def init_kv_caches(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float16
    ) -> list:
        """Allocate empty KV caches for all layers."""
        return [
            KVCache(batch_size, self.config.n_heads, self.config.max_len,
                    self.config.head_dim, device, dtype)
            for _ in range(self.config.n_layers)
        ]

    @torch.no_grad()
    def generate_ar(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Standard autoregressive generation with KV caching.

        Each step: one forward pass over ONE token → one new token.
        Total forward passes = max_new_tokens.

        Args:
            prompt_ids: (1, P) prompt token ids
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering (0 = no filtering)
        Returns:
            (1, P + max_new_tokens) full sequence
        """
        self.eval()
        device = prompt_ids.device
        B = prompt_ids.size(0)
        dtype = next(self.parameters()).dtype

        # Encode full prompt to build KV cache
        kv_caches = self.init_kv_caches(B, device, dtype)
        logits, kv_caches = self(prompt_ids, pos_offset=0, kv_caches=kv_caches)

        # Sample first new token from last position
        next_token = self._sample(logits[:, -1:], temperature, top_k)
        generated = [prompt_ids, next_token]
        pos = prompt_ids.size(1)

        for _ in range(max_new_tokens - 1):
            pos += 1
            logits, kv_caches = self(next_token, pos_offset=pos, kv_caches=kv_caches)
            next_token = self._sample(logits[:, -1:], temperature, top_k)
            generated.append(next_token)

        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate_card(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        block_size: int = 16,
        denoise_steps: int = 8,
        confidence_threshold: float = 0.9,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        CARD generation: confidence-based parallel decoding with KV caching.
        (Section 3.4, Eq. 9)

        Each "block" of K tokens is decoded in denoise_steps iterations.
        Effective speedup ≈ block_size / denoise_steps vs AR.

        Procedure per block:
          1. Snapshot the prefix KV cache
          2. Create K [MASK] tokens
          3. For each denoising step:
             - Forward pass over K tokens (using prefix cache, NOT updating it)
             - Unmask tokens where confidence > threshold
          4. Force-decode remaining masks on the last step
          5. Re-encode the finalized block to update the prefix cache

        The prefix cache is read-only during denoising because unmasking
        tokens changes the hidden states — cached KVs for partially-masked
        input would be stale. After finalizing, one clean forward pass
        updates the cache correctly.

        Args:
            prompt_ids: (1, P) prompt tokens
            max_new_tokens: total tokens to generate
            block_size: K — tokens per block
            denoise_steps: T_max — denoising iterations per block
            confidence_threshold: τ — unmask when max_prob > τ
            temperature: sampling temperature
        Returns:
            (1, P + generated) full sequence
        """
        self.eval()
        device = prompt_ids.device
        mask_id = self.config.mask_token_id
        B = prompt_ids.size(0)
        dtype = next(self.parameters()).dtype

        # Encode prompt to build initial KV cache
        kv_caches = self.init_kv_caches(B, device, dtype)
        _, kv_caches = self(prompt_ids, pos_offset=0, kv_caches=kv_caches)

        generated = [prompt_ids]
        current_len = prompt_ids.size(1)
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            K = min(block_size, max_new_tokens - tokens_generated)
            block = torch.full((B, K), mask_id, device=device, dtype=torch.long)

            # Snapshot prefix cache — denoising reads from this, doesn't modify it
            prefix_len = kv_caches[0].seq_len

            for step in range(denoise_steps):
                is_masked = (block == mask_id)
                if not is_masked.any():
                    break

                # Forward pass over just the block, using prefix cache as context
                # We create temporary caches that extend the prefix but don't persist
                temp_caches = [c.clone() for c in kv_caches]
                logits, _ = self(block, pos_offset=prefix_len, kv_caches=temp_caches)

                probs = (logits / temperature).softmax(dim=-1)
                max_probs, predicted = probs.max(dim=-1)

                # Unmask confident positions (Eq. 9)
                confident = is_masked & (max_probs > confidence_threshold)
                block[confident] = predicted[confident]

                # Last step: force-decode ALL remaining masks
                if step == denoise_steps - 1:
                    still_masked = (block == mask_id)
                    if still_masked.any():
                        for b in range(B):
                            mask_pos = still_masked[b].nonzero(as_tuple=True)[0]
                            if mask_pos.numel() > 0:
                                sampled = torch.multinomial(
                                    probs[b, mask_pos], num_samples=1
                                ).squeeze(-1)
                                block[b, mask_pos] = sampled

            # Finalize: re-encode the clean block to update KV cache correctly
            # The prefix cache still has seq_len = prefix_len, so this appends
            _, kv_caches = self(block, pos_offset=prefix_len, kv_caches=kv_caches)

            generated.append(block)
            tokens_generated += K
            current_len += K

        return torch.cat(generated, dim=1)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        """Sample from logits with temperature and top-k filtering."""
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=False).unsqueeze(-1)
        logits = logits / temperature
        if top_k > 0:
            topk_vals, _ = logits.topk(top_k, dim=-1)
            logits[logits < topk_vals[..., -1:]] = float("-inf")
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs.squeeze(1), num_samples=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
