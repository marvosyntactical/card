import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# CARD: Causal Autoregressive Diffusion Language Model
# =============================================================================
# Key architectural difference from standard MDLM:
#   - Causal (unidirectional) attention, NOT bidirectional
#   - This means the architecture is essentially a GPT + timestep conditioning
#   - KV caching works exactly as in standard autoregressive models
# =============================================================================


class SinusoidalTimestepEmbedding(nn.Module):
    """
    Embeds the continuous diffusion timestep t ∈ [0, 1] into a dense vector.
    Uses sinusoidal positional encoding (same idea as Vaswani et al.)
    followed by a 2-layer MLP to project into model dimension.
    
    This is the same component used in image diffusion models (DDPM, DiT).
    It's a tiny fraction of total parameters (~0.1%).
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
            t: (B,) float tensor, diffusion time in [0, 1]
        Returns:
            (B, d_model) timestep embedding
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=t.dtype)
            / half_dim
        )
        # (B, half_dim)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class CausalAdaLNBlock(nn.Module):
    """
    Transformer block with:
      1. CAUSAL self-attention (standard GPT-style triangular mask)
      2. Adaptive LayerNorm conditioned on diffusion timestep
    
    The causal mask is what distinguishes CARD from MDLM architecturally.
    Everything else (AdaLN, FFN) is standard.
    
    AdaLN injects timestep info via learned scale/shift on LayerNorm,
    following the DiT (Diffusion Transformer) approach. This tells each
    layer "how noisy is the input right now" so it can calibrate its
    denoising behavior.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Attention projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),  # Paper uses SiLU activation (Table 6)
            nn.Linear(d_ff, d_model, bias=False),
        )

        # AdaLN: elementwise_affine=False because scale/shift come from timestep
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Timestep -> (gamma1, beta1, gamma2, beta2) for adaptive normalization
        self.adaln_proj = nn.Linear(d_model, 4 * d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        kv_cache: tuple = None,
    ):
        """
        Args:
            x:        (B, L, D) token representations
            t_emb:    (B, D) timestep embedding
            kv_cache: optional (cached_K, cached_V) for inference
        Returns:
            x:        (B, L, D) updated representations
            kv_cache: (K, V) to store for next step
        """
        B, L, D = x.shape

        # Compute adaptive scale and shift from timestep
        # t_emb is (B, D), unsqueeze to (B, 1, D) for broadcasting over sequence
        adaln_params = self.adaln_proj(t_emb).unsqueeze(1)  # (B, 1, 4*D)
        gamma1, beta1, gamma2, beta2 = adaln_params.chunk(4, dim=-1)

        # ---- Causal Self-Attention ----
        h = self.norm1(x) * (1 + gamma1) + beta1

        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, L, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # KV caching for inference: prepend cached keys/values
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # Causal mask: each query position can only attend to positions <= itself
        # With KV cache, the query length may differ from key length
        # F.scaled_dot_product_attention handles this with is_causal=True
        # but for KV cache we need a custom mask
        if kv_cache is None:
            # Standard training or first inference pass: use built-in causal
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0
            )
        else:
            # Inference with KV cache: query attends to all cached + current
            # No future masking needed since we only have current positions as queries
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=0.0
            )

        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.dropout(self.out_proj(attn_out))

        # ---- Feedforward ----
        h = self.norm2(x) * (1 + gamma2) + beta2
        x = x + self.dropout(self.ff(h))

        return x, new_kv_cache


class CARDModel(nn.Module):
    """
    CARD: Causal Autoregressive Diffusion Language Model
    
    Architecture: GPT-style causal transformer + timestep conditioning via AdaLN.
    
    Key insight: This is architecturally almost identical to a standard GPT.
    The "diffusion" aspect lives in how training data is prepared (soft tail
    masking) and how the loss is weighted (context-aware reweighting), NOT
    in the architecture itself. This is why:
      - Training cost matches ARM (same attention pattern, same FLOPs)
      - KV caching works natively
      - You can initialize from a pretrained GPT checkpoint
    
    Config from paper (Table 6): 33 layers, d=1536, d_ff=4096, 24 heads, ~1B params
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
        self.mask_token_id = vocab_size - 1  # Convention: last token is [MASK]

        # Token + position embeddings (standard transformer)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Timestep embedding (diffusion-specific, tiny parameter overhead)
        self.time_emb = SinusoidalTimestepEmbedding(d_model)

        # Causal transformer blocks
        self.blocks = nn.ModuleList([
            CausalAdaLNBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output head: predicts p_θ(x_0,n | x^t_{<n}) for each position
        # This is identical to the LM head in GPT — just a linear projection to vocab
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optional: weight tying between input embeddings and output head
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        kv_caches: list = None,
    ):
        """
        Args:
            input_ids: (B, L) token ids, some may be mask_token_id
            t:         (B,) diffusion timestep in [0, 1]
            kv_caches: list of (K, V) tuples per layer, for inference
        Returns:
            logits:    (B, L, V) prediction logits for EACH position
            new_caches: updated KV caches
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device)

        # Standard transformer input: token emb + position emb
        h = self.tok_emb(input_ids) + self.pos_emb(positions)

        # Timestep conditioning: computed once, injected into every layer via AdaLN
        t_emb = self.time_emb(t)  # (B, d_model)

        new_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            h, new_cache = block(h, t_emb, kv_cache=cache_i)
            new_caches.append(new_cache)

        h = self.final_norm(h)

        # CRITICAL: the causal shift.
        # Position n's hidden state was computed from x^t_{≤n} (causal attention).
        # But we want to predict x_{n+1} (the NEXT token), not x_n.
        # This is the standard "shifted" prediction from GPT:
        #   logits[n] = prediction for position n+1
        # The paper's formulation: "each position predicts its original token
        # from the preceding noised context" — i.e., the input is shifted by 1.
        logits = self.lm_head(h)  # (B, L, V)

        return logits, new_caches


# =============================================================================
# TRAINING: Soft Tail Masking + Context-Aware Reweighting
# =============================================================================
# This is where CARD's novelty primarily lives — NOT in the architecture,
# but in the noise injection and loss weighting strategy.
# =============================================================================


def soft_tail_masking(
    x0: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    tail_factor: float = 1.5,
) -> torch.Tensor:
    """
    Soft Tail Masking (Section 3.2, Algorithm 1 lines 6-11).
    
    Instead of masking tokens uniformly (MDLM) or in fixed blocks (BD3LM),
    concentrate masks at the TAIL of the sequence within a relaxed window.
    
    Why? Under causal attention, early tokens have little context.
    If you mask token 2, the model sees [MASK] and has to guess — it's hopeless.
    But if you mask token 500, the model has 499 (mostly clean) predecessors.
    
    The "soft" part: the window is wider than the number of masks, so you get
    a MIX of clean and noisy tokens in the tail region. This preserves local
    context (clean neighbors) even within the corrupted zone.
    
    Args:
        x0: (B, L) clean token ids
        t:  (B,) noise level in [0, 1]. Higher t = more masks.
        mask_token_id: id for [MASK] token
        tail_factor: λ in the paper. Controls window width relative to mask count.
                     λ=1.0 → strict tail (solid block of masks)
                     λ>1.0 → relaxed window (masks spread over wider tail region)
    Returns:
        x_t: (B, L) corrupted sequence with masks concentrated at the tail
    """
    B, L = x0.shape
    x_t = x0.clone()

    for b in range(B):
        # Number of tokens to mask: proportional to t
        N = max(1, int(L * t[b].item()))

        # Window size: wider than N by factor λ
        # This creates the "soft" part — N masks spread over W > N positions
        W = min(L, int(N * tail_factor))

        # Window covers the last W positions of the sequence
        window_start = L - W
        window_indices = torch.arange(window_start, L, device=x0.device)

        # Randomly select N positions within the window to mask
        # (rather than masking all of them, which would be "strict tail")
        perm = torch.randperm(W, device=x0.device)[:N]
        mask_positions = window_indices[perm]

        x_t[b, mask_positions] = mask_token_id

    return x_t


def compute_context_aware_weights(
    x_t: torch.Tensor,
    mask_token_id: int,
    beta: float = 1.0,
    decay: float = 0.5,
) -> torch.Tensor:
    """
    Context-Aware Reweighting (Section 3.3, Algorithm 1 lines 13-17).
    
    Computes per-token loss weights based on how "ambiguous" each position's
    causal context is. Positions whose predecessors are heavily masked get
    lower weight, because predicting from noise is uninformative.
    
    Three factors determine ambiguity:
      1. Quantity: total number of masked tokens in the context
      2. Distance: nearby masks matter more (exponential decay)
      3. Density: consecutive masks are worse than isolated ones
         (a run of masks completely severs local dependencies)
    
    These are unified into a local ambiguity score S^local_n (Eq. 6-7):
        C_i = 𝟙[x_i is MASK] · (1 + 𝟙[x_{i-1} is MASK])
        S^local_n = Σ_{i=1}^{n} C_i · (1-p)^{n-i}
    
    Then the weight is w_n = 1 / (β + S^local_n), which downweights
    high-ambiguity positions.
    
    This is analogous to inverse-variance weighting in statistics:
    noisy predictions get less weight. The paper proves (Proposition 1)
    that this minimizes gradient variance.
    
    Args:
        x_t:           (B, L) corrupted token ids
        mask_token_id: id for [MASK] token
        beta:          smoothing constant (prevents division by zero)
        decay:         p in the paper, controls how fast distant noise
                       becomes irrelevant. 0.5 means noise 10 tokens away
                       has ~1/1000 the impact of adjacent noise.
    Returns:
        weights: (B, L) per-token loss weights
    """
    B, L = x_t.shape

    is_mask = (x_t == mask_token_id).float()  # (B, L)

    # Density term: C_i = is_mask[i] * (1 + is_mask[i-1])
    # Consecutive masks get cost 2, isolated masks get cost 1, clean tokens get 0
    prev_mask = F.pad(is_mask[:, :-1], (1, 0), value=0.0)  # shift right, pad with 0
    C = is_mask * (1.0 + prev_mask)  # (B, L)

    # Compute S^local_n via exponentially-decayed cumulative sum
    # S^local_n = Σ_{i=1}^{n} C_i · (1-p)^{n-i}
    #
    # This is a causal exponential moving sum. We compute it iteratively
    # because the exponential decay makes it a simple recurrence:
    #   S_n = C_n + (1-p) · S_{n-1}
    retain = 1.0 - decay
    S_local = torch.zeros(B, L, device=x_t.device)
    running = torch.zeros(B, device=x_t.device)
    for n in range(L):
        running = C[:, n] + retain * running
        S_local[:, n] = running

    # Inverse weighting: high ambiguity → low weight
    weights = 1.0 / (beta + S_local)

    return weights


def card_training_step(
    model: CARDModel,
    x0: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    tail_factor: float = 1.5,
    beta: float = 1.0,
    decay: float = 0.5,
):
    """
    One CARD training step (Algorithm 1).
    
    The procedure:
      1. Sample a random noise level t ~ U[0,1]
      2. Corrupt x0 via soft tail masking → x_t
      3. Feed x_t through the causal transformer
      4. Compute cross-entropy loss with context-aware weights
      5. Backprop
    
    Key insight: this is almost identical to ARM training!
    Same forward pass, same loss function (cross-entropy), same attention mask.
    The only differences are:
      - Input is corrupted (some tokens are [MASK])
      - Loss is reweighted per-token based on context quality
      - The model also receives a timestep t
    
    At t=0: no masks, this IS autoregressive training
    At t=1: all masks, generating from scratch
    
    This is why CARD matches ARM training speed (Table 6 comparison).
    
    Args:
        model:       CARDModel instance
        x0:          (B, L) clean token sequences
        optimizer:   optimizer instance
        tail_factor: λ for soft tail masking window width
        beta:        β for context-aware reweighting smoothing
        decay:       p for distance decay in reweighting
    Returns:
        loss:        scalar loss value (for logging)
    """
    B, L = x0.shape
    device = x0.device

    # Step 1: Sample noise level uniformly
    t = torch.rand(B, device=device)  # (B,) each sample gets its own t

    # Step 2: Corrupt input via soft tail masking
    # This replaces a tail-biased subset of tokens with [MASK]
    x_t = soft_tail_masking(x0, t, model.mask_token_id, tail_factor)

    # Step 3: Shifted causal input/target setup
    # Input:  x_t[0], x_t[1], ..., x_t[L-2]  (positions 0 to L-2)
    # Target: x_0[1], x_0[2], ..., x_0[L-1]  (positions 1 to L-1)
    # This is the standard GPT shift: predict next token from context
    input_ids = x_t[:, :-1]   # (B, L-1) — corrupted prefix
    targets = x0[:, 1:]       # (B, L-1) — clean next tokens

    # Step 4: Forward pass (identical cost to ARM — same causal attention)
    logits, _ = model(input_ids, t)  # (B, L-1, V)

    # Step 5: Compute context-aware weights on the INPUT sequence
    # We weight based on how corrupted the context is that each position sees
    weights = compute_context_aware_weights(
        input_ids, model.mask_token_id, beta, decay
    )  # (B, L-1)

    # Step 6: Weighted cross-entropy loss
    # Standard CE but each position gets a different weight
    # based on how informative its context is
    ce_loss = F.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
        reduction='none',
    ).reshape(B, -1)  # (B, L-1)

    # Apply context-aware weights and average
    loss = (weights * ce_loss).sum() / weights.sum()

    # Step 7: Standard backprop (nothing diffusion-specific here)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# =============================================================================
# INFERENCE: Confidence-Based Parallel Decoding with KV Caching
# =============================================================================

@torch.no_grad()
def card_generate(
    model: CARDModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 256,
    block_size: int = 16,
    denoise_steps: int = 8,
    confidence_threshold: float = 0.9,
    temperature: float = 1.0,
):
    """
    CARD inference with KV caching and confidence-based parallel decoding.
    (Section 3.4, Eq. 9)
    
    This is where CARD's speedup comes from:
    
    ARM generates 1 token per forward pass.
    CARD generates up to `block_size` tokens per iteration, where each
    iteration takes `denoise_steps` forward passes (but only over the
    new block, not the full sequence, thanks to KV caching).
    
    Effective speedup ≈ block_size / denoise_steps.
    Paper reports 1.7× to 4.0× depending on settings.
    
    The procedure:
      1. Encode the prompt, cache KV pairs
      2. Append K [MASK] tokens
      3. Iteratively denoise:
         - Forward pass over mask tokens only (using cached KV for prefix)
         - Unmask tokens where confidence > threshold
         - Repeat for denoise_steps iterations
      4. Force-decode any remaining masks
      5. Add generated block to KV cache
      6. Repeat from step 2 until done
    
    The KV cache is the critical enabler. Without causal attention,
    you'd have to recompute the full sequence at every denoise step.
    With it, each denoise step only processes the K new positions.
    
    Args:
        model:                CARDModel
        prompt_ids:           (1, P) prompt token ids
        max_new_tokens:       total tokens to generate
        block_size:           K — how many [MASK] tokens to append per block
        denoise_steps:        T_max — max denoising iterations per block
        confidence_threshold: τ — unmask when max prob exceeds this
        temperature:          sampling temperature
    Returns:
        generated_ids: (1, P + generated) full sequence
    """
    device = prompt_ids.device
    mask_id = model.mask_token_id

    # Step 1: Encode the prompt to build initial KV cache
    # t=0 for the prompt (it's clean, no noise)
    t_prompt = torch.zeros(1, device=device)
    _, kv_caches = model(prompt_ids, t_prompt)

    generated = prompt_ids.clone()
    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        # Step 2: Determine block size (may be smaller for last block)
        K = min(block_size, max_new_tokens - tokens_generated)

        # Initialize block as all [MASK]
        block = torch.full((1, K), mask_id, device=device, dtype=torch.long)

        # Step 3: Iterative denoising of this block
        for step in range(denoise_steps):
            # How many masks remain?
            is_masked = (block == mask_id)
            if not is_masked.any():
                break  # All tokens decoded, done with this block

            # Forward pass: only the block tokens, using cached prefix KV
            # t should reflect the "noise level" — decreasing each step
            # Heuristic: t goes from ~1 to ~0 across denoise steps
            t_val = torch.tensor(
                [(denoise_steps - step) / denoise_steps], device=device
            )

            logits, new_caches_block = model(block, t_val, kv_caches)
            # logits: (1, K, V)

            # Sample or take argmax
            probs = (logits / temperature).softmax(dim=-1)  # (1, K, V)
            max_probs, predicted = probs.max(dim=-1)        # (1, K) each

            # Unmask confident positions (Eq. 9)
            confident = max_probs[0] > confidence_threshold  # (K,)
            should_unmask = is_masked[0] & confident

            if should_unmask.any():
                block[0, should_unmask] = predicted[0, should_unmask]

            # On the last step, force-decode ALL remaining masks
            if step == denoise_steps - 1:
                still_masked = (block == mask_id)
                if still_masked.any():
                    # Sample from the distribution for remaining positions
                    sampled = torch.multinomial(
                        probs[0][still_masked[0]], num_samples=1
                    ).squeeze(-1)
                    block[0, still_masked[0]] = sampled

        # Step 4: Block is fully decoded. Update KV cache.
        # Re-encode the finalized block to get correct KV entries
        t_zero = torch.zeros(1, device=device)
        _, kv_caches = model(block, t_zero, kv_caches)

        # Append to generated sequence
        generated = torch.cat([generated, block], dim=1)
        tokens_generated += K

    return generated
