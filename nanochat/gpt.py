"""
GPT model (MLX port for Apple Silicon)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from nanochat.common import get_dist_info, print0


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (MQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    # MLX doesn't have rms_norm built-in, so we implement it
    var = mx.mean(mx.square(x), axis=-1, keepdims=True)
    return x * mx.rsqrt(var + 1e-6)


def apply_rotary_emb(x, cos, sin):
    # x shape: (B, T, H, D) or (B, H, T, D) depending on usage
    # cos, sin shape: (1, T, 1, D//2)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos - x2 * sin  # rotate pairs of dims
    y2 = x1 * sin + x2 * cos
    out = mx.concatenate([y1, y2], axis=-1)
    return out


def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep) equivalent"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    # MLX doesn't have repeat_interleave, so we expand and reshape
    x = mx.expand_dims(x, axis=2)  # (bs, n_kv_heads, 1, slen, head_dim)
    x = mx.broadcast_to(x, (bs, n_kv_heads, n_rep, slen, head_dim))
    x = mx.reshape(x, (bs, n_kv_heads * n_rep, slen, head_dim))
    return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def __call__(self, x, cos_sin, kv_cache):
        B, T, C = x.shape

        # Project the input to get queries, keys, and values
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Transpose to (B, H, T, D) for attention
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # Apply KV cache
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.shape[2]
        Tk = k.shape[2]

        # Apply MQA: replicate the key/value heads for each query head
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # Attention
        if kv_cache is None or Tq == Tk:
            # Training mode: causal attention
            y = self._causal_attention(q, k, v)
        elif Tq == 1:
            # Inference mode with single query
            y = self._full_attention(q, k, v)
        else:
            # Inference mode with multiple queries
            y = self._chunked_attention(q, k, v, Tq, Tk)

        # Re-assemble heads and project
        y = mx.transpose(y, (0, 2, 1, 3))  # (B, T, H, D)
        y = mx.reshape(y, (B, T, -1))
        y = self.c_proj(y)
        return y

    def _causal_attention(self, q, k, v):
        """Causal self-attention for training"""
        B, H, Tq, D = q.shape
        Tk = k.shape[2]

        # Compute attention scores
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(D)

        # Apply causal mask
        mask = mx.tril(mx.ones((Tq, Tk)))
        mask = mx.where(mask == 0, float('-inf'), 0.0)
        scores = scores + mask

        # Softmax and weighted sum
        attn = mx.softmax(scores, axis=-1)
        y = attn @ v
        return y

    def _full_attention(self, q, k, v):
        """Full attention for single-token inference"""
        D = q.shape[-1]
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(D)
        attn = mx.softmax(scores, axis=-1)
        y = attn @ v
        return y

    def _chunked_attention(self, q, k, v, Tq, Tk):
        """Attention for multi-token inference with cache"""
        D = q.shape[-1]
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(D)

        # Create attention mask
        prefix_len = Tk - Tq
        mask = mx.zeros((Tq, Tk))
        if prefix_len > 0:
            mask[:, :prefix_len] = 0.0  # Can attend to prefix
        # Causal mask within the chunk
        causal_mask = mx.tril(mx.ones((Tq, Tq)))
        mask[:, prefix_len:] = mx.where(causal_mask == 0, float('-inf'), 0.0)

        scores = scores + mask
        attn = mx.softmax(scores, axis=-1)
        y = attn @ v
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))  # relu^2 activation
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin

    def init_weights(self):
        """Initialize weights using custom scheme"""
        # Will be called after model creation
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                fan_out = module.weight.shape[0]
                fan_in = module.weight.shape[1]
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                module.weight = mx.random.normal(module.weight.shape, scale=std)
            elif isinstance(module, nn.Embedding):
                module.weight = mx.random.normal(module.weight.shape, scale=1.0)

        # Apply to all modules
        self.apply(_init_weights)

        # Zero out specific weights
        self.lm_head.weight = mx.zeros_like(self.lm_head.weight)
        for block in self.h:
            block.mlp.c_proj.weight = mx.zeros_like(block.mlp.c_proj.weight)
            block.attn.c_proj.weight = mx.zeros_like(block.attn.c_proj.weight)

        # Recompute rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        """Precompute rotary embeddings"""
        channel_range = mx.arange(0, head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.expand_dims(t, axis=1) * mx.expand_dims(inv_freq, axis=0)
        cos, sin = mx.cos(freqs), mx.sin(freqs)
        # Add batch and head dims
        cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=2)  # (1, T, 1, D//2)
        sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=2)
        return cos, sin

    def get_device(self):
        """MLX uses unified memory, return default device"""
        return mx.default_device()

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model"""
        # Count parameters
        params = tree_flatten(self.parameters())
        nparams = sum(p.size for p in params if isinstance(p, mx.array))
        nparams_embedding = self.wte.weight.size

        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """Setup optimizers for different parameter groups"""
        import mlx.optimizers as optim
        from nanochat.muon import Muon

        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate parameters into groups
        matrix_params = []
        for block in self.h:
            matrix_params.extend([
                block.attn.c_q.weight, block.attn.c_k.weight,
                block.attn.c_v.weight, block.attn.c_proj.weight,
                block.mlp.c_fc.weight, block.mlp.c_proj.weight
            ])

        embedding_params = [self.wte.weight]
        lm_head_params = [self.lm_head.weight]

        # Scale LR by model dimension
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Create AdamW optimizer for embeddings
        adamw_optimizer = optim.AdamW(
            learning_rate=embedding_lr * dmodel_lr_scale,
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=weight_decay
        )

        # Create Muon optimizer for matrix params
        muon_optimizer = Muon(
            learning_rate=matrix_lr,
            momentum=0.95
        )

        return [adamw_optimizer, muon_optimizer], [embedding_params + lm_head_params, matrix_params]

    def __call__(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.shape

        # Get rotary embeddings
        assert T <= self.cos.shape[1], f"Sequence length {T} exceeds rotary cache {self.cos.shape[1]}"

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        # Forward through transformer
        x = self.wte(idx)
        x = norm(x)
        for block in self.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Compute logits
        softcap = 15
        if targets is not None:
            # Training mode: compute loss
            logits = self.lm_head(x)
            logits = softcap * mx.tanh(logits / softcap)  # logits softcap

            # Cross entropy loss
            logits_flat = mx.reshape(logits, (-1, logits.shape[-1]))
            targets_flat = mx.reshape(targets, (-1,))

            # Mask out -1 (ignore index)
            mask = targets_flat != -1
            logits_masked = logits_flat[mask]
            targets_masked = targets_flat[mask]

            loss = nn.losses.cross_entropy(logits_masked, targets_masked, reduction=loss_reduction)
            return loss
        else:
            # Inference mode: return logits
            logits = self.lm_head(x)
            logits = softcap * mx.tanh(logits / softcap)
            return logits

    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        Simple implementation for batch size 1.
        """
        assert isinstance(tokens, list)
        mx.random.seed(seed)

        ids = mx.array([tokens])  # Add batch dim
        for _ in range(max_tokens):
            logits = self(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)

            if top_k is not None:
                # Top-k filtering
                top_logits, top_indices = mx.topk(logits, min(top_k, logits.shape[-1]))
                logits = mx.full_like(logits, float('-inf'))
                logits = mx.scatter(logits, top_indices, top_logits, axis=-1)

            if temperature > 0:
                logits = logits / temperature
                probs = mx.softmax(logits, axis=-1)
                next_ids = mx.random.categorical(probs, num_samples=1)
            else:
                next_ids = mx.argmax(logits, axis=-1, keepdims=True)

            ids = mx.concatenate((ids, next_ids), axis=1)
            token = int(next_ids[0, 0])
            yield token
