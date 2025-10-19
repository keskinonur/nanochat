"""
MLX port of loss evaluation functions.
"""

import math
import mlx.core as mx


def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Calculate bits per byte (bpb) metric - vocab size independent.

    This normalizes loss by the number of bytes that target tokens represent,
    making it comparable across different vocab sizes.

    Args:
        model: The GPT model
        batches: Iterator yielding (x, y) batches
        steps: Number of batches to evaluate
        token_bytes: mx.array of shape (vocab_size,) with byte count per token
                     (0 for special tokens to exclude from metric)

    Returns:
        bpb: Bits per byte metric
    """
    # Initialize accumulators
    total_nats = 0.0
    total_bytes = 0

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)

        # Get per-token loss (no reduction)
        loss2d = model(x, y, loss_reduction="none")  # (B, T)
        loss_flat = mx.reshape(loss2d, (-1,))  # flatten
        y_flat = mx.reshape(y, (-1,))  # flatten

        # Check if we have any ignore_index tokens (< 0)
        has_ignored = bool(mx.any(y_flat < 0).item())

        if has_ignored:
            # More complex path: handle ignore_index tokens
            valid = y_flat >= 0
            # For invalid targets, use 0 as placeholder (won't be used due to mask)
            y_safe = mx.where(valid, y_flat, mx.zeros_like(y_flat))

            # Map valid targets to their byte length
            # Use take to index token_bytes (MLX doesn't support fancy indexing)
            num_bytes_flat = mx.take(token_bytes, y_safe)

            # Zero out bytes for invalid/special tokens
            num_bytes_flat = mx.where(
                valid, num_bytes_flat, mx.zeros_like(num_bytes_flat)
            )

            # Only count loss where num_bytes > 0
            mask = (num_bytes_flat > 0).astype(mx.float32)
            total_nats += float(mx.sum(loss_flat * mask).item())
            total_bytes += int(mx.sum(num_bytes_flat).item())
        else:
            # Fast path: no ignored targets
            num_bytes_flat = mx.take(token_bytes, y_flat)
            mask = (num_bytes_flat > 0).astype(mx.float32)
            total_nats += float(mx.sum(loss_flat * mask).item())
            total_bytes += int(mx.sum(num_bytes_flat).item())

    # Calculate bpb
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
