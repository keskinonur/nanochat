"""
Ultra-safe test script for MLX to avoid system crashes.

This script uses very conservative settings:
- Tiny model (2 layers)
- Very small batch size (1)
- Short sequences (128 tokens)
- Very few iterations (5)
- Explicit memory management

Run as: python -m scripts.safe_test_mlx
"""

import os
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import print0, print_banner

print_banner()
print0("=" * 60)
print0("ULTRA-SAFE MLX TEST")
print0("=" * 60)

# Ultra-conservative settings
depth = 2
max_seq_len = 128  # Very short sequences
device_batch_size = 1  # Single example at a time
vocab_size = 1000  # Small vocabulary
num_iterations = 5  # Just 5 iterations
grad_accum_steps = 1  # No gradient accumulation

print0(f"Settings:")
print0(f"  depth: {depth}")
print0(f"  max_seq_len: {max_seq_len}")
print0(f"  device_batch_size: {device_batch_size}")
print0(f"  vocab_size: {vocab_size}")
print0(f"  num_iterations: {num_iterations}")
print0("")

# Model configuration
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)

model_config = GPTConfig(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=depth,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim
)

print0("Initializing model...")
model = GPT(model_config)
model.init_weights()

# Count parameters
flat_params = tree_flatten(model.parameters())
num_params = sum(v.size for k, v in flat_params)
print0(f"Parameters: {num_params:,}")
print0("")

# Simple optimizer (just learning rate, no fancy stuff)
learning_rate = 0.001

print0("Starting training loop...")
print0("-" * 60)

for step in range(num_iterations):
    t0 = time.time()

    # Create small dummy batch
    x = mx.random.randint(0, vocab_size, (device_batch_size, max_seq_len))
    y = mx.random.randint(0, vocab_size, (device_batch_size, max_seq_len))

    # Define loss function
    def loss_fn(params):
        model.update(params)
        return model(x, y)

    # Compute loss and gradients
    loss_val, grads = mx.value_and_grad(loss_fn)(model.parameters())

    # Simple gradient descent update
    current_params = model.parameters()
    updated_params = tree_map(
        lambda p, g: p - learning_rate * g,
        current_params,
        grads
    )
    model.update(updated_params)

    # Force evaluation to complete the computation
    mx.eval(loss_val)
    mx.eval(updated_params)

    t1 = time.time()
    dt = t1 - t0

    print0(f"Step {step+1}/{num_iterations} | loss: {float(loss_val):.6f} | time: {dt*1000:.1f}ms")

    # Small delay to prevent overwhelming the system
    time.sleep(0.1)

print0("-" * 60)
print0("✅ Test completed successfully!")
print0("")
print0("Your MLX installation is working correctly.")
print0("If this worked without crashing, you can try:")
print0("  1. Slightly larger models (depth=4)")
print0("  2. Longer sequences (max_seq_len=256)")
print0("  3. Batch size 2-4")
print0("")
print0("For full training, you'll need to tune these parameters")
print0("based on what your system can handle.")
