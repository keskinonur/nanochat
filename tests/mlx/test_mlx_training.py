"""
Test MLX training with a tiny model
"""
import os
import time
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from nanochat.gpt import GPT, GPTConfig
from nanochat.muon import Muon
from nanochat.adamw import AdamW

print("Testing MLX training pipeline...")

# Create tiny model
config = GPTConfig(
    sequence_len=128,
    vocab_size=1000,
    n_layer=2,
    n_head=2,
    n_kv_head=2,
    n_embd=128
)

model = GPT(config)
model.init_weights()

# tree_flatten returns list of (key, value) tuples
flat_params = tree_flatten(model.parameters())
num_params = sum(v.size for k, v in flat_params)
print(f"Model parameters: {num_params:,}")

# Training settings
batch_size = 2
seq_len = 128
num_steps = 5
lr = 0.01

print(f"\nTraining for {num_steps} steps...")

# Simple training loop
for step in range(num_steps):
    t0 = time.time()

    # Create dummy batch
    x = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
    y = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

    # Define loss function
    def loss_fn(params):
        model.update(params)
        return model(x, y)

    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss_val, grads = loss_and_grad_fn(model.parameters())

    # Simple SGD update
    current_params = model.parameters()
    updated_params = tree_map(
        lambda p, g: p - lr * g,
        current_params,
        grads
    )
    model.update(updated_params)

    # Force evaluation
    mx.eval(loss_val)

    t1 = time.time()
    dt = (t1 - t0) * 1000  # ms

    print(f"Step {step+1}/{num_steps} | Loss: {float(loss_val):.4f} | Time: {dt:.2f}ms")

print("\nTraining test passed!")
