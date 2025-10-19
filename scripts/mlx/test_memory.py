"""
Test script to check for memory leaks in SFT training loop.
Run with: python -m scripts.mlx.test_memory
"""

import os
import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from nanochat.common import get_base_dir, setup_default_logging
from nanochat.checkpoint_manager import load_model

setup_default_logging()
import logging
logger = logging.getLogger(__name__)

# Load the model
model, tokenizer, meta = load_model("mid", mx.gpu, phase="train", model_tag=None, step=None)

# Simple training data
def generate_dummy_batch(batch_size=1, seq_len=128):
    inputs = mx.random.randint(0, tokenizer.get_vocab_size(), (batch_size, seq_len))
    targets = mx.random.randint(0, tokenizer.get_vocab_size(), (batch_size, seq_len))
    return inputs, targets

# Loss function
def loss_fn(model, inputs, targets):
    return model(inputs, targets=targets)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Setup optimizers
from nanochat.adamw import AdamW
from nanochat.muon import Muon

optimizers, param_groups = model.setup_optimizers(
    unembedding_lr=0.004,
    embedding_lr=0.2,
    matrix_lr=0.02,
    weight_decay=0.0,
)

# Build parameter key sets
flat_model_params = dict(tree_flatten(model.parameters()))
param_group_keys = []
for param_group in param_groups:
    group_keys = set()
    for param in param_group:
        for key, model_param in flat_model_params.items():
            if id(model_param) == id(param):
                group_keys.add(key)
                break
    param_group_keys.append(group_keys)

logger.info("Starting memory leak test...")
logger.info("Running 100 training steps to monitor memory...")

for step in range(100):
    t0 = time.time()

    # Generate dummy batch
    inputs, targets = generate_dummy_batch()

    # Forward and backward
    loss, grads = loss_and_grad_fn(model, inputs, targets)

    # Apply optimizer updates
    flat_model_params = dict(tree_flatten(model.parameters()))
    flat_grads = dict(tree_flatten(grads))

    for i, (opt, group_keys) in enumerate(zip(optimizers, param_group_keys)):
        group_grads = {k: flat_grads[k] for k in group_keys if k in flat_grads}
        if not group_grads:
            continue
        group_params = {k: flat_model_params[k] for k in group_keys}

        if i == 0:  # AdamW
            updated_params = opt.update(group_params, group_grads)
            for key, param in updated_params.items():
                flat_model_params[key] = param
        else:  # Muon
            updates = opt.update(group_params, group_grads)
            for key, update in updates.items():
                flat_model_params[key] = flat_model_params[key] + update

    # Update model
    model.update(tree_unflatten(list(flat_model_params.items())))
    mx.eval(model.parameters())

    # Force memory cleanup
    del flat_model_params, flat_grads, grads
    mx.eval(mx.array([0]))

    t1 = time.time()
    if step % 10 == 0:
        logger.info(f"Step {step:03d} | loss: {float(loss):.4f} | dt: {(t1-t0)*1000:.2f}ms")

logger.info("Memory leak test complete! If this completed without crashing, memory management is OK.")
logger.info("Check Activity Monitor to see if memory usage stabilized or kept growing.")
