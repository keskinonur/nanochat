"""
Train base model on Apple Silicon with MLX. Run as:

python scripts/base_train_mlx.py

Note: MLX uses unified memory and runs on a single device (no distributed training).
"""

import os
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from mlx.utils import tree_flatten

print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Model architecture
depth = 20 # the depth of the Transformer model to train
max_seq_len = 2048 # max context length
# Training horizon
num_iterations = -1 # explicit number of steps (-1 = disable)
target_flops = -1.0 # calculate from target_flops (-1 = disable)
target_param_data_ratio = 20 # Chinchilla=20 (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for embedding parameters (AdamW)
unembedding_lr = 0.004 # learning rate for unembedding parameters (AdamW)
weight_decay = 0.0 # weight decay for embedding/unembedding (AdamW)
matrix_lr = 0.02 # learning rate for matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
# Evaluation
eval_every = 250 # eval validation bpb every N steps
eval_tokens = 20*524288 # tokens to evaluate on
core_metric_every = 2000 # eval CORE metric every N steps
core_metric_max_per_task = 500 # examples per task for CORE
sample_every = 2000 # sample from model every N steps
# Output
model_tag = "" # override model tag for checkpoint dir name
# CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0

# wandb logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb()  # For now, always use dummy (can add wandb later)

# Tokenizer - for now we skip this for basic testing
try:
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")
except:
    print0("Warning: Tokenizer not available, using default vocab_size=50304")
    tokenizer = None
    vocab_size = 50304

# Model kwargs derived from depth
num_layers = depth
model_dim = depth * 64 # aspect ratio 64
num_heads = max(1, (model_dim + 127) // 128) # head dim 128
num_kv_heads = num_heads # 1:1 MQA ratio
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Calculate gradient accumulation
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize Model
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim
)
model_config = GPTConfig(**model_config_kwargs)
model = GPT(model_config)
model.init_weights()

# Count parameters
num_params = sum(p.size for p in tree_flatten(model.parameters()))
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize Optimizers
# For now, use a simple approach - will refine later
adamw_params = []
muon_params = []

# Separate parameters into groups
for block in model.h.layers:
    muon_params.extend([
        block.attn.c_q.weight, block.attn.c_k.weight,
        block.attn.c_v.weight, block.attn.c_proj.weight,
        block.mlp.c_fc.weight, block.mlp.c_proj.weight
    ])

adamw_params.extend([model.wte.weight, model.lm_head.weight])

# Create optimizers
from nanochat.adamw import AdamW
from nanochat.muon import Muon

adamw_optimizer = AdamW(
    learning_rate=embedding_lr,
    betas=(0.8, 0.95),
    eps=1e-10,
    weight_decay=weight_decay
)

muon_optimizer = Muon(
    learning_rate=matrix_lr,
    momentum=0.95
)

print0("Optimizers initialized")

# -----------------------------------------------------------------------------
# Initialize DataLoaders
print0("Note: DataLoader not yet ported - using dummy data for now")
# For testing, we'll create dummy data
def create_dummy_batch():
    """Create dummy batch for testing"""
    x = mx.random.randint(0, vocab_size, (device_batch_size, max_seq_len))
    y = mx.random.randint(0, vocab_size, (device_batch_size, max_seq_len))
    return x, y

# -----------------------------------------------------------------------------
# Learning rate scheduler
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
print0("Starting training loop...")
min_val_loss = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # TODO: Add evaluation logic here
    # if last_step or step % eval_every == 0:
    #     ...

    # TODO: Add CORE metric evaluation
    # if last_step or (step > 0 and step % core_metric_every == 0):
    #     ...

    # TODO: Add sampling
    # if master_process and (last_step or (step > 0 and step % sample_every == 0)):
    #     ...

    # Save checkpoint at end
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", output_dirname)

        # Flatten parameters for saving
        flat_params = dict(tree_flatten(model.parameters()))

        save_checkpoint(
            checkpoint_dir,
            step,
            flat_params,
            None,  # Skip optimizer state for now
            {
                "step": step,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )
        print0(f"Checkpoint saved to {checkpoint_dir}")

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Training step
    t0 = time.time()

    # Get dummy data
    x, y = create_dummy_batch()

    # Define loss function for this batch
    def loss_fn(params):
        # Update model parameters
        model.update(params)
        # Forward pass
        loss = model(x, y)
        return loss

    # Compute loss and gradients
    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    # Accumulate gradients
    total_loss = 0
    accumulated_grads = None

    for micro_step in range(grad_accum_steps):
        x, y = create_dummy_batch()

        # Forward and backward
        loss_val, grads = loss_and_grad_fn(model.parameters())
        total_loss += loss_val

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            # Add gradients
            accumulated_grads = mx.tree_map(lambda a, b: a + b, accumulated_grads, grads)

        mx.eval(loss_val)

    # Average loss and gradients
    avg_loss = total_loss / grad_accum_steps
    accumulated_grads = mx.tree_map(lambda g: g / grad_accum_steps, accumulated_grads)

    # Gradient clipping
    if grad_clip > 0.0:
        grad_norm = mx.sqrt(sum(mx.sum(mx.square(g)) for g in tree_flatten(accumulated_grads)))
        if grad_norm > grad_clip:
            scale = grad_clip / grad_norm
            accumulated_grads = mx.tree_map(lambda g: g * scale, accumulated_grads)

    # Apply learning rate schedule
    lrm = get_lr_multiplier(step)

    # Update parameters (simplified - needs proper parameter group handling)
    # TODO: Separate AdamW and Muon parameter updates

    # For now, just apply simple SGD-style update
    current_params = model.parameters()
    updated_params = mx.tree_map(
        lambda p, g: p - (matrix_lr * lrm) * g,
        current_params,
        accumulated_grads
    )
    model.update(updated_params)

    t1 = time.time()
    dt = t1 - t0

    # -------------------------------------------------------------------------
    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * float(avg_loss)
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt

    # MLX on M2 Ultra - rough estimate of peak FLOPs
    # M2 Ultra has ~27 TFLOPS FP32, ~54 TFLOPS FP16
    promised_flops_per_sec = 54e12  # 54 TFLOPS for FP16
    mfu = 100 * flops_per_sec / promised_flops_per_sec

    if step > 10:
        total_training_time += dt

    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")

# Final stats
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation loss: {min_val_loss:.4f}")

# Cleanup
compute_cleanup()
print0("Training complete!")
