"""
Midtrain the model on Apple Silicon with MLX.
Same as pretraining but simpler - loads pretrained base model and finetunes on conversation data.

Run as:
python -m scripts.mid_train_mlx

Or with options:
python -m scripts.mid_train_mlx --device_batch_size=16 --model_tag=d20 --step=10000
"""

from collections import deque
import os
import time
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.loss_eval_mlx import evaluate_bpb

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name
model_tag = None # model tag to load (e.g., "d20")
step = None # step to load (None = last step)
max_seq_len = 2048
device_batch_size = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0 # initial LR fraction
weight_decay = 0.0
final_lr_frac = 0.0 # final LR fraction
eval_every = 150
eval_tokens = 20*524288
total_batch_size = 524288
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0

# wandb logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb()

# Load the model and tokenizer
print0("Loading base model...")
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
    print0(f"WARNING: base model used device_batch_size {pretrain_batch_size}, you're using {device_batch_size}")

depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# TODO: Load token_bytes for evaluation
# token_bytes = get_token_bytes(device=device)

# Initialize optimizers
from nanochat.adamw import AdamW
from nanochat.muon import Muon

adamw_optimizer = AdamW(
    learning_rate=embedding_lr * init_lr_frac,
    betas=(0.8, 0.95),
    eps=1e-10,
    weight_decay=weight_decay
)

muon_optimizer = Muon(
    learning_rate=matrix_lr * init_lr_frac,
    momentum=0.95
)

print0("Optimizers initialized")

# Midtraining data mixture
base_dir = get_base_dir()
train_dataset = TaskMixture([
    SmolTalk(split="train"), # 460K rows
    MMLU(subset="auxiliary_train", split="train"), # 100K rows
    GSM8K(subset="main", split="train"), # 8K rows
]) # total: ~568K rows

val_dataset = TaskMixture([
    SmolTalk(split="test"), # 24K rows
    MMLU(subset="all", split="test", stop=5200), # 5.2K rows
    GSM8K(subset="main", split="test", stop=420), # 420 rows
]) # total: ~30K rows

# DataLoader - MLX version
last_step = False
approx_progress = 0.0

def mid_data_generator(split):
    """Generate batches from conversation datasets"""
    global last_step, approx_progress
    assert split in {"train", "val"}
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0

    needed_tokens = device_batch_size * max_seq_len + 1
    token_buffer = deque()
    import numpy as np
    scratch = np.empty(needed_tokens, dtype=np.int64)
    cursor = ddp_rank

    while True:
        # Accumulate tokens
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size
                if split == "train":
                    last_step = True

        # Build batch
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()

        inputs_np = scratch[:-1].astype(np.int32)
        targets_np = scratch[1:].astype(np.int32)

        # Convert to MLX arrays
        inputs = mx.array(inputs_np.reshape(device_batch_size, max_seq_len))
        targets = mx.array(targets_np.reshape(device_batch_size, max_seq_len))

        if split == "train":
            approx_progress = cursor / dataset_size

        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
progress = 0

# Learning rate scheduler
def get_lr_multiplier(progress):
    return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# Training loop
print0("Starting midtraining...")
x, y = next(train_loader)
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0

while True:
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Evaluation
    if last_step or step % eval_every == 0:
        print0(f"Step {step:05d} | Validation evaluation...")
        # TODO: Implement full evaluation with token_bytes
        # val_loader = build_val_loader()
        # eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        # val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        # print0(f"Validation bpb: {val_bpb:.4f}")
        print0(f"Validation evaluation skipped (not yet implemented)")

    # Save checkpoint
    if master_process and last_step:
        output_dirname = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)

        flat_params = dict(tree_flatten(model.parameters()))
        save_checkpoint(
            checkpoint_dir,
            step,
            flat_params,
            None,
            {
                "step": step,
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                },
                "user_config": user_config,
            }
        )
        print0(f"Checkpoint saved to {checkpoint_dir}")

    if last_step:
        break

    # -------------------------------------------------------------------------
    # Training step
    t0 = time.time()

    def loss_fn(params):
        model.update(params)
        return model(x, y)

    loss_and_grad_fn = mx.value_and_grad(loss_fn)

    # Accumulate gradients
    total_loss = 0
    accumulated_grads = None

    for micro_step in range(grad_accum_steps):
        loss_val, grads = loss_and_grad_fn(model.parameters())
        total_loss += loss_val

        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)

        mx.eval(loss_val)
        x, y = next(train_loader)
        progress = max(progress, approx_progress)

    # Average
    avg_loss = total_loss / grad_accum_steps
    accumulated_grads = tree_map(lambda g: g / grad_accum_steps, accumulated_grads)

    # Update parameters
    lrm = get_lr_multiplier(progress)
    current_params = model.parameters()
    updated_params = tree_map(
        lambda p, g: p - (matrix_lr * init_lr_frac * lrm) * g,
        current_params,
        accumulated_grads
    )
    model.update(updated_params)

    t1 = time.time()
    dt = t1 - t0
    step += 1

    # -------------------------------------------------------------------------
    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * float(avg_loss)
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec = 54e12  # M2 Ultra FP16
    mfu = 100 * flops_per_sec / promised_flops_per_sec

    if step > 10:
        total_training_time += dt

    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}% | total time: {total_training_time/60:.2f}m")

# Final stats
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Cleanup
compute_cleanup()
print0("Midtraining complete!")
