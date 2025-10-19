"""
Finetune a midtrained model to be a chat model - MLX port for Apple Silicon.
Run with:

python -m scripts.mlx.chat_sft_mlx

Optional arguments:
python -m scripts.mlx.chat_sft_mlx -- --device_batch_size=2 --num_epochs=1
"""

import os
import copy
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten, tree_map

from nanochat.common import get_base_dir, setup_default_logging
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from scripts.mlx.chat_eval_mlx import run_chat_eval

from tasks.common import TaskMixture, TaskSequence
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.smoltalk import SmolTalk

# Set up logging
setup_default_logging()
import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# input model options
source = "mid" # base|mid , which checkpoint to load the model from (base model or midtrained model)
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
# compute/precision
device_batch_size = 1 # reduced to 1 to avoid memory crashes
# optimization
num_epochs = 1
max_iterations = -1 # override number of iterations (-1 = use num_epochs * num_iterations)
target_examples_per_step = 1  # no gradient accumulation to minimize memory usage
grad_clip = 1.0
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging there of
eval_every = 100
eval_steps = 10  # reduced from 100 to 10 to avoid memory issues
eval_metrics_every = -1  # disabled expensive evaluations (MMLU, ARC, etc.) to prevent crashes
save_checkpoint_every = 50  # save checkpoint every 50 steps to avoid losing progress on crashes
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# MLX uses unified memory, no need for distributed training setup
device = mx.gpu
master_process = True

logger.info(f"MLX device: {device}")
logger.info(f"Running on Apple Silicon with unified memory")

# Load the model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
engine = Engine(model, tokenizer) # will be used for inline model evaluation only

# -----------------------------------------------------------------------------
# Task data mixture we'll train on

train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    GSM8K(subset="main", split="train"), # 8K rows
    SmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
]) # 2.3K + 1.1K + 8K + 10K = 21.4K rows
val_ds = SmolTalk(split="test") # general conversations, 24K rows (though we don't actually use all of it)

logger.info(f"Training dataset size: {len(train_ds)}")
logger.info(f"Validation dataset size: {len(val_ds)}")

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        input_rows = []
        target_rows = []
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_array = mx.array(ids, dtype=mx.int32)

            # Build input row (tokens 0..n-2), pad with pad_token_id
            input_row = ids_array[:-1]
            pad_inp = ncols - (n - 1)
            if pad_inp > 0:
                pad_vals = mx.full((pad_inp,), pad_token_id, dtype=mx.int32)
                input_row = mx.concatenate([input_row, pad_vals], axis=0)
            input_rows.append(mx.expand_dims(input_row, axis=0))

            # Build target row (tokens 1..n-1) with mask applied and padded by -1
            row_targets = ids_array[1:]
            mask_array = mx.array(mask[1:], dtype=mx.int32)
            row_targets = mx.where(mask_array == 0, -1, row_targets)
            pad_tgt = ncols - (n - 1)
            if pad_tgt > 0:
                pad_vals_t = mx.full((pad_tgt,), -1, dtype=mx.int32)
                row_targets = mx.concatenate([row_targets, pad_vals_t], axis=0)
            target_rows.append(mx.expand_dims(row_targets, axis=0))

        inputs = mx.concatenate(input_rows, axis=0)
        targets = mx.concatenate(target_rows, axis=0)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(len(dataset)):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size
logger.info(f"Target examples per step: {target_examples_per_step}")
logger.info(f"Device batch size: {device_batch_size}")
logger.info(f"Examples per step: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
logger.info(f"=> Setting grad accum steps: {grad_accum_steps}")

num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if max_iterations >= 0 and num_iterations > max_iterations:
    logger.info(f"Number of iterations is too high: {num_iterations}, capping to {max_iterations}")
    num_iterations = max_iterations
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

logger.info(f"Total training iterations: {num_iterations}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers, param_groups = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

# Store initial learning rates for scheduler
initial_lrs = [opt.learning_rate for opt in optimizers]

# Build parameter key sets for each optimizer group (do this once)
flat_model_params = dict(tree_flatten(model.parameters()))
param_group_keys = []
for param_group in param_groups:
    group_keys = set()
    for param in param_group:
        # Find this parameter in the flat model params by object identity
        for key, model_param in flat_model_params.items():
            if id(model_param) == id(param):
                group_keys.add(key)
                break
    param_group_keys.append(group_keys)

logger.info("Optimizers initialized")
logger.info(f"Parameter groups: {[len(keys) for keys in param_group_keys]}")

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Loss and value_and_grad function
def loss_fn(model, inputs, targets):
    # In MLX GPT, passing targets triggers loss computation
    return model(inputs, targets=targets)

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Go!
logger.info("Starting SFT training...")
step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1
    t0 = time.time()

    # evaluate the validation loss
    if last_step or step % eval_every == 0:
        logger.info(f"Step {step:05d} | Running validation...")
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            loss = loss_fn(model, val_inputs, val_targets)
            losses.append(loss)
        val_loss = mx.mean(mx.array(losses))
        val_loss_item = float(val_loss)
        logger.info(f"Step {step:05d} | Validation loss: {val_loss_item:.6f}")

    # evaluate metrics
    if eval_metrics_every > 0 and (last_step or (step > 0 and step % eval_metrics_every == 0)):
        logger.info(f"Step {step:05d} | Running evaluation metrics...")
        metrics = {}
        # note that because these are inside no_grad equivalent, we can usually afford to at least ~2X the batch size
        metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
        metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
        metrics["gsm8k_acc"] = run_chat_eval("GSM8K", model, tokenizer, engine, max_problems=64)
        metrics["humaneval_acc"] = run_chat_eval("HumanEval", model, tokenizer, engine, max_problems=64)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        logger.info(f"Step {step:05d} | {metrics_str}")
    else:
        metrics = {}  # empty metrics if we're not evaluating

    if last_step:
        break

    # evaluate the gradient
    num_tokens = 0
    total_loss = 0.0
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        loss, grads = loss_and_grad_fn(model, train_inputs, train_targets)
        total_loss += float(loss)

        # Scale gradients by grad_accum_steps
        grads = tree_map(lambda x: x / grad_accum_steps, grads)

        # Accumulate gradients
        if micro_step == 0:
            accumulated_grads = grads
        else:
            accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)

        num_tokens += int(mx.sum(train_targets >= 0))

    train_loss = total_loss / grad_accum_steps

    # Gradient clipping (global norm)
    if grad_clip and grad_clip > 0:
        flat_grads_list = [g for _, g in tree_flatten(accumulated_grads)]
        global_norm = mx.sqrt(sum(mx.sum(mx.square(g)) for g in flat_grads_list))
        # scale = min(1, grad_clip / (global_norm + 1e-6))
        scale = grad_clip / (global_norm + 1e-6)
        # Only scale if norm exceeds clip threshold
        scale = mx.minimum(mx.array(1.0), scale)
        accumulated_grads = tree_map(lambda g: g * scale, accumulated_grads)

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for i, opt in enumerate(optimizers):
        opt.learning_rate = initial_lrs[i] * lrm

    # Apply optimizer updates to each parameter group
    flat_model_params = dict(tree_flatten(model.parameters()))
    flat_grads = dict(tree_flatten(accumulated_grads))

    # Apply each optimizer to its parameter group
    for i, (opt, group_keys) in enumerate(zip(optimizers, param_group_keys)):
        # Extract gradients for this group
        group_grads = {k: flat_grads[k] for k in group_keys if k in flat_grads}

        if not group_grads:
            continue

        # Extract parameters for this group
        group_params = {k: flat_model_params[k] for k in group_keys}

        # Compute updates for this group
        if i == 0:  # AdamW - returns updated parameters directly
            updated_params = opt.update(group_params, group_grads)
            # Replace parameters with updated versions
            for key, param in updated_params.items():
                flat_model_params[key] = param
        else:  # Muon - returns parameter updates (deltas)
            updates = opt.update(group_params, group_grads)
            # Apply updates to model parameters
            for key, update in updates.items():
                flat_model_params[key] = flat_model_params[key] + update

    # Update model with new parameters
    model.update(tree_unflatten(list(flat_model_params.items())))
    mx.eval(model.parameters())  # Ensure updates are applied

    # Force memory cleanup to prevent crashes
    del flat_model_params, flat_grads, accumulated_grads, grads
    mx.eval(mx.array([0]))  # Force graph evaluation to free memory

    # logging
    t1 = time.time()
    dt = (t1 - t0) * 1000  # ms
    logger.info(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss:.6f} | lrm: {lrm:.6f} | num_tokens: {num_tokens:,} | dt: {dt:.2f}ms")
    step += 1

    # Save intermediate checkpoints
    if master_process and step % save_checkpoint_every == 0:
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag_save = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag_save)
        model_config_kwargs = model.config.__dict__
        model_data = dict(tree_flatten(model.parameters()))
        save_checkpoint(
            checkpoint_dir,
            step,
            model_data,
            None,
            {
                "step": step,
                "model_config": model_config_kwargs,
            }
        )
        logger.info(f"💾 Saved intermediate checkpoint at step {step}")

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag_save = f"d{depth}" # base the model tag on the depth of the base model
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag_save)
    model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer

    # Convert model parameters to dict for saving
    model_data = dict(tree_flatten(model.parameters()))

    save_checkpoint(
        checkpoint_dir,
        step,
        model_data,
        None, # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_loss": val_loss_item,
            **metrics,
            "model_config": model_config_kwargs,
        }
    )
    logger.info(f"✅ Saved model checkpoint to {checkpoint_dir}")

logger.info("SFT training complete!")
