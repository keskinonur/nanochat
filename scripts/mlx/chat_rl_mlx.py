"""
Reinforcement learning on GSM8K via "GRPO" for Apple Silicon with MLX.

Simplified REINFORCE-style RL:
1) No trust region / KL regularization
2) On-policy, no PPO ratio+clip
3) Token-level GAPO-style normalization
4) Advantage = (r - mu) instead of z-score

Run as:
python -m scripts.chat_rl_mlx

Note: MLX uses unified memory and runs on a single device (no distributed training).
"""

import os
import itertools
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# RL hyperparameters
run = "dummy" # wandb run name
source = "sft" # mid|sft
device_batch_size = 4 # conservative for MLX (no forward pass will go above this)
examples_per_step = 8 # reduced for single device
num_samples = 8 # number of samples per example (reduced from 16)
max_new_tokens = 256
temperature = 1.0
top_k = 50
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # how many epochs of gsm8k to train on
save_every = 60 # every how many steps to save the model
eval_every = 60 # every how many steps to evaluate the model for val pass@k
eval_examples = 200 # reduced for single device
# CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Init compute
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = True  # Always master in single-device MLX
# MLX doesn't need autocast (automatic mixed precision)

# wandb logging init
wandb_run = DummyWandb()  # For now, always use dummy

# Init model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="eval")
engine = Engine(model, tokenizer)

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Calculated number of steps: {num_steps}")

def get_batch():
    """
    Generator that yields training batches.
    Each batch contains multiple samples for a single example.
    """
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    for example_idx in itertools.cycle(range(len(train_task))):
        # Get the conversation
        conversation = train_task[example_idx]

        # Tokenize, priming for completion
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation
        # Split into smaller batches to avoid OOM
        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size

        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            # MLX doesn't need autocast
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=device_batch_size,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate rewards
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad sequences to same length
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [
            seq + [assistant_end] * (max_length - len(seq))
            for seq in generated_token_sequences
        ]
        padded_masks = [
            mask + [0] * (max_length - len(mask))
            for mask in masks
        ]

        # Convert to MLX arrays
        ids = mx.array(padded_generated_token_sequences, dtype=mx.int32)
        mask_ids = mx.array(padded_masks, dtype=mx.int32)

        # Generate autoregressive inputs and targets
        inputs = ids[:, :-1]
        targets = ids[:, 1:].copy()  # copy to avoid in-place modification

        # Mask out padding and prompt tokens
        mask_bool = mask_ids[:, 1:] == 0
        targets = mx.where(mask_bool, -1, targets)

        rewards = mx.array(rewards, dtype=mx.float32)

        # Calculate advantages (subtract mean)
        mu = mx.mean(rewards)
        advantages = rewards - mu

        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k

def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and yields records of evaluation outcomes.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)

    for idx in range(max_examples):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate k samples
        assert num_samples <= device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})

        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init optimizers (simplified for MLX)
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

# Learning rate scheduler
def get_lr_multiplier(it):
    return 1.0 - it / num_steps

print0(f"Total sequences per step: {examples_per_step * num_samples}")
print0(f"Examples per step: {examples_per_step}")

# Kick off training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate the model periodically
    if step % eval_every == 0:
        model.eval()
        passk = mx.zeros(device_batch_size)

        records_iter = run_gsm8k_eval(
            val_task, tokenizer, engine,
            num_samples=device_batch_size,
            max_examples=eval_examples,
            temperature=1.0
        )
        records = list(records_iter)

        for k in range(1, device_batch_size + 1):
            passk_k = sum(
                any(o["is_correct"] for o in r["outcomes"][:k])
                for r in records
            )
            passk = passk.at[k - 1].set(passk_k / len(records))

        print_passk = [
            f"Pass@{k}: {float(passk[k - 1]):.4f}"
            for k in range(1, device_batch_size + 1)
        ]
        print0(f"Step {step} | {', '.join(print_passk)}")

    # Forward/Backward on rollouts
    rewards_list = []
    sequence_lengths = []

    for example_step in range(examples_per_step):
        # Get one batch
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)

        # Prepare model for training
        model.train()

        # Split into smaller batches if needed
        batch_size = inputs_all.shape[0]
        assert batch_size % device_batch_size == 0
        num_passes = batch_size // device_batch_size

        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]

            # Define loss function for this batch
            def loss_fn(params):
                model.update(params)
                # Calculate negative log probabilities (loss with no reduction)
                loss_per_token = model(inputs, targets)

                # For RL, we need per-token losses, not averaged
                # Reshape to (B, T)
                B, T = inputs.shape
                logits = model(inputs)  # Get logits

                # Manual cross-entropy per token
                V = logits.shape[-1]
                logits_flat = mx.reshape(logits, (-1, V))
                targets_flat = mx.reshape(targets, (-1,))

                # Log softmax (manual implementation)
                logits_max = mx.max(logits_flat, axis=-1, keepdims=True)
                logits_shifted = logits_flat - logits_max
                exp_logits = mx.exp(logits_shifted)
                sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
                log_probs_flat = logits_shifted - mx.log(sum_exp)

                # Gather log probs for target tokens
                # Create one-hot for targets
                targets_mask = (targets_flat != -1).astype(mx.float32)
                valid_targets = mx.where(targets_flat == -1, 0, targets_flat)

                # Get log prob for each target
                indices = mx.arange(logits_flat.shape[0])
                target_log_probs = log_probs_flat[indices, valid_targets]
                target_log_probs = mx.reshape(target_log_probs, (B, T))

                # Calculate PG objective
                pg_obj = mx.sum(target_log_probs * mx.expand_dims(advantages, -1))

                # Normalize
                num_valid = mx.sum(targets_mask).astype(mx.float32)
                num_valid = mx.maximum(num_valid, 1.0)
                pg_obj = pg_obj / (num_valid * num_passes * examples_per_step)

                # Loss is negative of objective (we minimize)
                return -pg_obj

            # Compute gradients
            loss_val, grads = mx.value_and_grad(loss_fn)(model.parameters())

            # Apply updates (simplified - TODO: separate AdamW and Muon for different params)
            lrm = get_lr_multiplier(step)
            current_params = model.parameters()
            updated_params = tree_map(
                lambda p, g: p - (matrix_lr * init_lr_frac * lrm) * g,
                current_params,
                grads
            )
            model.update(updated_params)

            # Force evaluation
            mx.eval(loss_val)
            mx.eval(updated_params)

            print0(f"Step {step}/{num_steps} | Example {example_step} | Pass {pass_idx} | loss: {float(loss_val):.6f} | Avg reward: {float(mx.mean(rewards)):.4f}")

        # For logging
        rewards_list.append(float(mx.mean(rewards_all)))
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # Logging
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    print0(f"Step {step}/{num_steps} | Avg reward: {mean_reward:.4f} | Avg seq length: {mean_sequence_length:.2f}")

    # Save checkpoint periodically
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__

        # Flatten parameters for saving
        flat_params = dict(tree_flatten(model.parameters()))

        save_checkpoint(
            checkpoint_dir,
            step,
            flat_params,
            None,  # Skip optimizer state
            {"model_config": model_config_kwargs}
        )
        print0(f"✅ Saved checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[user_config])

compute_cleanup()
print0("RL training complete!")
