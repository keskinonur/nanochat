"""
Utilities for saving and loading model/optim/state checkpoints - MLX port.
"""
import os
import re
import glob
import json
import logging
import mlx.core as mx
import mlx.nn as nn

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    assert int(os.environ.get('RANK', 0)) == 0 # prevent footguns for now
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the model state (parameters) using MLX's save function
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.npz")
    mx.savez(model_path, **model_data)
    log0(f"Saved model file to: {model_path}")
    # Save the optimizer state (useful for SFT or any other fine-tuning)
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.npz")
        # MLX optimizer states may need special handling
        if isinstance(optimizer_data, dict):
            mx.savez(optimizer_path, **optimizer_data)
        else:
            # If it's a list, save each optimizer's state separately
            for i, opt_state in enumerate(optimizer_data):
                opt_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_{i}.npz")
                if isinstance(opt_state, dict):
                    mx.savez(opt_path, **opt_state)
        log0(f"Saved optimizer file to: {optimizer_path}")
    # Save the metadata dict as json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"Saved metadata file to: {meta_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    # Load the model state using MLX's load function
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.npz")
    model_data = dict(mx.load(model_path))
    # Load the optimizer state if requested
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.npz")
        if os.path.exists(optimizer_path):
            optimizer_data = dict(mx.load(optimizer_path))
        else:
            # Try loading multiple optimizer files
            optimizer_data = []
            i = 0
            while True:
                opt_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_{i}.npz")
                if not os.path.exists(opt_path):
                    break
                optimizer_data.append(dict(mx.load(opt_path)))
                i += 1
    # Load the metadata
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # Clean up any potential naming issues
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)

    # Create the model
    model = GPT(model_config)

    # Load the model state - MLX uses update() method
    model.update(model_data)

    # Note: In MLX, there's no explicit train/eval mode switching like PyTorch
    # We can add a flag if needed, but MLX handles this differently
    model._phase = phase

    # Load the Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    # attempt to guess the model tag: take the biggest model available
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.npz with the highest step
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.npz"))
    if not checkpoint_files:
        # Also try .pt for backwards compatibility with PyTorch checkpoints
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # guess the step by defaulting to the last step
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
