import os
import mlx.core as mx
from nanochat.gpt import GPT, GPTConfig
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint

# Create model
config = GPTConfig(
    sequence_len=256,
    vocab_size=1000,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=256
)

model = GPT(config)
model.init_weights()

# Get model parameters
params = model.parameters()

# Flatten parameters for saving (mx.savez expects flat dict)
from mlx.utils import tree_flatten
flat_params = dict(tree_flatten(params))

# Create test checkpoint directory
test_dir = "/tmp/nanochat_test_checkpoint"
os.makedirs(test_dir, exist_ok=True)

# Save checkpoint
meta_data = {
    "step": 100,
    "model_config": config.__dict__
}

save_checkpoint(
    test_dir,
    step=100,
    model_data=flat_params,
    optimizer_data=None,
    meta_data=meta_data
)

print("Checkpoint saved!")

# Load checkpoint
loaded_data, _, loaded_meta = load_checkpoint(
    test_dir,
    step=100,
    device=mx.default_device(),
    load_optimizer=False
)

print(f"Checkpoint loaded!")
print(f"Loaded meta: {loaded_meta}")
print(f"Model config from checkpoint: {loaded_meta['model_config']}")

# Clean up
import shutil
shutil.rmtree(test_dir)
print("Test cleanup complete!")
print("\nCheckpoint test passed!")
