# MLX Port Testing Guide

This guide will help you test the MLX port on your M2 Ultra with 192GB unified RAM.

## Prerequisites

1. **Install dependencies**:
```bash
# Make sure you're on the mlx-port branch
git branch

# Install dependencies with uv
uv sync
source .venv/bin/activate
```

2. **Build the tokenizer** (unchanged from original):
```bash
# Install Rust/Cargo if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build rustbpe tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

## Test 1: Model Initialization

Test that the GPT model can be created and initialized:

```python
import mlx.core as mx
from nanochat.gpt import GPT, GPTConfig

# Create a small test model
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

print(f"Model created successfully!")
print(f"Parameters: {sum(p.size for p in model.parameters())}")
```

## Test 2: Forward Pass

Test that the model can perform a forward pass:

```python
import mlx.core as mx
from nanochat.gpt import GPT, GPTConfig

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

# Create dummy input
batch_size = 2
seq_len = 10
ids = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass with loss
loss = model(ids, targets=targets)
print(f"Loss: {loss}")
print(f"Loss shape: {loss.shape}")
print(f"Forward pass successful!")

# Forward pass without loss (inference)
logits = model(ids)
print(f"Logits shape: {logits.shape}")
print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
```

## Test 3: Tokenizer Integration

Test that the tokenizer works with MLX:

```python
from nanochat.tokenizer import get_tokenizer
import mlx.core as mx

# Get tokenizer
tokenizer = get_tokenizer()
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Test encoding
text = "Hello, world! This is a test."
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens[:20]}...")  # First 20 tokens
print(f"Number of tokens: {len(tokens)}")

# Test decoding
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Test with BOS token
bos_id = tokenizer.get_bos_token_id()
tokens_with_bos = tokenizer.encode(text, prepend=bos_id)
print(f"First token (should be BOS): {tokens_with_bos[0]}")
print(f"BOS token ID: {bos_id}")
```

## Test 4: Inference Engine

Test the inference engine with generation:

```python
import mlx.core as mx
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer

# Create small model
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

# Get tokenizer
tokenizer = get_tokenizer()

# Create engine
engine = Engine(model, tokenizer)

# Generate some tokens (using random vocab IDs since model is untrained)
prompt_tokens = [1, 2, 3, 4, 5]  # Dummy tokens
print("Generating tokens...")

generated = []
for token_column, token_masks in engine.generate(
    prompt_tokens,
    num_samples=1,
    max_tokens=10,
    temperature=1.0,
    seed=42
):
    token = token_column[0]
    generated.append(token)
    print(f"Generated token: {token}")

print(f"\nGenerated {len(generated)} tokens")
print(f"All tokens: {generated}")
```

## Test 5: KV Cache

Test that the KV cache works correctly:

```python
import mlx.core as mx
from nanochat.engine import KVCache

# Create a small KV cache
kv_cache = KVCache(
    batch_size=2,
    num_heads=4,
    seq_len=100,
    head_dim=64,
    num_layers=4
)

print(f"KV cache shape: {kv_cache.kv_shape}")
print(f"Initial position: {kv_cache.get_pos()}")

# Simulate inserting some keys/values
B, H, T, D = 2, 4, 5, 64
k = mx.random.normal((B, H, T, D))
v = mx.random.normal((B, H, T, D))

# Insert into layer 0
k_cached, v_cached = kv_cache.insert_kv(0, k, v)
print(f"After insert - position: {kv_cache.get_pos()}")
print(f"Cached K shape: {k_cached.shape}")
print(f"Cached V shape: {v_cached.shape}")

print("KV cache test passed!")
```

## Test 6: Checkpoint Save/Load

Test checkpoint saving and loading:

```python
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
    model_data=params,
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
print(f"Number of parameters: {len(loaded_data)}")

# Clean up
import shutil
shutil.rmtree(test_dir)
print("Test cleanup complete!")
```

## Test 7: Optimizers

Test the Muon and AdamW optimizers:

```python
import mlx.core as mx
import mlx.nn as nn
from nanochat.muon import Muon
from nanochat.adamw import AdamW

# Create simple parameters
params = {
    "weight": mx.random.normal((100, 50)),
    "bias": mx.random.normal((50,))
}

# Create gradients (simulated)
grads = {
    "weight": mx.random.normal((100, 50)) * 0.01,
    "bias": mx.random.normal((50,)) * 0.01
}

# Test Muon
print("Testing Muon optimizer...")
muon = Muon(learning_rate=0.01, momentum=0.9)
updates = muon.update(params, grads)
print(f"Muon updates computed: {len(updates)} parameters")

# Test AdamW
print("\nTesting AdamW optimizer...")
adamw = AdamW(learning_rate=0.001)
adamw_updates = adamw.update(params, grads)
print(f"AdamW updates computed")

print("\nOptimizer tests passed!")
```

## Expected Results

### Test 1 (Model Init)
- Should print model creation success
- Parameters count should be > 0

### Test 2 (Forward Pass)
- Loss should be a scalar value
- Logits shape should match (batch_size, seq_len, vocab_size)

### Test 3 (Tokenizer)
- Should encode and decode text correctly
- Vocab size should be ~65536

### Test 4 (Engine)
- Should generate tokens without errors
- Output should be a list of token IDs

### Test 5 (KV Cache)
- Should insert and retrieve K/V without errors
- Position should increment correctly

### Test 6 (Checkpoints)
- Should save and load without errors
- Loaded data should match saved data structure

### Test 7 (Optimizers)
- Should compute updates without errors
- Updates dict should have same keys as params

## Troubleshooting

### Import Errors
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv sync
```

### MLX Not Found
```bash
# Install MLX explicitly
pip install mlx>=0.21.0
```

### Tokenizer Errors
```bash
# Rebuild tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Memory Issues
- The M2 Ultra should handle these tests easily with 192GB
- If you see memory errors, reduce batch_size or model size in tests

## Next Steps

After all tests pass:
1. Port and test training scripts
2. Run a short training loop
3. Test full inference pipeline
4. Create MLX-specific speedrun script

## Reporting Issues

If you encounter issues, please note:
- Which test failed
- Error message and stack trace
- MLX version (`python -c "import mlx; print(mlx.__version__)"`)
- macOS version
- Available memory at time of failure
