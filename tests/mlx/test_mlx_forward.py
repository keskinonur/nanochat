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

print("Testing forward pass with loss...")
# Forward pass with loss
loss = model(ids, targets=targets)
print(f"Loss: {loss}")
print(f"Loss shape: {loss.shape}")
print(f"Forward pass with loss successful!")

print("\nTesting forward pass for inference...")
# Forward pass without loss (inference)
logits = model(ids)
print(f"Logits shape: {logits.shape}")
print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
print(f"Forward pass for inference successful!")
