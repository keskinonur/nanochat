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

# Count parameters properly in MLX
def count_params(params):
    count = 0
    if isinstance(params, dict):
        for v in params.values():
            count += count_params(v)
    elif isinstance(params, list):
        for v in params:
            count += count_params(v)
    elif hasattr(params, 'size'):
        count += params.size
    return count

total_params = count_params(model.parameters())
print(f"Parameters: {total_params:,}")