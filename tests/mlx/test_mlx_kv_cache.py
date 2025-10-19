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

print("\nKV cache test passed!")
