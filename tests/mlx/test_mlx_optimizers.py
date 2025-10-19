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
print(f"Update keys: {list(updates.keys())}")

# Test AdamW
print("\nTesting AdamW optimizer...")
adamw = AdamW(learning_rate=0.001)
adamw_updates = adamw.update(params, grads)
print(f"AdamW updates computed: {len(adamw_updates)} parameters")
print(f"Update keys: {list(adamw_updates.keys())}")

print("\nOptimizer tests passed!")
