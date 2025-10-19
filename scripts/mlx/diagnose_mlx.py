"""
Diagnostic script to check MLX setup and memory limits.

Run as: python -m scripts.diagnose_mlx
"""

import sys
import psutil
import mlx.core as mx

print("=" * 60)
print("MLX DIAGNOSTIC")
print("=" * 60)

# System information
print("\n1. SYSTEM INFORMATION")
print("-" * 60)
print(f"Python version: {sys.version}")

# Memory information
mem = psutil.virtual_memory()
print(f"\nTotal RAM: {mem.total / (1024**3):.1f} GB")
print(f"Available RAM: {mem.available / (1024**3):.1f} GB")
print(f"Used RAM: {mem.used / (1024**3):.1f} GB")
print(f"RAM usage: {mem.percent}%")

# MLX information
print("\n2. MLX INFORMATION")
print("-" * 60)
print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")

# Test basic operations
print("\n3. BASIC MLX TESTS")
print("-" * 60)

try:
    # Test 1: Simple array creation
    print("✓ Creating small array... ", end="")
    a = mx.array([1, 2, 3])
    mx.eval(a)
    print("OK")

    # Test 2: Matrix multiplication
    print("✓ Matrix multiplication... ", end="")
    x = mx.random.normal((100, 100))
    y = mx.random.normal((100, 100))
    z = x @ y
    mx.eval(z)
    print("OK")

    # Test 3: Larger allocation
    print("✓ Larger array (10MB)... ", end="")
    large = mx.random.normal((1000, 1000))
    mx.eval(large)
    print("OK")

    # Test 4: Very large allocation
    print("✓ Very large array (100MB)... ", end="")
    very_large = mx.random.normal((3000, 3000))
    mx.eval(very_large)
    print("OK")

    # Test 5: Gradient computation
    print("✓ Gradient computation... ", end="")
    def f(x):
        return mx.sum(x ** 2)
    grad_fn = mx.grad(f)
    x = mx.array([1.0, 2.0, 3.0])
    g = grad_fn(x)
    mx.eval(g)
    print("OK")

    print("\n✅ All basic tests passed!")

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    print("\nThis suggests an issue with your MLX installation.")
    print("Try reinstalling MLX: pip install --upgrade mlx")

# Recommendations
print("\n4. RECOMMENDED SETTINGS FOR YOUR SYSTEM")
print("-" * 60)
available_gb = mem.available / (1024**3)

if available_gb > 100:
    print("You have plenty of RAM (>100GB available)")
    print("Recommended settings:")
    print("  --depth=20")
    print("  --device_batch_size=16")
    print("  --max_seq_len=1024")
elif available_gb > 50:
    print("You have good RAM (50-100GB available)")
    print("Recommended settings:")
    print("  --depth=12")
    print("  --device_batch_size=8")
    print("  --max_seq_len=512")
elif available_gb > 20:
    print("You have moderate RAM (20-50GB available)")
    print("Recommended settings:")
    print("  --depth=8")
    print("  --device_batch_size=4")
    print("  --max_seq_len=256")
else:
    print("You have limited RAM (<20GB available)")
    print("Recommended settings:")
    print("  --depth=4")
    print("  --device_batch_size=2")
    print("  --max_seq_len=128")

print("\n5. SAFE TEST COMMAND")
print("-" * 60)
print("Try this ultra-safe test first:")
print("  python -m scripts.safe_test_mlx")
print("\nIf that works, gradually increase the parameters.")

print("\n6. COMMON CRASH CAUSES")
print("-" * 60)
print("• Batch size too large for memory")
print("• Gradient accumulation creating too many intermediate arrays")
print("• Not calling mx.eval() to free memory")
print("• Memory leaks in training loop")
print("• System swap thrashing (memory pressure)")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
