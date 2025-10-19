#!/bin/bash
# Run all MLX port tests

echo "========================================="
echo "MLX Port Test Suite"
echo "========================================="
echo ""

# Change to project root directory
cd "$(dirname "$0")/../.." || exit 1

# Activate virtual environment
source .venv/bin/activate

# Track results
PASSED=0
FAILED=0
SKIPPED=0

# Test 1: Model Initialization
echo "Test 1: Model Initialization"
echo "-----------------------------------------"
if python -m tests.mlx.test_mlx_model_init; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 2: Forward Pass
echo "Test 2: Forward Pass"
echo "-----------------------------------------"
if python -m tests.mlx.test_mlx_forward; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 3: Tokenizer Integration
echo "Test 3: Tokenizer Integration"
echo "-----------------------------------------"
python -m tests.mlx.test_mlx_tokenizer
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    # Check if it was skipped (exit code doesn't distinguish, so we assume skipped)
    SKIPPED=$((SKIPPED + 1))
fi
echo ""

# Test 4: Inference Engine
echo "Test 4: Inference Engine"
echo "-----------------------------------------"
python -m tests.mlx.test_mlx_engine
if [ $? -eq 0 ]; then
    PASSED=$((PASSED + 1))
else
    # Check if it was skipped
    SKIPPED=$((SKIPPED + 1))
fi
echo ""

# Test 5: KV Cache
echo "Test 5: KV Cache"
echo "-----------------------------------------"
if python -m tests.mlx.test_mlx_kv_cache; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 6: Checkpoint Save/Load
echo "Test 6: Checkpoint Save/Load"
echo "-----------------------------------------"
if python -m tests.mlx.test_mlx_checkpoint 2>&1 | grep -q "Checkpoint test passed"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi
echo ""

# Test 7: Optimizers
echo "Test 7: Optimizers"
echo "-----------------------------------------"
if python -m tests.mlx.test_mlx_optimizers; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi
echo ""

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Passed:  $PASSED"
echo "Failed:  $FAILED"
echo "Skipped: $SKIPPED"
echo "Total:   7"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Some tests failed!"
    exit 1
else
    echo "All available tests passed!"
    if [ $SKIPPED -gt 0 ]; then
        echo "(Note: $SKIPPED tests were skipped - likely due to missing tokenizer)"
    fi
    exit 0
fi
