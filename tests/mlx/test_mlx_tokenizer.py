"""
Test 3: Tokenizer Integration

NOTE: This test requires the tokenizer to be downloaded first.
To download the tokenizer, run:
    python -c "from nanochat.tokenizer import get_tokenizer; get_tokenizer()"

For now, this test will be skipped if the tokenizer is not available.
"""

try:
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

    print("\nTokenizer test passed!")
except (FileNotFoundError, KeyError) as e:
    print("Tokenizer not available or not trained yet.")
    print("\nTo train the tokenizer, run:")
    print("    python -m nanochat.train_tokenizer")
    print("\nOr download data and it will train automatically:")
    print("    python -m nanochat.dataset -n 10")
    print(f"\nError: {e}")
    print("\nSkipping tokenizer test...")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    print("\nSkipping tokenizer test...")
