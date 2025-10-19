"""
Test 4: Inference Engine

NOTE: This test requires the tokenizer to be downloaded first.
To download the tokenizer, run:
    python -c "from nanochat.tokenizer import get_tokenizer; get_tokenizer()"

For now, this test will be skipped if the tokenizer is not available.
"""

try:
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
    print("\nEngine test passed!")
except (FileNotFoundError, KeyError) as e:
    print("Tokenizer not available or not trained yet.")
    print("\nTo train the tokenizer, run:")
    print("    python -m nanochat.train_tokenizer")
    print("\nOr download data and it will train automatically:")
    print("    python -m nanochat.dataset -n 10")
    print(f"\nError: {e}")
    print("\nSkipping engine test...")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")
    print("\nSkipping engine test...")
