"""
Interactive chat CLI for MLX-based models on Apple Silicon.

Run as:
python -m scripts.chat_cli_mlx -i sft
python -m scripts.chat_cli_mlx -i mid -p "What is the capital of France?"
"""
import argparse
import mlx.core as mx
from nanochat.common import compute_init
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source: sft|mid|rl|base")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Single prompt mode')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling')
args = parser.parse_args()

# Init model and tokenizer
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
print(f"Loading {args.source} model...")
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
print(f"Model loaded: {meta.get('model_config', {})}")

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

# Create Engine
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode (MLX)")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("Type 'clear' to start a new conversation")
print("-" * 50)

conversation_tokens = [bos]

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    # Handle commands
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Add user message
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Generate assistant response
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    response_tokens = []
    print("\nAssistant: ", end="", flush=True)

    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = int(token_column[0])  # num_samples=1, convert to Python int

        # Validate token is in vocab range
        if token < 0 or token >= tokenizer.get_vocab_size():
            print(f"\n[Warning: Invalid token {token}, skipping]", flush=True)
            continue

        response_tokens.append(token)
        try:
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
        except Exception as e:
            print(f"\n[Error decoding token {token}: {e}]", flush=True)
            continue

    print()

    # Ensure assistant_end token
    if response_tokens:
        if response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)
    else:
        # No valid tokens generated
        print("[Error: No valid tokens generated]")
        conversation_tokens.append(assistant_end)

    # Single prompt mode
    if args.prompt:
        break
