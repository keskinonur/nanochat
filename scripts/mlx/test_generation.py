"""
Debug script to test model generation
"""
import mlx.core as mx
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Load model
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
model, tokenizer, meta = load_model("mid", device, phase="eval")
engine = Engine(model, tokenizer)

# Build a simple conversation
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")

# Create conversation tokens
conversation_tokens = [bos]
conversation_tokens.append(user_start)
conversation_tokens.extend(tokenizer.encode("hello"))
conversation_tokens.append(user_end)
conversation_tokens.append(assistant_start)

print(f"Conversation tokens: {conversation_tokens}")
print(f"Decoded: {tokenizer.decode(conversation_tokens)}")
print(f"\nGenerating (max 20 tokens, temp=0.6, top_k=50)...\n")

# Generate
count = 0
for token_column, token_masks in engine.generate(
    conversation_tokens,
    num_samples=1,
    max_tokens=20,
    temperature=0.6,
    top_k=50
):
    token = int(token_column[0])
    count += 1
    print(f"Token {count}: {token} (mask={token_masks[0]})")

    if token == assistant_end:
        print("  -> assistant_end (stopping)")
        break
    elif token == bos:
        print("  -> bos (stopping)")
        break
    else:
        try:
            decoded = tokenizer.decode([token])
            print(f"  -> '{decoded}'")
        except Exception as e:
            print(f"  -> [decode error: {e}]")

print(f"\nGenerated {count} tokens total")
