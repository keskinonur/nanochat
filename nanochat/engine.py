"""
Engine for efficient inference of our models - MLX port for Apple Silicon.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

The whole thing is made as efficient as possible using MLX's unified memory model.
"""

import mlx.core as mx
import mlx.nn as nn
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        return None

def use_calculator(expr):
    """Evaluate a math expression safely."""
    expr = expr.replace(",", "")
    if any([x not in "0123456789*+-/.() " for x in expr]):
        return None
    if "**" in expr:
        return None
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    MLX port uses unified memory for efficient caching.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0  # current position in time in the cache

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                assert dim1 == dim2, f"Dim mismatch: {dim1} != {dim2}"
            elif ix == 2:
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"

        # 2) initialize the cache
        self.kv_cache = mx.zeros(self.kv_shape, dtype=other.kv_cache.dtype)

        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache[:, :, :, :, :other.pos, :]

        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache
        if self.kv_cache is None:
            self.kv_cache = mx.zeros(self.kv_shape, dtype=k.dtype)

        # Insert new keys/values to the cache
        B, H, T_add, D = k.shape
        t0, t1 = self.pos, self.pos + T_add

        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.shape[4]:
            t_needed = t1 + 1024
            t_needed = (t_needed + 1023) & ~1023  # round up to nearest multiple of 1024
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            # Create new larger cache and copy
            new_cache = mx.zeros(current_shape, dtype=self.kv_cache.dtype)
            new_cache[:, :, :, :, :self.kv_cache.shape[4], :] = self.kv_cache
            self.kv_cache = new_cache

        # Insert k, v into the cache
        # MLX doesn't support item assignment, so we need to use array operations
        cache_k = self.kv_cache[layer_idx, 0]
        cache_v = self.kv_cache[layer_idx, 1]

        # Update the cache slices
        cache_k = mx.concatenate([
            cache_k[:, :, :t0, :],
            k,
            cache_k[:, :, t1:, :]
        ], axis=2)
        cache_v = mx.concatenate([
            cache_v[:, :, :t0, :],
            v,
            cache_v[:, :, t1:, :]
        ], axis=2)

        # Update the cache
        kv_layer = mx.stack([cache_k, cache_v], axis=0)
        kv_cache_list = [
            self.kv_cache[i] if i != layer_idx else kv_layer
            for i in range(self.kv_cache.shape[0])
        ]
        self.kv_cache = mx.stack(kv_cache_list, axis=0)

        # Return the full cached keys/values up to current position
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]

        # Increment pos after the last layer processes
        if layer_idx == self.kv_cache.shape[0] - 1:
            self.pos = t1

        return key_view, value_view


# -----------------------------------------------------------------------------
def sample_next_token(logits, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"

    if temperature == 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    if top_k is not None:
        k = min(top_k, logits.shape[-1])
        vals, idx = mx.topk(logits, k, axis=-1)
        vals = vals / temperature
        probs = mx.softmax(vals, axis=-1)
        # Sample from top-k
        choice = mx.random.categorical(mx.log(probs), num_samples=1)
        return mx.take_along_axis(idx, choice, axis=-1)
    else:
        logits = logits / temperature
        probs = mx.softmax(logits, axis=-1)
        return mx.random.categorical(mx.log(probs), num_samples=1)


# -----------------------------------------------------------------------------
class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Generate tokens with KV cache optimization"""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        mx.random.seed(seed)

        # Get special tokens
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": m.n_layer
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )

        ids = mx.array([tokens], dtype=mx.int32)
        logits = self.model(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, temperature, top_k)
        sampled_tokens = [int(next_ids[0, 0])]

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop conditions
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            # Get sampled tokens
            if first_iteration:
                sampled_tokens = [sampled_tokens[0]] * num_samples
                first_iteration = False
            else:
                logits = self.model(ids, kv_cache=kv_cache_decode)
                logits = logits[:, -1, :]
                next_ids = sample_next_token(logits, temperature, top_k)
                sampled_tokens = [int(next_ids[i, 0]) for i in range(num_samples)]

            # Process each row
            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)

                state.current_tokens.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            ids = mx.array([token_column], dtype=mx.int32).T  # (B, 1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that returns final token sequences.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break

        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to verify the Engine works correctly.
    """
    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence
    generated_tokens = []
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens

    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0]
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")

    # compare sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
