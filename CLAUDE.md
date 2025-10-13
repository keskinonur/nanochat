# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a full-stack implementation of an LLM like ChatGPT in a minimal, hackable, dependency-lite codebase. It runs the entire pipeline from tokenization through pretraining, finetuning, evaluation, inference, and web serving on a single 8XH100 node. The project emphasizes simplicity and educational clarity over configurability - it is not a framework but a cohesive "strong baseline" designed to be maximally readable and forkable.

## Common Commands

### Environment Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### Tokenizer
```bash
# Build the Rust tokenizer (required before training)
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Train tokenizer on ~2B characters
python -m scripts.tok_train --max_chars=2000000000

# Evaluate tokenizer
python -m scripts.tok_eval
```

### Training

#### Single GPU (slower, but produces identical results)
```bash
# Base model pretraining
python -m scripts.base_train -- --depth=20

# Midtraining
python -m scripts.mid_train

# Supervised fine-tuning
python -m scripts.chat_sft

# Reinforcement learning (optional)
python -m scripts.chat_rl
```

#### Multi-GPU (8 GPUs)
```bash
# Base model pretraining
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# Supervised fine-tuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Reinforcement learning (optional)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

### Evaluation
```bash
# Evaluate base model on CORE metric
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Evaluate chat model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# Evaluate specific tasks
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GSM8K
```

### Inference
```bash
# Chat via CLI (with optional prompt)
python -m scripts.chat_cli
python -m scripts.chat_cli -p "Why is the sky blue?"

# Serve web UI (ChatGPT-style interface)
python -m scripts.chat_web
```

### Data
```bash
# Download pretraining data shards
python -m nanochat.dataset -n 240  # downloads 240 shards (~24GB)
```

### Testing
```bash
# Run tests (especially for tokenizer)
python -m pytest tests/test_rustbpe.py -v -s
```

### Full Pipeline
```bash
# Run the complete $100 speedrun (takes ~4 hours on 8XH100)
bash speedrun.sh

# Or in a screen session with logging
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# With wandb logging
WANDB_RUN=speedrun bash speedrun.sh
```

## Architecture

### Training Pipeline Stages

1. **Tokenization** (`rustbpe/`): Custom BPE tokenizer written in Rust for efficient training. Exports vocab for use with the model. Trains on ~2B characters to vocab size 65536.

2. **Base Model Pretraining** (`scripts/base_train.py`): Trains the base GPT model from scratch using Chinchilla scaling (20x tokens:params ratio). The d20 model is 561M parameters.

3. **Midtraining** (`scripts/mid_train.py`): Teaches the base model conversation special tokens, tool use (calculator), and multiple choice format. Bridges gap between pretraining and chat.

4. **Supervised Finetuning (SFT)** (`scripts/chat_sft.py`): Domain adaptation on task mixture including ARC, GSM8K, and SmolTalk conversations. Trains on ~21.4K examples.

5. **Reinforcement Learning (Optional)** (`scripts/chat_rl.py`): Currently only supports GSM8K for math reasoning improvements.

### Core Modules

**`nanochat/gpt.py`**: The GPT model implementation with notable features:
- Rotary embeddings (no learned positional embeddings)
- QK normalization
- Untied weights (separate token embedding and lm_head)
- ReLU² activation in MLP
- Norm after token embedding
- No learnable params in RMSNorm
- No bias in linear layers
- Multi-Query Attention (MQA) support for efficient inference

**`nanochat/engine.py`**: Efficient inference engine with:
- KV cache management with dynamic growth
- Batch generation with prefill optimization
- Tool use state machine (calculator integration)
- Per-row state tracking during generation
- Token forcing for tool outputs

**`nanochat/checkpoint_manager.py`**: Handles saving and loading of model checkpoints. Checkpoints are organized by stage:
- `base_checkpoints/`: Base pretrained models
- `mid_checkpoints/`: Midtrained models
- `chatsft_checkpoints/`: SFT models
- `chatrl_checkpoints/`: RL models

Model tags are typically based on depth (e.g., `d20` for 20 layers).

**`nanochat/configurator.py`**: Simple configuration system that allows CLI overrides via `--key=value` syntax. Directly modifies globals() - simple but effective for this codebase.

**`nanochat/dataloader.py`**: Distributed data loading for pretraining data shards.

**`nanochat/tokenizer.py`**: Python wrapper around the Rust BPE tokenizer with conversation rendering support.

**`tasks/`**: Task implementations (MMLU, ARC, GSM8K, HumanEval, SmolTalk) with:
- `Task` base class for datasets
- `TaskMixture` for mixing multiple tasks during training
- `TaskSequence` for curriculum-based sequential training
- Evaluation methods (categorical for multiple choice, generative for open-ended)

### Optimizers

The codebase uses a split optimizer strategy:
- **Muon** (`nanochat/muon.py`): For linear layer weights (matrix parameters)
- **AdamW** (`nanochat/adamw.py`): For embeddings and lm_head
- Learning rates are scaled by ∝1/√(d_model/768) where d_model is the model dimension

### Configuration & Hyperparameters

All training scripts use the configurator pattern. To modify hyperparameters:
```bash
# Override via CLI
python -m scripts.base_train -- --depth=26 --device_batch_size=16

# Or via config file
python -m scripts.base_train config/my_config.py
```

Key hyperparameters:
- `device_batch_size`: Per-GPU batch size (reduce if OOM)
- `depth`: Number of transformer layers (determines model size)
- `max_seq_len`: Context length (default 2048)
- `target_param_data_ratio`: Chinchilla ratio (default 20)

### Memory Management

If you encounter OOM errors:
1. Reduce `device_batch_size` (e.g., 32→16→8→4→2→1)
2. The code automatically compensates with gradient accumulation
3. For larger models (d26+), halve the device_batch_size from default

### Model Sizing

The model architecture is derived from depth:
- `num_layers = depth`
- `model_dim = depth * 64` (aspect ratio of 64)
- `num_heads = max(1, (model_dim + 127) // 128)` (head_dim of 128)
- `num_kv_heads = num_heads` (1:1 MQA ratio)

Examples:
- d20: 561M params, ~$100 on 8XH100 (~4 hours)
- d26: Slightly outperforms GPT-2, ~$300, ~12 hours

### Data Pipeline

Pretraining data comes from HuggingFace FineWeb in shards:
- Each shard: ~250M characters (~100MB compressed)
- Stored in `~/.cache/nanochat/` by default
- Total dataset: 1822 shards available
- For d20: Need ~240 shards (Chinchilla scaling)

### Checkpoint Loading

Use `load_model(source, device, phase)` where:
- `source`: "base" | "mid" | "sft" | "rl"
- `device`: torch device
- `phase`: "train" | "eval"
- Optional: `model_tag` (e.g., "d20") and `step` (checkpoint step)

### Logging & Reporting

- WandB integration: Set `WANDB_RUN=<name>` (or "dummy" to skip)
- Training generates `report.md` with all metrics and evaluation results
- Use `nanochat.report.get_report().log()` to add to report

### Special Tokens

The tokenizer uses special tokens for conversation and tool use:
- `<|bos|>`: Beginning of sequence
- `<|user_start|>`, `<|user_end|>`: User messages
- `<|assistant_start|>`, `<|assistant_end|>`: Assistant messages
- `<|python_start|>`, `<|python_end|>`: Calculator expressions
- `<|output_start|>`, `<|output_end|>`: Calculator outputs

### Evaluation Metrics

- **CORE**: Comprehensive metric across multiple base model tasks
- **ARC** (Easy/Challenge): Multiple choice reasoning
- **GSM8K**: Grade school math problems
- **HumanEval**: Code generation
- **MMLU**: Massive multitask language understanding
- **SmolTalk**: Conversational quality (ChatCORE)

### Directory Structure

- `nanochat/`: Core library code (models, training, data)
- `scripts/`: Entry points for training, eval, inference
- `tasks/`: Task dataset implementations
- `rustbpe/`: Rust tokenizer implementation
- `tests/`: Test suite (primarily tokenizer tests)
- `dev/`: Development utilities
- `~/.cache/nanochat/`: Default location for artifacts (data, checkpoints, eval bundles)

### Inference Details

For chat applications, use the `Engine` class from `nanochat/engine.py`:
- Supports batch generation with shared prefill
- Implements tool use (calculator) via state machine
- KV cache with automatic prefilling and growth
- Temperature and top-k sampling

The simple `model.generate()` method exists but is slower - use Engine for production inference.
