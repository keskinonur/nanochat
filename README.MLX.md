# NanoChat MLX Port

Fully-featured MLX rewrite of [nanochat](https://github.com/karpathy/nanochat) tuned for Apple Silicon.

## Status
- ✅ **Port complete** – core model, training, inference, evaluation, and tooling run natively on MLX.
- ✅ **Infrastructure fixes (Oct 19 2025)** – top‑k sampling, FLOP estimation, and loss masking now align with MLX semantics.
- ✅ **Reference SFT checkpoint** – `~/.cache/nanochat/chatsft_checkpoints/d8/model_000400.npz` (mirrored at `models/d8_sft_good.npz`) provides the best chat quality to date.

> Test the checkpoint quickly:
> ```bash
> python -m scripts.mlx.chat_cli_mlx \
>   -i sft -g d8 -s 400 -t 0.2 -k 30 -p "Hi there!"
> ```

## Quick Start

```bash
# 1. Install dependencies (uses uv)
uv sync

# 2. Verify environment
./tests/mlx/run_all_tests.sh
python -m scripts.mlx.diagnose_mlx

# 3. Prepare data (downloads FineWeb shards & trains tokenizer)
python -m nanochat.dataset -n 50
```

## Training Workflow (Apple Silicon)

| Stage | Command | Notes |
| --- | --- | --- |
| Base pretraining | `python -m scripts.mlx.base_train_mlx_v2 --depth=20 --use_real_data=True` | Single-device MLX loop; adjust `device_batch_size` to avoid swap pressure. |
| Midtraining | `python -m scripts.mlx.mid_train_mlx --model_tag=d20` | Formats base model for conversational data. |
| Supervised fine-tuning (SFT) | `python -m scripts.mlx.chat_sft_mlx --model_tag=d20 --device_batch_size=4` | Latest run converged best around step 400; keep LR modest after that. |
| (Optional) RL finetune | `python -m scripts.mlx.chat_rl_mlx --model_tag=d20` | Experimental; requires additional reward setup. |

### Safe defaults
- `device_batch_size`: 4–8 on 64 GB, 8–16 on 128 GB+, depending on sequence length.
- `total_batch_size`: 16,384 for pretraining scripts (already scaled down from PyTorch).
- Prefer `temperature ≤ 0.3` and modest `top_k` for early SFT checkpoints.

## Inference & Evaluation

```bash
# CLI chat (streaming)
python -m scripts.mlx.chat_cli_mlx -i sft -g d8 -s 400

# Web UI (http://localhost:8000)
python -m scripts.mlx.chat_web_mlx -i sft -g d8 -s 400

# CORE metric evaluation
python -m scripts.mlx.base_eval_mlx

# Chat benchmark sweep (ARC, MMLU, GSM8K, HumanEval)
python -m scripts.mlx.chat_eval_mlx -i sft -a "ARC-Easy|MMLU|GSM8K|HumanEval"
```

## Architecture Snapshot

- **Core MLX model** – `nanochat/gpt.py` (rotary embeddings, QK norm, MQA, relu² MLP, functional RMSNorm).
- **Optimizers** – `nanochat/muon.py` (Newton–Schulz orthogonalisation) and `nanochat/adamw.py`.
- **Inference engine** – `nanochat/engine.py` (streaming, KV cache growth, calculator tool support).
- **Checkpointing** – `.npz` format via `nanochat/checkpoint_manager.py`.
- **Data pipeline** – MLX-native arrays and unified memory, `nanochat/dataloader.py`.
- **Tests** – `tests/mlx/` covers init, forward, engine, KV cache, optimisers, training loop, checkpoint round-trip.

## History & Milestones
- **Port foundation** – Started from the PyTorch reference, rewrote the GPT model, optimisers, dataloaders, engine, and checkpoint handling to pure MLX while embracing unified memory and single-device execution.
- **Training & evaluation** – Reimplemented base, mid, and SFT pipelines plus the CORE/chat evaluation suite on Apple Silicon; all automated tests run under MLX.
- **Runtime hardening (Oct 19 2025)** – Fixed top‑k sampling semantics, MLX tree reductions for FLOP estimates, and loss masking so training and inference behave like the original implementation.
- **Reference fine-tune** – Conducted MLX SFT on conversation data; step 400 (`d8`) currently yields the best chat experience and is mirrored at `models/d8_sft_good.npz` for convenience.

## Documentation & Support
- This README is the single source of truth for the MLX port.
- Historical progress reports and intermediate guides have been archived; consult git history if you need past milestones or file inventories.
- For code-level insights, read the modules referenced in the *Architecture Snapshot* section and the original upstream docs at [karpathy/nanochat](https://github.com/karpathy/nanochat).

## Hardware Guidance

| Chip | RAM | Recommended use |
| --- | --- | --- |
| M1/M2 (8–16 GB) | Minimal | Toy training (depth ≤ 4) and inference. |
| M1/M2 Pro/Max (32–64 GB) | Medium | Models up to ~300 M params. |
| M2 Ultra (128–192 GB) | Ideal | Full 400 M param baseline; matches reference runs. |

Unified memory means no manual `.to(device)` calls; MLX schedules CPU/GPU transparently.

## Release Checklist
- [x] Core + mid + SFT training pipelines in MLX.
- [x] Inference stack with streaming + tool use.
- [x] Evaluation suite (CORE + chat tasks).
- [x] MLX-native tests and diagnostics.
- [x] Documentation cleaned up (this README is now canonical).

## License & Credits
- Same license as upstream nanochat.
- Thanks to [@karpathy](https://github.com/karpathy) for the original project and the MLX team for the runtime.

Happy training on Apple Silicon! 🚀
