# MLX Port Status

This document tracks the progress of porting nanochat to MLX for Apple Silicon.

## Target Hardware
- **Device**: Apple Silicon M2 Ultra
- **Unified RAM**: 192GB
- **Advantages**: Unified memory architecture, no CPU<->GPU data transfers

## Completed Components ✅

### Core Infrastructure
- [x] **pyproject.toml** - Replaced PyTorch with MLX (>=0.21.0)
- [x] **common.py** - MLX device management, removed distributed training
- [x] **gpt.py** - Complete GPT model port with:
  - Rotary embeddings
  - QK normalization
  - Multi-Query Attention (MQA)
  - ReLU² activation
  - RMSNorm (no learnable params)
  - Causal attention (training & inference modes)
- [x] **muon.py** - Custom Muon optimizer with Newton-Schulz orthogonalization
- [x] **adamw.py** - Wrapper around MLX's built-in AdamW
- [x] **engine.py** - Inference engine with:
  - KV cache management
  - Dynamic cache growth
  - Tool use (calculator integration)
  - Batch generation support
- [x] **dataloader.py** - Data loading with MLX arrays and unified memory
- [x] **checkpoint_manager.py** - Model save/load using `.npz` format

## Remaining Work 🚧

### Training Scripts
- [ ] **base_train.py** - Base model pretraining
- [ ] **mid_train.py** - Midtraining for conversation format
- [ ] **chat_sft.py** - Supervised fine-tuning
- [ ] **chat_rl.py** - Reinforcement learning (optional)

### Evaluation Scripts
- [ ] **base_eval.py** - CORE metric evaluation
- [ ] **base_loss.py** - Loss evaluation
- [ ] **chat_eval.py** - Chat model evaluation
- [ ] **core_eval.py** - Core evaluation utilities
- [ ] **loss_eval.py** - Loss evaluation utilities

### Inference Scripts
- [ ] **chat_cli.py** - CLI chat interface
- [ ] **chat_web.py** - Web UI for ChatGPT-style interface

### Other Components
- [ ] **execution.py** - Execution utilities (if needed)
- [ ] **report.py** - Reporting utilities
- [ ] **speedrun.sh** - MLX-specific speedrun script

### Tasks
- [ ] **tasks/*.py** - Evaluation task implementations (may work as-is)

## Key Architectural Differences

### PyTorch → MLX Changes

1. **No Distributed Training**
   - MLX focuses on single-device
   - Removed all `torchrun` and DDP code
   - Simplified to single M2 Ultra device

2. **Unified Memory Model**
   - No explicit `.to(device)` calls needed
   - Automatic memory management between CPU and GPU
   - More efficient with 192GB unified RAM

3. **Array Operations**
   - `torch.tensor` → `mx.array`
   - `torch.cuda.synchronize()` → not needed (automatic)
   - Different API for some operations (e.g., masking, slicing)

4. **Model Saving**
   - `.pt` files → `.npz` files
   - `torch.save/load` → `mx.savez/load`

5. **Autocast/Mixed Precision**
   - MLX handles precision automatically
   - No explicit autocast context needed

6. **Compilation**
   - No `torch.compile()` equivalent yet
   - MLX uses JIT compilation automatically

## Testing Strategy

1. **Unit Tests**
   - Test individual components (GPT model, optimizers, engine)
   - Verify KV cache correctness
   - Check tokenizer integration

2. **Integration Tests**
   - Test data loading pipeline
   - Verify checkpoint save/load
   - Test inference engine end-to-end

3. **Training Test**
   - Small-scale training run (few steps)
   - Verify loss decreases
   - Check memory usage

4. **Full Pipeline**
   - Run shortened speedrun (reduced data/steps)
   - Verify all stages work
   - Generate samples from trained model

## Performance Expectations

### vs PyTorch on 8XH100
- **Training Speed**: Likely slower (single device vs 8 GPUs)
- **Memory Efficiency**: Better (unified 192GB vs 8x80GB with copies)
- **Cost**: $0 (already own hardware) vs ~$24/hour

### Optimization Opportunities
- Batch size can be increased due to 192GB RAM
- No inter-GPU communication overhead
- Efficient KV cache with unified memory
- MLX is optimized for Apple Silicon

## Next Steps

1. Port remaining training scripts
2. Port evaluation infrastructure
3. Port inference scripts
4. Create MLX-specific speedrun script
5. Run comprehensive tests
6. Document MLX-specific optimizations
7. Update CLAUDE.md with MLX instructions

## Notes

- The tokenizer (rustbpe) remains unchanged - works with both PyTorch and MLX
- Most task implementations should work as-is (they operate on token IDs)
- The web server (FastAPI) is framework-agnostic
- Focus on getting a minimal training→inference pipeline working first
