# autoresearch-at-home — Apple Silicon (MPS) Fork

This fork adds **Apple Silicon support** to [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home), letting you run collaborative AI training on any Mac with an M-series chip.

The original project is CUDA-only. We patched `train.py` and `prepare.py` so everything auto-detects MPS vs CUDA at startup — no manual config needed.

## Benchmark (Mac mini M4, 10-core GPU, 16GB)

| Metric | This fork (MPS) | Original (H100) |
|--------|----------------|-----------------|
| Steps in 5 min | 100 | ~1,500+ |
| Final loss | 5.33 | ~1.8 |
| Throughput | ~4,800 tok/sec | ~500K+ tok/sec |
| Time per step | ~3.4s | ~0.2s |

~100x slower than H100 — inherent hardware gap (no Flash Attention 3, no `torch.compile` on MPS). But it **works**, and that's the point.

## Quick start

```bash
git clone https://github.com/bdecrem/autoresearch-at-home.git
cd autoresearch-at-home
uv sync           # install deps
uv run prepare.py # download data + train tokenizer
uv run train.py   # auto-detects MPS, trains for 5 minutes
```

No CUDA required. No config changes. Just clone and run.

## What we changed

All changes are conditional — CUDA paths are untouched when running on NVIDIA GPUs.

- **Device auto-detection**: MPS → CUDA → CPU fallback at startup
- **Flash Attention 3 → PyTorch SDPA**: FA3 is CUDA-only; we use `F.scaled_dot_product_attention` on MPS
- **float32 instead of bfloat16**: MPS bfloat16 is unstable; float32 runs natively without conversion overhead
- **No `torch.compile`**: Not yet supported on MPS backend
- **Batch size 4** (vs 64 on CUDA): Fits in 16GB unified memory
- **Total batch 16K tokens** (vs 524K): Adjusted for smaller per-device batch
- **Reduced eval tokens**: 10×524K (vs 40×524K) for faster eval passes
- **`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`**: Lets MPS use all available memory
- **Device-aware dataloader**: Conditional `pin_memory`, device buffer allocation
- **Fixed sync/memory calls**: `torch.mps.synchronize()` and `torch.mps.driver_allocated_memory()`

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- At least 16GB unified memory
- Python 3.10+
- PyTorch 2.1+ (ships with MPS backend)

## Project structure

```
train.py        — model, optimizer, training loop (MPS + CUDA)
prepare.py      — data prep + tokenizer (MPS + CUDA)
coordinator.py  — Ensue integration for the research swarm
pyproject.toml  — dependencies (uses default PyPI torch for Mac compatibility)
```

---

*For the original project, CUDA setup, and collaborative swarm protocol, see the [upstream repo](https://github.com/mutable-state-inc/autoresearch-at-home).*

## License

MIT
