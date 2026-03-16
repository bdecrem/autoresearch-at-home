# autoresearch-at-home — Apple Silicon Fork

Fork of [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) that runs on **Mac with Apple Silicon**. The original is CUDA-only. This fork auto-detects MPS vs CUDA at startup — no config needed, and all CUDA paths are untouched.

## Quick start

```bash
git clone https://github.com/bdecrem/autoresearch-at-home.git
cd autoresearch-at-home
uv sync
uv run prepare.py          # download data + train tokenizer (~5 min)
uv run train.py            # single 5-minute training run
```

To run multiple cycles with swarm coordination:

```bash
uv run python run_swarm.py --cycles 5 --agent-id my-mac
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ unified memory
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Benchmark (Mac mini M4, 10-core GPU, 16GB)

| Metric | This fork (MPS) | Original (H100) |
|--------|----------------|-----------------|
| Steps in 5 min | 23 | ~1,500+ |
| val_bpb | 2.01 | ~1.8 |
| Throughput | ~2,700 tok/sec | ~500K+ tok/sec |
| Peak memory | 4.1 GB | varies |
| Startup time | ~10 min | ~30s |

~60x slower than H100 — inherent hardware gap. But it works, reports real metrics, and participates in the swarm.

## What changed from upstream

- **Device auto-detection**: MPS → CUDA → CPU fallback
- **FA3 → PyTorch SDPA**: Flash Attention 3 is CUDA-only; we use `F.scaled_dot_product_attention` on MPS
- **float32 precision**: MPS float16/bfloat16 is unstable for this workload
- **`torch.compile` with `aot_eager`**: Works on MPS, cuts startup from hours to minutes
- **Batch size 8, total batch 64K tokens**: Tuned for 16GB unified memory
- **`kernels` dependency skipped on macOS**: CUDA-only package, breaks `uv sync` on Mac
- **MPS memory reporting**: `peak_vram_mb` now reports actual usage via `torch.mps.driver_allocated_memory()`
- **VRAM tier detection**: Apple Silicon detected via system RAM for swarm leaderboard

## Project structure

```
train.py        — model, optimizer, training loop (the file you modify for experiments)
prepare.py      — data download + tokenizer training (run once, don't modify)
coordinator.py  — Ensue swarm coordination (claiming, publishing, leaderboard)
run_swarm.py    — automated loop: train → publish → repeat
setup_hub.py    — one-time Ensue hub initialization (admin only)
program.md      — experiment protocol and loop guidelines
collab.md       — collaborative swarm coordination protocol
pyproject.toml  — dependencies
```

## Swarm participation

To join the collaborative research swarm:

1. Get an API key from the hub admin
2. Save it: `echo "your-key" > .autoresearch-key`
3. Run: `uv run python run_swarm.py --cycles 5 --agent-id your-name`

Results are published to the shared Ensue network. See [collab.md](collab.md) for the full protocol.

## Troubleshooting

- **MPS deadlock** (0% CPU, high memory): Kill the process and restart. This is a known PyTorch MPS issue.
- **`kernels` install error**: Should not happen on this fork — the dependency is skipped on macOS. If it does, run `uv sync` again.
- **Out of memory**: Reduce `DEVICE_BATCH_SIZE` in train.py (default 8, try 4).

---

*Upstream: [mutable-state-inc/autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) — MIT License*
