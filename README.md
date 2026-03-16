# autoresearch-at-home — Apple Silicon Fork

Fork of [autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) that runs on **Mac with Apple Silicon**. Join the collaborative research swarm from your Mac mini, MacBook, or any M-series machine.

The swarm is a distributed experiment: dozens of machines collectively figuring out the best way to train a small language model in 5 minutes. Each machine tries different training settings, reports what worked, and everyone learns from each other's results.

## Quick start

```bash
git clone https://github.com/bdecrem/autoresearch-at-home.git
cd autoresearch-at-home
uv sync                    # install dependencies
uv run prepare.py          # download data + train tokenizer (one-time, ~5 min)
uv run train.py            # single 5-minute training run to verify setup
```

## Join the swarm

Get an API key from the hub admin, then:

```bash
echo "your-key" > .autoresearch-key
uv run python run_swarm.py --cycles 5 --agent-id your-name --mode baseline
```

This runs 5 training cycles and publishes results to the shared network. Each cycle takes about **15 minutes** (10 min startup + 5 min training). The startup time is `torch.compile` — it's slow on MPS but only happens once per run.

## Help find better settings

The default config is our current best for Apple Silicon. To help improve it:

```bash
uv run python run_swarm.py --cycles 10 --agent-id your-name --mode search
```

Search mode automatically tweaks one hyperparameter per cycle (learning rates, weight decay, batch size, etc.), keeps changes that improve the score, and discards the rest. It never modifies `train.py` in the repo — it uses a temporary copy. At the end it prints the winning changes so you can update the repo deliberately.

## Current best (Mac mini M4, 16GB)

| Metric | Score |
|--------|-------|
| val_bpb | 1.958 |
| Steps in 5 min | ~33 |
| Throughput | ~2,700 tok/sec |
| Peak memory | 4.7 GB |

For comparison, an H100 scores 0.926 with 1,500+ steps. The gap is hardware — we get ~33 training steps in 5 minutes vs 1,000+ on CUDA. The hyperparameter search is about squeezing the most learning out of those limited steps.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ unified memory
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## What changed from upstream

All changes are conditional — CUDA paths are untouched when running on NVIDIA GPUs.

- **Flash Attention 3 → PyTorch SDPA**: FA3 is CUDA-only
- **float32 precision**: MPS float16/bfloat16 causes NaN for this workload
- **`torch.compile` with `aot_eager` backend**: Works on MPS, cuts startup from hours to ~10 min
- **Batch size and total batch tuned for 16GB unified memory**
- **`kernels` dependency skipped on macOS**: CUDA-only, breaks install on Mac
- **MPS memory reporting and VRAM tier detection**: So the swarm leaderboard classifies Apple Silicon correctly

## Project structure

```
train.py        — model + training loop (the file experiments modify)
prepare.py      — data download + tokenizer (run once, don't modify)
coordinator.py  — Ensue swarm coordination (claiming, publishing, leaderboard)
run_swarm.py    — swarm loop: baseline mode or hyperparameter search mode
program.md      — experiment protocol
collab.md       — collaborative swarm protocol
```

## Troubleshooting

- **MPS deadlock** (0% CPU, high memory): Kill the process and restart. Known PyTorch MPS issue.
- **Out of memory**: Reduce `DEVICE_BATCH_SIZE` in train.py (default 8, try 4).
- **Slow first run**: `torch.compile` takes ~10 minutes on MPS. Subsequent runs in the same session reuse the cache.

---

*Upstream: [mutable-state-inc/autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) — MIT License*
