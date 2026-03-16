#!/usr/bin/env python3
"""
Hyperparameter search swarm loop: tweak → train → publish → repeat.

Usage:
    uv run python run_swarm.py [--cycles N] [--agent-id NAME] [--mode baseline|search]

Modes:
  baseline  — run the current train.py config repeatedly (no changes)
  search    — vary one hyperparameter per cycle, keep improvements, revert failures

train.py in the repo is never modified. Search mode uses a temporary copy.
When a better config is found, it prints the winning settings so you can
update train.py deliberately.
"""

import argparse
import hashlib
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime

from coordinator import Coordinator

TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")

# ---------------------------------------------------------------------------
# Hyperparameter search space (one tweak per cycle)
# ---------------------------------------------------------------------------

SEARCH_SPACE = [
    # (param_name, regex_pattern, candidates_fn)
    # candidates_fn takes current value and returns list of alternatives to try
    ("MATRIX_LR", r"^MATRIX_LR\s*=\s*([\d.]+)",
     lambda v: [round(v * f, 4) for f in [0.75, 0.85, 1.15, 1.25]]),
    ("WEIGHT_DECAY", r"^WEIGHT_DECAY\s*=\s*([\d.]+)",
     lambda v: [round(v * f, 3) for f in [0.7, 0.85, 1.15, 1.3]]),
    ("EMBEDDING_LR", r"^EMBEDDING_LR\s*=\s*([\d.]+)",
     lambda v: [round(v * f, 2) for f in [0.7, 0.85, 1.15, 1.3]]),
    ("UNEMBEDDING_LR", r"^UNEMBEDDING_LR\s*=\s*([\d.]+)",
     lambda v: [round(v * f, 4) for f in [0.75, 0.85, 1.15, 1.25]]),
    ("SCALAR_LR", r"^SCALAR_LR\s*=\s*([\d.]+)",
     lambda v: [round(v * f, 2) for f in [0.7, 0.85, 1.15, 1.3]]),
    ("FINAL_LR_FRAC", r"^FINAL_LR_FRAC\s*=\s*([\d.]+)",
     lambda v: [round(x, 3) for x in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05] if x != v]),
    ("WARMDOWN_RATIO", r"^WARMDOWN_RATIO\s*=\s*([\d.]+)",
     lambda v: [round(x, 2) for x in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0] if x != v]),
    ("ASPECT_RATIO", r"^ASPECT_RATIO\s*=\s*(\d+)",
     lambda v: [x for x in [48, 52, 56, 60, 64] if x != v]),
    ("TOTAL_BATCH_SIZE", r"^TOTAL_BATCH_SIZE\s*=.*else\s*2\*\*(\d+)",
     lambda v: [x for x in [14, 15, 16, 17] if x != v]),
]

_tried = set()


def get_current_value(source, param_name, pattern):
    for line in source.split("\n"):
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    return None


def apply_tweak(source, param_name, pattern, old_val, new_val):
    lines = source.split("\n")
    new_lines = []
    for line in lines:
        m = re.match(pattern, line)
        if m:
            if param_name == "TOTAL_BATCH_SIZE":
                line = re.sub(r"else\s*2\*\*\d+", f"else 2**{int(new_val)}", line)
            else:
                old_str = m.group(1)
                if "." in old_str or "." in str(new_val):
                    new_str = str(float(new_val))
                else:
                    new_str = str(int(new_val))
                line = line[:m.start(1)] + new_str + line[m.end(1):]
        new_lines.append(line)
    return "\n".join(new_lines)


def pick_tweak(source):
    random.shuffle(SEARCH_SPACE)
    for param_name, pattern, candidates_fn in SEARCH_SPACE:
        current = get_current_value(source, param_name, pattern)
        if current is None:
            continue
        candidates = candidates_fn(current)
        random.shuffle(candidates)
        for candidate in candidates:
            key = f"{param_name}={candidate}"
            if key not in _tried:
                _tried.add(key)
                return param_name, pattern, current, candidate
    _tried.clear()
    return pick_tweak(source)


# ---------------------------------------------------------------------------
# Training cycle
# ---------------------------------------------------------------------------

def parse_train_output(output: str) -> dict:
    metrics = {}
    for line in output.strip().split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("step"):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def run_cycle(coord, cycle_num, mode, best_bpb, best_source):
    """Run one train+eval cycle. Returns (result_dict, winning_source_or_None)."""
    run_id = hashlib.md5(f"{time.time()}-{cycle_num}".encode()).hexdigest()[:6]

    base_source = best_source
    tweak_info = None
    train_source = base_source

    if mode == "search":
        param_name, pattern, old_val, new_val = pick_tweak(base_source)
        train_source = apply_tweak(base_source, param_name, pattern, old_val, new_val)
        tweak_info = {"param": param_name, "old": old_val, "new": new_val}
        if param_name == "TOTAL_BATCH_SIZE":
            desc = f"{param_name} 2**{int(old_val)}->2**{int(new_val)} (M4 MPS) [{run_id}]"
        else:
            desc = f"{param_name} {old_val}->{new_val} (M4 MPS) [{run_id}]"
        print(f"[search] Tweaking: {desc}")
    else:
        desc = f"baseline MPS run cycle {cycle_num} (M4 batch8) [{run_id}]"

    print(f"\n{'='*54}")
    print(f"  CYCLE {cycle_num} {'[SEARCH]' if mode == 'search' else '[BASELINE]'}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if tweak_info:
        print(f"  {tweak_info['param']}: {tweak_info['old']} -> {tweak_info['new']}")
    print(f"{'='*54}\n")

    # Claim experiment
    exp_key = coord.claim_experiment(desc)
    if not exp_key:
        print("[swarm] Claim failed, running with timestamped key...")
        desc = f"{desc} t={int(time.time())}"
        exp_key = coord.claim_experiment(desc)

    # Write temp copy in the project directory (so imports work)
    project_dir = os.path.dirname(__file__)
    tmp_train = os.path.join(project_dir, "_train_experiment.py")
    with open(tmp_train, "w") as f:
        f.write(train_source)

    print("[swarm] Starting train.py...")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, tmp_train],
        capture_output=True,
        text=True,
        cwd=project_dir,
    )

    elapsed = time.time() - t0
    try:
        os.remove(tmp_train)
    except OSError:
        pass

    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            print(line)

    if result.returncode != 0:
        print(f"[swarm] train.py FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-500:])
        status = "discard"
        val_bpb = None
        metrics = {}
    else:
        metrics = parse_train_output(result.stdout)
        val_bpb = metrics.get("val_bpb")

        if val_bpb is None:
            print("[swarm] Could not parse val_bpb!")
            return {"status": "parse_error", "elapsed": elapsed}, None

        if mode == "search" and best_bpb is not None and val_bpb >= best_bpb:
            print(f"[search] {val_bpb:.6f} >= best {best_bpb:.6f} -- DISCARD")
            status = "discard"
        else:
            if mode == "search" and best_bpb is not None:
                print(f"[search] {val_bpb:.6f} < best {best_bpb:.6f} -- KEEP!")
            status = "keep"

        print(f"\n[swarm] {'KEEP' if status == 'keep' else 'DISCARD'} val_bpb = {val_bpb:.6f} in {elapsed:.0f}s")

    # Publish to Ensue
    if exp_key and val_bpb is not None:
        coord.publish_result(
            experiment_key=exp_key,
            val_bpb=val_bpb,
            memory_gb=metrics.get("peak_vram_mb", 0) / 1024,
            status=status,
            description=desc,
            train_py_source=train_source,
            extra_metrics={
                "training_seconds": metrics.get("training_seconds"),
                "total_seconds": metrics.get("total_seconds"),
                "mfu_percent": metrics.get("mfu_percent"),
                "num_steps": metrics.get("num_steps"),
                "num_params_M": metrics.get("num_params_M"),
                "total_tokens_M": metrics.get("total_tokens_M"),
                "depth": metrics.get("depth"),
                "device": "mps",
                "chip": "M4",
                "tweak": tweak_info,
            },
        )
        if tweak_info:
            insight = f"{tweak_info['param']} {tweak_info['old']}->{tweak_info['new']}: "
            insight += f"val_bpb={val_bpb:.6f} ({status}). "
            if status == "keep":
                insight += f"Improved by {best_bpb - val_bpb:.6f} on M4 MPS (~33 steps/5min)."
            else:
                insight += f"No improvement on M4 MPS (~33 steps/5min)."
            coord.post_insight(insight)

        print(f"[swarm] Published to Ensue: {exp_key}")

    winning_source = train_source if status == "keep" else None

    return {
        "status": status if val_bpb is not None else "fail",
        "val_bpb": val_bpb,
        "elapsed": elapsed,
        "tweak": tweak_info,
    }, winning_source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autoresearch swarm loop")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0 = infinite)")
    parser.add_argument("--agent-id", default="mave-m4", help="Agent identifier")
    parser.add_argument("--mode", choices=["baseline", "search"], default="search",
                        help="baseline = repeat same config, search = vary hyperparameters")
    args = parser.parse_args()

    coord = Coordinator()
    coord.agent_id = args.agent_id

    if not coord.test_connectivity():
        print("ERROR: Cannot connect to Ensue. Check .autoresearch-key")
        sys.exit(1)

    coord.announce()

    # Start from the repo's train.py as baseline
    with open(TRAIN_SCRIPT) as f:
        best_source = f.read()

    cycle = 1
    results = []
    best_bpb = None

    try:
        while True:
            if args.cycles > 0 and cycle > args.cycles:
                break

            result, winning_source = run_cycle(coord, cycle, args.mode, best_bpb, best_source)
            results.append(result)

            if winning_source and result.get("val_bpb") is not None:
                if best_bpb is None or result["val_bpb"] < best_bpb:
                    best_bpb = result["val_bpb"]
                    best_source = winning_source
                    if result.get("tweak"):
                        tw = result["tweak"]
                        print(f"\n[search] NEW BEST: {best_bpb:.6f}")
                        print(f"[search] Winning change: {tw['param']} {tw['old']} -> {tw['new']}")

            # Summary
            keeps = [r for r in results if r["status"] == "keep"]
            discards = [r for r in results if r["status"] == "discard"]
            if best_bpb:
                print(f"\n[swarm] Cycles: {len(results)} | Keeps: {len(keeps)} | Discards: {len(discards)} | Best: {best_bpb:.6f}")

            cycle += 1

            if args.cycles > 0 and cycle > args.cycles:
                break

            print("[swarm] Next cycle in 10s...")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n[swarm] Interrupted. Shutting down.")

    # Final summary
    keeps = [r for r in results if r["status"] == "keep"]
    print(f"\n{'='*54}")
    print(f"  SWARM SESSION COMPLETE")
    print(f"  Cycles: {len(results)} ({len(keeps)} kept, {len(results) - len(keeps)} discarded)")
    if best_bpb:
        print(f"  Best val_bpb: {best_bpb:.6f}")
    if results:
        print(f"  Total time: {sum(r['elapsed'] for r in results):.0f}s")
    if keeps:
        print(f"\n  To apply the best config, update train.py with these changes:")
        for r in keeps:
            tw = r.get("tweak")
            if tw:
                print(f"    {tw['param']}: {tw['old']} -> {tw['new']} (val_bpb={r['val_bpb']:.6f})")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
