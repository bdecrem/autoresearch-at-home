#!/usr/bin/env python3
"""
Hyperparameter search swarm loop: adopt best → tweak → train → publish → repeat.

Usage:
    uv run python run_swarm.py [--cycles N] [--agent-id NAME] [--mode baseline|search]

Modes:
  baseline  — run the current train.py config repeatedly (no changes)
  search    — vary one hyperparameter per cycle, keep improvements, revert failures

Each cycle:
  1. (search mode) Pick a hyperparameter to tweak in train.py
  2. Claim experiment on Ensue
  3. Run train.py (5-min training + eval)
  4. Publish result, insight, and hypothesis to swarm
  5. Keep or revert the change based on val_bpb
"""

import argparse
import copy
import hashlib
import json
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
     lambda v: [round(x, 2) for x in [0.5, 0.7, 0.8, 0.9, 1.0] if x != v]),
    ("ASPECT_RATIO", r"^ASPECT_RATIO\s*=\s*(\d+)",
     lambda v: [x for x in [48, 52, 56, 60, 64] if x != v]),
    ("TOTAL_BATCH_SIZE", r"^TOTAL_BATCH_SIZE\s*=.*else\s*2\*\*(\d+)",
     lambda v: [x for x in [14, 15, 16, 17] if x != v]),
]

# Track what we've already tried so we don't repeat
_tried = set()


def read_train_py():
    with open(TRAIN_SCRIPT) as f:
        return f.read()


def write_train_py(source):
    with open(TRAIN_SCRIPT, "w") as f:
        f.write(source)


def get_current_value(source, param_name, pattern):
    """Extract current value of a hyperparameter from train.py source."""
    for line in source.split("\n"):
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    return None


def apply_tweak(source, param_name, pattern, old_val, new_val):
    """Replace a hyperparameter value in train.py source. Returns new source."""
    lines = source.split("\n")
    new_lines = []
    for line in lines:
        m = re.match(pattern, line)
        if m:
            if param_name == "TOTAL_BATCH_SIZE":
                # Special case: replace the exponent in "else 2**N"
                line = re.sub(r"else\s*2\*\*\d+", f"else 2**{int(new_val)}", line)
            else:
                # Replace the matched number with the new value
                old_str = m.group(1)
                if "." in old_str or "." in str(new_val):
                    new_str = str(float(new_val))
                else:
                    new_str = str(int(new_val))
                line = line[:m.start(1)] + new_str + line[m.end(1):]
        new_lines.append(line)
    return "\n".join(new_lines)


def pick_tweak(source):
    """Pick a random hyperparameter and a random new value to try."""
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
    # If we've tried everything, reset and try again
    _tried.clear()
    return pick_tweak(source)


# ---------------------------------------------------------------------------
# Training cycle
# ---------------------------------------------------------------------------

def parse_train_output(output: str) -> dict:
    """Extract metrics from train.py's final summary."""
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


def run_cycle(coord, cycle_num, mode, best_bpb):
    """Run one train+eval cycle. In search mode, tweak one hyperparameter."""
    run_id = hashlib.md5(f"{time.time()}-{cycle_num}".encode()).hexdigest()[:6]

    original_source = read_train_py()
    tweak_info = None

    if mode == "search":
        param_name, pattern, old_val, new_val = pick_tweak(original_source)
        new_source = apply_tweak(original_source, param_name, pattern, old_val, new_val)
        write_train_py(new_source)
        tweak_info = {"param": param_name, "old": old_val, "new": new_val}
        if param_name == "TOTAL_BATCH_SIZE":
            desc = f"{param_name} 2**{int(old_val)}->2**{int(new_val)} (M4 MPS) [{run_id}]"
        else:
            desc = f"{param_name} {old_val}->{new_val} (M4 MPS) [{run_id}]"
        print(f"[search] Tweaking: {desc}")
    else:
        desc = f"baseline MPS run cycle {cycle_num} (seraph config M4) [{run_id}]"

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

    # Run training
    print("[swarm] Starting train.py...")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__),
    )

    elapsed = time.time() - t0

    if result.stdout:
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            print(line)

    if result.returncode != 0:
        print(f"[swarm] train.py FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-500:])
        # Revert on failure
        if mode == "search":
            print(f"[search] REVERTING {tweak_info['param']} back to {tweak_info['old']}")
            write_train_py(original_source)
        status = "discard"
        val_bpb = None
    else:
        metrics = parse_train_output(result.stdout)
        val_bpb = metrics.get("val_bpb")

        if val_bpb is None:
            print("[swarm] Could not parse val_bpb!")
            if mode == "search":
                write_train_py(original_source)
            return {"status": "parse_error", "elapsed": elapsed}

        # Decide keep or revert
        if mode == "search" and best_bpb is not None and val_bpb >= best_bpb:
            print(f"[search] {val_bpb:.6f} >= best {best_bpb:.6f} — REVERTING")
            write_train_py(original_source)
            status = "discard"
        else:
            if mode == "search":
                print(f"[search] {val_bpb:.6f} < best {best_bpb:.6f} — KEEPING!")
            status = "keep"

        print(f"\n[swarm] {'KEEP' if status == 'keep' else 'DISCARD'} val_bpb = {val_bpb:.6f} in {elapsed:.0f}s")

    # Publish to Ensue
    if exp_key and val_bpb is not None:
        train_source = read_train_py()
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
        # Publish insight
        if tweak_info:
            insight = f"{tweak_info['param']} {tweak_info['old']}->{tweak_info['new']}: "
            insight += f"val_bpb={val_bpb:.6f} ({status}). "
            if status == "keep":
                insight += f"Improved by {best_bpb - val_bpb:.6f} on M4 MPS (33 steps/5min)."
            else:
                insight += f"No improvement on M4 MPS (33 steps/5min)."
            coord.post_insight(insight)

        print(f"[swarm] Published to Ensue: {exp_key}")

    return {
        "status": status if val_bpb is not None else "fail",
        "val_bpb": val_bpb,
        "elapsed": elapsed,
        "tweak": tweak_info,
    }


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

    cycle = 1
    results = []
    best_bpb = None

    try:
        while True:
            if args.cycles > 0 and cycle > args.cycles:
                break

            result = run_cycle(coord, cycle, args.mode, best_bpb)
            results.append(result)

            # Update best
            if result.get("val_bpb") is not None and result["status"] == "keep":
                if best_bpb is None or result["val_bpb"] < best_bpb:
                    best_bpb = result["val_bpb"]

            # Summary
            ok_results = [r for r in results if r.get("val_bpb") is not None]
            keeps = [r for r in results if r["status"] == "keep"]
            discards = [r for r in results if r["status"] == "discard"]
            print(f"\n[swarm] Cycles: {len(results)} | Keeps: {len(keeps)} | Discards: {len(discards)} | Best: {best_bpb:.6f}" if best_bpb else "")

            cycle += 1

            if args.cycles > 0 and cycle > args.cycles:
                break

            print("[swarm] Next cycle in 10s...")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n[swarm] Interrupted. Shutting down.")

    # Final summary
    ok_results = [r for r in results if r.get("val_bpb") is not None]
    keeps = [r for r in ok_results if r["status"] == "keep"]
    print(f"\n{'='*54}")
    print(f"  SWARM SESSION COMPLETE")
    print(f"  Cycles: {len(results)} ({len(keeps)} kept, {len(results) - len(keeps)} discarded)")
    if best_bpb:
        print(f"  Best val_bpb: {best_bpb:.6f}")
    if ok_results:
        print(f"  Total time: {sum(r['elapsed'] for r in results):.0f}s")
    if keeps:
        print(f"  Improvements kept:")
        for r in keeps:
            tw = r.get("tweak")
            if tw:
                print(f"    {tw['param']} {tw['old']}->{tw['new']}: {r['val_bpb']:.6f}")
            else:
                print(f"    baseline: {r['val_bpb']:.6f}")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
