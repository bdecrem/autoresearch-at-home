#!/usr/bin/env python3
"""
Simple swarm loop: train → eval → publish → repeat.

Usage:
    uv run python run_swarm.py [--cycles N] [--agent-id NAME]

Each cycle:
  1. Claim a "baseline MPS run" experiment on Ensue
  2. Run train.py (captures output for val_bpb)
  3. Publish result to the swarm
  4. Repeat
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

from coordinator import Coordinator

TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train.py")


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


def run_cycle(coord: Coordinator, cycle_num: int) -> dict:
    """Run one train+eval cycle and publish results."""
    import hashlib
    run_id = hashlib.md5(f"{time.time()}-{cycle_num}".encode()).hexdigest()[:6]
    desc = f"baseline MPS run cycle {cycle_num} (float32 batch8 M4) [{run_id}]"
    
    print(f"\n{'='*54}")
    print(f"  CYCLE {cycle_num}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*54}\n")
    
    # Claim experiment
    exp_key = coord.claim_experiment(desc)
    if not exp_key:
        # If claim fails (duplicate), just use a timestamped key
        print("[swarm] Claim failed (possibly duplicate), running anyway...")
        desc = f"baseline MPS run cycle {cycle_num} t={int(time.time())}"
        exp_key = coord.claim_experiment(desc)
        if not exp_key:
            print("[swarm] Still can't claim, skipping publish but running anyway")
    
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
    
    # Print train output live-ish
    if result.stdout:
        # Just print the last 20 lines (summary)
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            print(line)
    
    if result.returncode != 0:
        print(f"[swarm] train.py FAILED (exit {result.returncode})")
        if result.stderr:
            print(result.stderr[-500:])
        return {"status": "fail", "elapsed": elapsed}
    
    # Parse results
    metrics = parse_train_output(result.stdout)
    val_bpb = metrics.get("val_bpb")
    
    if val_bpb is None:
        print("[swarm] Could not parse val_bpb from output!")
        return {"status": "parse_error", "elapsed": elapsed}
    
    print(f"\n[swarm] ✅ val_bpb = {val_bpb:.6f} in {elapsed:.0f}s")
    
    # Publish to Ensue
    if exp_key:
        train_source = open(TRAIN_SCRIPT).read()
        coord.publish_result(
            experiment_key=exp_key,
            val_bpb=val_bpb,
            memory_gb=metrics.get("peak_vram_mb", 0) / 1024,
            status="keep",
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
            },
        )
        print(f"[swarm] Published to Ensue: {exp_key}")
    
    return {"status": "ok", "val_bpb": val_bpb, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Autoresearch swarm loop")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0 = infinite)")
    parser.add_argument("--agent-id", default="mave-m4", help="Agent identifier")
    args = parser.parse_args()
    
    coord = Coordinator()
    coord.agent_id = args.agent_id
    
    # Test connectivity
    if not coord.test_connectivity():
        print("ERROR: Cannot connect to Ensue. Check .autoresearch-key")
        sys.exit(1)
    
    coord.announce()
    
    cycle = 1
    results = []
    
    try:
        while True:
            if args.cycles > 0 and cycle > args.cycles:
                break
            
            result = run_cycle(coord, cycle)
            results.append(result)
            
            # Summary so far
            ok_results = [r for r in results if r["status"] == "ok"]
            if ok_results:
                bpbs = [r["val_bpb"] for r in ok_results]
                print(f"\n[swarm] Cycles: {len(results)} | Best: {min(bpbs):.6f} | Avg: {sum(bpbs)/len(bpbs):.6f}")
            
            cycle += 1
            
            if args.cycles > 0 and cycle > args.cycles:
                break
            
            # Brief pause between cycles
            print("[swarm] Next cycle in 10s...")
            time.sleep(10)
    
    except KeyboardInterrupt:
        print("\n[swarm] Interrupted. Shutting down.")
    
    # Final summary
    ok_results = [r for r in results if r["status"] == "ok"]
    print(f"\n{'='*54}")
    print(f"  SWARM SESSION COMPLETE")
    print(f"  Cycles: {len(results)} ({len(ok_results)} successful)")
    if ok_results:
        bpbs = [r["val_bpb"] for r in ok_results]
        print(f"  Best val_bpb: {min(bpbs):.6f}")
        print(f"  Total time: {sum(r['elapsed'] for r in results):.0f}s")
    print(f"{'='*54}")


if __name__ == "__main__":
    main()
