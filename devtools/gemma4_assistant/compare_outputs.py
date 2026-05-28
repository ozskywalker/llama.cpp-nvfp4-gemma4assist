#!/usr/bin/env python3
"""Diff the ggml replay harness outputs (out/*.bin, float32) against hf_ref.npz."""
import argparse, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", default="hf_ref.npz")
ap.add_argument("--out", default="out")
ap.add_argument("--tol", type=float, default=2e-2)
args = ap.parse_args()

d = np.load(args.ref)
names = [f"layer.{i}.out" for i in range(4)] + ["last_hidden_state", "logits"]
nbad = 0
for n in names:
    p = os.path.join(args.out, n + ".bin")
    if not os.path.isfile(p):
        print(f"  {n:22s} MISSING"); nbad += 1; continue
    got = np.fromfile(p, dtype=np.float32)
    exp = d[n].astype(np.float32).reshape(-1)
    if got.size != exp.size:
        print(f"  {n:22s} SIZE {got.size} vs {exp.size}"); nbad += 1; continue
    diff = np.abs(got - exp)
    rel = diff.max() / (np.abs(exp).max() + 1e-9)
    ok = rel < args.tol
    nbad += not ok
    print(f"  {n:22s} maxabs={diff.max():.4e} rel={rel:.2e} {'OK' if ok else 'XX'}")

# also report greedy token agreement
got_l = np.fromfile(os.path.join(args.out, "logits.bin"), dtype=np.float32)
print(f"\nargmax(replay)={int(got_l.argmax())}  argmax(HF)={int(d['logits'][0,-1].argmax())}")
print("ALL MATCH" if nbad == 0 else f"{nbad} MISMATCH(es)")
