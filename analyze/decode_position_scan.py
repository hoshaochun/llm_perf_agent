"""Scan attention-score latency across decode positions.

Bonus stage of the pipeline (decode mode only).

For each decode position p in {1, 2, 4, 8, 16, ...} we look at the
corresponding decode iter (iter index = prefill_iters_to_skip + p - 1)
and compute:

  - iter_dur_us:   total duration of all kernels in the iter
  - sum_attn_us:   sum of duration of attn_score kernels in the iter
  - attn %:        sum_attn_us / iter_dur_us

This exposes how attention compute scales with KV-cache size as the
decode position grows.

Output: `out/<profile_name>/decode_position_scan.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PREFILL_ITERS_TO_SKIP = 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_name")
    ap.add_argument("--max-pow2", type=int, default=12,
                    help="largest power-of-2 decode position to scan "
                         "(default 12 = position 4096)")
    args = ap.parse_args()

    out_dir = ROOT / "out" / args.profile_name
    canonical = json.loads((out_dir / "canonical.json").read_text())
    seg = json.loads((out_dir / "segmented.json").read_text())
    flow = pd.read_parquet(out_dir / "kernel_flow.parquet").reset_index(drop=True)

    if seg["mode"] != "decode":
        print("# decode_position_scan only applies to decode mode; skipping")
        return 0

    iter_starts = seg["all_iter_starts"]
    n_iters = len(iter_starts)

    attn_kernel_names = {e["name"] for e in canonical["layer_pattern"]
                         if e["label"] == "attn_score"}
    if not attn_kernel_names:
        print("# no attn_score kernels in canonical; skipping")
        return 0
    print(f"# attn_score kernel(s): {sorted(attn_kernel_names)}")

    positions = [1]
    while positions[-1] < (1 << args.max_pow2):
        positions.append(positions[-1] * 2)

    rows = []
    for pos in positions:
        iter_idx = PREFILL_ITERS_TO_SKIP + pos - 1
        if iter_idx >= n_iters:
            break
        iter_start = iter_starts[iter_idx]
        iter_end = (iter_starts[iter_idx + 1] - 1
                    if iter_idx + 1 < n_iters else len(flow) - 1)
        chunk = flow.iloc[iter_start:iter_end + 1]
        iter_dur_us = int(chunk["dur_ns"].sum()) / 1e3
        attn_mask = chunk["short_name"].isin(attn_kernel_names)
        sum_attn_us = int(chunk.loc[attn_mask, "dur_ns"].sum()) / 1e3
        rows.append({
            "pos": pos,
            "iter_idx": iter_idx,
            "iter_dur_us": round(iter_dur_us, 3),
            "sum_attn_us": round(sum_attn_us, 3),
        })

    out_obj = {
        "profile_name": args.profile_name,
        "prefill_iters_to_skip": PREFILL_ITERS_TO_SKIP,
        "attn_kernel_names": sorted(attn_kernel_names),
        "rows": rows,
    }
    out_path = out_dir / "decode_position_scan.json"
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"-> wrote {out_path}")
    print(f"# scanned {len(rows)} decode positions: "
          f"{[r['pos'] for r in rows]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
