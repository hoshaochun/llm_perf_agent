"""Aggregate the rep-iter latency breakdown.

Stage 5 of the pipeline.

Reads:
  - canonical.json   (per-position labels for one canonical layer)
  - segmented.json   (rep iter's layer-loop kernels with actual durations)
  - lm_head.json     (the LLM-identified lm_head kernel + its duration)

For each canonical position, sums durations across the N layer instances
of the rep iter (one forward pass).  Adds the lm_head duration.  Reports
Attention, FFN, LM Head, and Other categories.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CATEGORY_GROUPING = {
    "Attention": ["qkv_projection", "attn_score", "o_projection"],
    "FFN":       ["ffn_up_projection", "ffn_down_projection"],
    "LM Head":   ["lm_head"],
}


def aggregate(gpu: str, model: str, profile_name: str) -> dict:
    out_dir = ROOT / "out" / gpu / model.replace("/", "_") / profile_name
    canonical = json.loads((out_dir / "canonical.json").read_text())
    seg = json.loads((out_dir / "segmented.json").read_text())
    lmh = json.loads((out_dir / "lm_head.json").read_text())

    period = int(canonical["period"])
    label_by_pos = {int(e["pos"]): e["label"]
                    for e in canonical["layer_pattern"]}
    name_by_pos = {int(e["pos"]): e["name"]
                   for e in canonical["layer_pattern"]}

    totals_us: dict[str, float] = defaultdict(float)
    per_pos_durs: dict[int, list[float]] = defaultdict(list)

    for r in seg["layer_loop"]:
        pos = int(r["pos"])
        label = label_by_pos[pos]
        totals_us[label] += float(r["dur_us"])
        per_pos_durs[pos].append(float(r["dur_us"]))

    totals_us["lm_head"] += float(lmh["lm_head_dur_us"])

    return {
        "profile_name": profile_name,
        "mode": seg["mode"],
        "rep_iter_index": seg["rep_iter_index"],
        "num_layers": int(seg["num_layers"]),
        "period": period,
        "lm_head_name": lmh["lm_head_name"],
        "lm_head_dur_us": float(lmh["lm_head_dur_us"]),
        "totals_us": dict(totals_us),
        "sum_labeled_us": sum(totals_us.values()),
        "per_pos_mean_us": {p: round(sum(v) / len(v), 3)
                            for p, v in per_pos_durs.items()},
        "per_pos_label": label_by_pos,
        "per_pos_name": name_by_pos,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_name", help="profile name under out/<gpu>/<model>/")
    ap.add_argument("--gpu", required=True,
                    help="GPU name; reads/writes out/<gpu>/<model>/<profile_name>/")
    ap.add_argument("--model", required=True,
                    help="model name; reads/writes out/<gpu>/<model>/<profile_name>/")
    args = ap.parse_args()
    agg = aggregate(args.gpu, args.model, args.profile_name)
    out_path = (ROOT / "out" / args.gpu / args.model.replace("/", "_")
                / args.profile_name / "breakdown.json")
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"-> wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
