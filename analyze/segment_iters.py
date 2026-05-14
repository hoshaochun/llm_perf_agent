"""Segment the trace into per-iter layer-loop + epilogue+prologue.

Stage 3 of the pipeline.

Reads `out/<profile_name>/kernel_flow.parquet` and `out/<profile_name>/canonical.json`,
and segments the full trace using the canonical layer pattern as a
sliding-window template.

For each position p in the trace, we count how many of the next P
kernel names match the canonical pattern position-by-position;
positions where the match fraction >= MATCH_THRESHOLD (default 0.7)
are LAYER STARTS.  This:

  - REJECTS same-name false positives (e.g. the final norm kernel
    which has the same name as the per-layer pre-attn norm but is
    followed by lm_head + sampling, not by qkv).
  - ACCEPTS layer 0 of an iter when its first kernel is fused with
    embedding (different name from pos 0 of the canonical, but the
    remaining P-1 positions still match).

Layer starts spaced exactly P apart belong to the same iter; gaps
larger than P mark iter boundaries.

For each iter we compute:
  layer_loop = trace[layer_loop_start, layer_loop_start + N*P - 1]
  epi+prologue = trace[layer_loop_end + 1, next_iter_layer_loop_start - 1]

We pick ONE rep iter (default: iter 5 for decode skipping prefill, 0
for prefill) and emit:
  - the rep iter's full layer-loop kernels (N*P entries)
  - the rep iter's last-layer kernels (final P entries)
  - the rep iter's epi+prologue kernels (variable length)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MATCH_THRESHOLD = 0.7  # fraction of canonical positions that must match
PREFILL_ITERS_TO_SKIP = 5
EPI_MAX_KERNELS = 400  # cap on emitted epi+prologue (lm_head is near its front)


def find_layer_starts(flow: pd.DataFrame, canonical_names: list[str],
                      threshold: float = MATCH_THRESHOLD) -> np.ndarray:
    """Return trace positions p where trace[p:p+P] matches canonical."""
    P = len(canonical_names)
    n = len(flow)
    short_names = flow["short_name"].to_numpy()
    match_count = np.zeros(n - P + 1, dtype=np.int32)
    for i in range(P):
        match_count += (short_names[i:n - P + 1 + i] ==
                        canonical_names[i]).astype(np.int32)
    min_matches = int(np.ceil(threshold * P))
    return np.where(match_count >= min_matches)[0]


def split_decode_prefill(iters: list[dict],
                         flow: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Separate decode iters from chunked-prefill iters by wall-clock duration.

    vLLM's chunked prefill produces several long iters at the start of the
    trace; the canonical decode layer pattern matches both at >= 70 %
    (kernel names mostly overlap -- only the attention kernel differs),
    so we can't tell them apart by kernel-name match alone.  Iter
    duration, however, is dramatically larger for prefill chunks (many
    tokens per request) than for decode iters (one token per request).

    Strategy: sort iter durations, find the largest absolute gap; if
    that gap is large relative to the shorter cluster's max duration
    (here: > 50 %), treat iters below the gap as decode and the rest
    as prefill.  If no clear cluster boundary exists, leave the list
    alone (returns all iters as decode).
    """
    if len(iters) < 2:
        for it in iters:
            chunk = flow.iloc[it["layer_loop_start"]:it["layer_loop_end"] + 1]
            it["dur_us"] = int(chunk["dur_ns"].sum()) / 1e3
        return iters, []
    for it in iters:
        chunk = flow.iloc[it["layer_loop_start"]:it["layer_loop_end"] + 1]
        it["dur_us"] = int(chunk["dur_ns"].sum()) / 1e3
    sorted_iters = sorted(iters, key=lambda it: it["dur_us"])
    durs = [it["dur_us"] for it in sorted_iters]
    gaps = [durs[i + 1] - durs[i] for i in range(len(durs) - 1)]
    max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    max_gap = gaps[max_gap_idx]
    short_cluster_max = durs[max_gap_idx]
    if short_cluster_max > 0 and max_gap > 0.5 * short_cluster_max:
        short = [it for it in iters if it["dur_us"] <= short_cluster_max]
        long = [it for it in iters if it["dur_us"] > short_cluster_max]
        # Assign the larger cluster to decode (output_len decode iters
        # vs a handful of chunked-prefill iters).  For long input_len
        # the prefill cluster is slower; for short input_len it's
        # faster — but in both cases there are more decode iters.
        if len(short) >= len(long):
            return short, long
        return long, short
    return iters, []


def group_into_iters(matches: np.ndarray, period: int, num_layers: int
                     ) -> list[dict]:
    """Group P-spaced layer matches into per-iter records."""
    if len(matches) == 0:
        return []
    deltas = np.diff(matches)
    breaks = np.where(deltas != period)[0]
    starts = [0, *(int(b) + 1 for b in breaks)]
    ends = [*(int(b) for b in breaks), len(matches) - 1]

    iters: list[dict] = []
    skipped: list[int] = []
    for rs, re_ in zip(starts, ends):
        K = re_ - rs + 1
        first = int(matches[rs])
        last = int(matches[re_])
        if K == num_layers:
            ll_start = first
            n_in_loop = num_layers
        elif K == num_layers - 1:
            # Layer 0 missing (different first kernel name, e.g. embedding-
            # fused norm); try extending backward by one period.
            if first - period >= 0:
                ll_start = first - period
                n_in_loop = num_layers
            else:
                # Trace starts before the canonical's pos 0 has room to
                # appear; accept the iter as N-1 layers (totals will be
                # short by ~1/N).
                ll_start = first
                n_in_loop = num_layers - 1
        else:
            skipped.append(K)
            continue
        ll_end = ll_start + n_in_loop * period - 1
        iters.append({"layer_loop_start": ll_start,
                      "layer_loop_end": ll_end,
                      "first_match": first,
                      "last_match": last,
                      "K": K,
                      "n_in_loop": n_in_loop})
    if skipped:
        print(f"# {len(skipped)} run(s) skipped (K values: "
              f"{skipped[:10]}{'...' if len(skipped) > 10 else ''})")
    return iters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_name")
    ap.add_argument("--gpu", required=True,
                    help="GPU name; reads/writes out/<gpu>/<model>/<profile_name>/")
    ap.add_argument("--model", required=True,
                    help="model name; reads/writes out/<gpu>/<model>/<profile_name>/")
    ap.add_argument("--mode", required=True, choices=("prefill", "decode"))
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--rep-iter", type=int, default=None,
                    help="0-based iter index to use as the rep iter "
                         f"(default: {PREFILL_ITERS_TO_SKIP} for decode, "
                         "0 for prefill)")
    args = ap.parse_args()

    out_dir = ROOT / "out" / args.gpu / args.model.replace("/", "_") / args.profile_name
    flow = pd.read_parquet(out_dir / "kernel_flow.parquet").reset_index(drop=True)
    canonical = json.loads((out_dir / "canonical.json").read_text())
    period = int(canonical["period"])
    canonical_names = [e["name"] for e in canonical["layer_pattern"]]
    print(f"# period P = {period}, num_layers = {args.num_layers}")

    matches = find_layer_starts(flow, canonical_names)
    min_matches = int(np.ceil(MATCH_THRESHOLD * period))
    print(f"# {len(matches)} layer-start matches "
          f"(>= {min_matches}/{period} canonical positions matched)")

    iters = group_into_iters(matches, period, args.num_layers)
    if not iters:
        print("ERROR: no complete iters detected", file=sys.stderr)
        return 1
    K_dist = sorted({it["K"] for it in iters})
    print(f"# {len(iters)} complete iters detected (K values: {K_dist})")

    is_prefill = args.mode == "prefill"

    # In decode mode, filter out vLLM's chunked-prefill iters by latency.
    # See split_decode_prefill() for rationale; prefill layers share most
    # kernel names with decode layers (only attn differs) so the canonical
    # scan in segment_iters can't tell them apart on its own.
    if not is_prefill:
        decode_iters, prefill_iters = split_decode_prefill(iters, flow)
        if prefill_iters:
            max_dec = max(it["dur_us"] for it in decode_iters) / 1000.0
            min_pre = min(it["dur_us"] for it in prefill_iters) / 1000.0
            print(f"# Filtered {len(prefill_iters)} chunked-prefill iter(s) "
                  f"by latency (decode <= {max_dec:.1f} ms, "
                  f"prefill >= {min_pre:.1f} ms)")
        iters = decode_iters
        if not iters:
            print("ERROR: no decode iters left after prefill filter",
                  file=sys.stderr)
            return 1
    if args.rep_iter is not None:
        rep = args.rep_iter
    else:
        rep = 0 if is_prefill else PREFILL_ITERS_TO_SKIP
    if rep >= len(iters):
        print(f"ERROR: requested rep_iter={rep} but only {len(iters)} "
              f"iters detected", file=sys.stderr)
        return 1
    rep_info = iters[rep]
    next_info = iters[rep + 1] if rep + 1 < len(iters) else None

    ll_start = rep_info["layer_loop_start"]
    ll_end = rep_info["layer_loop_end"]
    epi_start = ll_end + 1
    epi_end = (next_info["layer_loop_start"] - 1
               if next_info is not None else len(flow) - 1)
    # The lm_head sits in the first handful of kernels after the layer
    # loop; cap the emitted epi+prologue so find_lm_head's LLM prompt
    # stays bounded even when iters are far apart (e.g. a long trace
    # where only a few decode iters matched the canonical pattern).
    epi_end = min(epi_end, epi_start + EPI_MAX_KERNELS - 1)

    layer_loop = flow.iloc[ll_start:ll_end + 1].reset_index(drop=True)
    epi = flow.iloc[epi_start:epi_end + 1].reset_index(drop=True)
    last_layer = flow.iloc[ll_end + 1 - period:ll_end + 1].reset_index(drop=True)
    t0 = int(layer_loop["start"].iloc[0])

    layer_loop_records = []
    for j, r in enumerate(layer_loop.itertuples()):
        layer_loop_records.append({
            "i": j,
            "layer": j // period,
            "pos": j % period,
            "t_us": round((int(r.start) - t0) / 1e3, 2),
            "dur_us": round(r.dur_ns / 1e3, 3),
            "name": r.short_name,
        })

    last_layer_records = []
    for j, r in enumerate(last_layer.itertuples()):
        last_layer_records.append({
            "i": j,
            "pos": j,
            "t_us": round((int(r.start) - t0) / 1e3, 2),
            "dur_us": round(r.dur_ns / 1e3, 3),
            "name": r.short_name,
        })

    epi_records = []
    for j, r in enumerate(epi.itertuples()):
        epi_records.append({
            "i": j,
            "t_us": round((int(r.start) - t0) / 1e3, 2),
            "dur_us": round(r.dur_ns / 1e3, 3),
            "name": r.short_name,
        })

    out_obj = {
        "profile_name": args.profile_name,
        "mode": args.mode,
        "num_layers": args.num_layers,
        "period": period,
        "n_iters_detected": len(iters),
        "rep_iter_index": rep,
        "rep_iter_K": rep_info["K"],
        "rep_iter_layer_loop_start": int(ll_start),
        "rep_iter_layer_loop_end": int(ll_end),
        "rep_iter_epi_start": int(epi_start),
        "rep_iter_epi_end": int(epi_end),
        "all_iter_starts": [int(it["layer_loop_start"]) for it in iters],
        "layer_loop": layer_loop_records,
        "last_layer": last_layer_records,
        "epi_prologue": epi_records,
    }
    out_path = out_dir / "segmented.json"
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"\n-> wrote {out_path}")
    print(f"# rep iter = {rep}; layer_loop = trace[{ll_start}..{ll_end}] "
          f"({len(layer_loop)} kernels = {args.num_layers} x {period})")
    print(f"# epi+prologue   = trace[{epi_start}..{epi_end}] "
          f"({len(epi)} kernels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
