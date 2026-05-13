"""Compute roofline (theoretical) latency per operation and compare to actual.

Reads `out/<profile_name>/breakdown.json` (actual durations from the
analysis pipeline) and `out/<profile_name>/segmented.json` (rep-iter
metadata), then uses `theoretical.predictor` to compute the theoretical
bound for each operation under the corresponding workload
(model + GPU + batch size + sequence length).

The output table shows, per labelled operation:
  - actual (ms)         -- summed across N layer instances of the rep iter
  - theor bound (ms)    -- max(compute, memory) from the roofline model
  - compute (ms)        -- compute-bound time at the GPU's peak FLOPS
  - memory (ms)         -- memory-bound time at the GPU's peak bandwidth
  - bound               -- whichever is larger (compute / memory)
  - actual/theor ratio  -- 1.0x = at theoretical peak; > 1 = slower than peak

A final `others` row reports the actual time spent in kernels not
covered by the roofline model (norms, residuals, activations,
MoE-routing helpers, ...) -- it has no theoretical bound.

For decode profiles a second table compares attn_score latency at
each scanned decode position against its theoretical bound (the
predictor is re-evaluated with kv_len set to that position).

Usage:
    uv run python theoretical/compare.py <profile_name> \\
        --model <model-name> [--gpu <gpu-name>]

Workload params (batch_size, input_len, decode_pos) are auto-derived
from the profile_name (`prefill_in<L>`, `decode_bs<B>_out<O>`) and from
`segmented.json` (decode_pos = rep_iter_index); they can be overridden
on the CLI.  The output JSON is written to
`reports/<model>/<profile_name>/theoretical_latency.json`.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.hw_specs import PRESET_GPUS
from configs.model_specs import PRESET_MODELS
from theoretical.predictor import (
    Request,
    estimate_forward_latency,
    expert_latency_cache,
)

# Match the alias table in profile/max_output_len.py so users can pass
# either the short directory name or the full HuggingFace key.
MODEL_NAME_ALIASES = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit":
        "cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
    "Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
}

# Map breakdown-label  ->  predictor operation key.
LABEL_TO_OP = [
    ("qkv_projection",      "qkv"),
    ("attn_score",          "attn_score"),
    ("o_projection",        "o"),
    ("ffn_up_projection",   "up_proj"),
    ("ffn_down_projection", "down_proj"),
    ("lm_head",             "llm_head"),
]
MODELED_LABELS = {label for label, _ in LABEL_TO_OP}


def resolve_model(name: str):
    key = MODEL_NAME_ALIASES.get(name, name)
    if key not in PRESET_MODELS:
        sys.exit(f"unknown model {name!r}; presets={list(PRESET_MODELS)}")
    return PRESET_MODELS[key]


def resolve_gpu(name: str):
    if name not in PRESET_GPUS:
        sys.exit(f"unknown gpu {name!r}; presets={list(PRESET_GPUS)}")
    return PRESET_GPUS[name]


def parse_decode_name(profile_name: str) -> tuple[int, int] | None:
    m = re.match(r"decode_bs(\d+)_out(\d+)$", profile_name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_prefill_name(profile_name: str) -> int | None:
    m = re.match(r"prefill_in(\d+)$", profile_name)
    return int(m.group(1)) if m else None


def build_requests(mode: str, batch_size: int, input_len: int,
                   decode_pos: int) -> list[Request]:
    """Construct the Request list the predictor expects.

    For prefill we have a single request whose entire prompt is being
    processed in one shot.  For decode each of the `batch_size` requests
    has already-prefilled `input_len + decode_pos - 1` tokens in its KV
    cache and is generating a single new token (cur_input_len = 1).
    """
    if mode == "prefill":
        return [Request(
            start_time=0, input_len=input_len, output_len=1, id=0,
            cur_input_len=input_len, kv_len=0,
        )]
    kv_len = input_len + decode_pos - 1
    return [Request(
        start_time=0, input_len=input_len, output_len=decode_pos + 1, id=i,
        cur_input_len=1, kv_len=kv_len,
    ) for i in range(batch_size)]


def compute_theoretical(requests: list[Request], model, gpu) -> dict:
    """Roofline latency per op, summed across all N layers + lm_head."""
    return estimate_forward_latency(requests, None, gpu, model)


def compute_breakdown_rows(actual_us: dict[str, float],
                           theor: dict) -> list[dict]:
    """One row per modeled op + a final `others` row for unmodeled time."""
    rows: list[dict] = []
    for label, opkey in LABEL_TO_OP:
        a_us = float(actual_us.get(label, 0.0))
        op = theor[opkey]
        b_us = op.bound * 1e6
        c_us = op.c * 1e6
        m_us = op.m * 1e6
        bound_kind = "compute" if op.c >= op.m else "memory"
        ratio = (a_us / b_us) if b_us > 0 else float("inf")
        rows.append({
            "label": label,
            "actual_us": a_us,
            "theor_bound_us": b_us,
            "theor_compute_us": c_us,
            "theor_memory_us": m_us,
            "bound_kind": bound_kind,
            "ratio_actual_over_theor": ratio,
        })
    others_actual_us = sum(
        float(v) for k, v in actual_us.items() if k not in MODELED_LABELS
    )
    rows.append({
        "label": "others",
        "actual_us": others_actual_us,
        "theor_bound_us": 0.0,
        "theor_compute_us": 0.0,
        "theor_memory_us": 0.0,
        "bound_kind": None,
        "ratio_actual_over_theor": None,
    })
    return rows


def print_breakdown(rows: list[dict], gpu_name: str, model_name: str) -> None:
    print()
    print("=" * 110)
    print(f" ACTUAL vs THEORETICAL LATENCY  (gpu={gpu_name}, "
          f"model={model_name})")
    print("=" * 110)
    print(f" {'operation':<22}  {'actual (ms)':>11}  {'theor (ms)':>10}  "
          f"{'compute':>9}  {'memory':>9}  {'bound':>7}  "
          f"{'actual/theor':>12}")
    print(" " + "-" * 108)
    for r in rows:
        is_others = r["label"] == "others"
        ratio = r["ratio_actual_over_theor"]
        if is_others or ratio is None:
            ratio_str = f"{'-':>11}"
            theor_str = f"{'-':>10}"
            comp_str = f"{'-':>9}"
            mem_str = f"{'-':>9}"
            bound_str = f"{'-':>7}"
        else:
            ratio_str = f"{ratio:>10.2f}x"
            theor_str = f"{r['theor_bound_us']/1000:>10.3f}"
            comp_str = f"{r['theor_compute_us']/1000:>9.3f}"
            mem_str = f"{r['theor_memory_us']/1000:>9.3f}"
            bound_str = f"{r['bound_kind']:>7}"
        print(f" {r['label']:<22}  "
              f"{r['actual_us']/1000:>11.3f}  "
              f"{theor_str}  {comp_str}  {mem_str}  {bound_str}  "
              f"{ratio_str} ")

    # Totals: actual = full iter (incl. others); theor = sum of modeled only.
    sum_actual_ms = sum(r["actual_us"] for r in rows) / 1000.0
    sum_theor_ms = sum(
        r["theor_bound_us"] for r in rows if r["label"] != "others"
    ) / 1000.0
    print(" " + "-" * 108)
    ratio_total = (sum_actual_ms / sum_theor_ms) if sum_theor_ms > 0 else 0.0
    print(f" {'TOTAL (iter)':<22}  "
          f"{sum_actual_ms:>11.3f}  {sum_theor_ms:>10.3f}  "
          f"{'':>9}  {'':>9}  {'':>7}  "
          f"{ratio_total:>10.2f}x ")


def compute_decode_pos_compare(out_dir: Path, batch_size: int,
                               input_len: int, model, gpu) -> list[dict] | None:
    """Read decode_position_scan.json and compare attn vs theoretical."""
    scan_path = out_dir / "decode_position_scan.json"
    if not scan_path.exists():
        return None
    scan = json.loads(scan_path.read_text())
    out: list[dict] = []
    for r in scan["rows"]:
        pos = int(r["pos"])
        random.seed(0)
        expert_latency_cache.clear()
        reqs = build_requests("decode", batch_size, input_len, pos)
        theor = compute_theoretical(reqs, model, gpu)
        attn = theor["attn_score"]
        attn_actual_us = float(r["sum_attn_us"])
        attn_theor_us = attn.bound * 1e6
        ratio = (attn_actual_us / attn_theor_us) if attn_theor_us > 0 else None
        out.append({
            "pos": pos,
            "kv_len": input_len + pos - 1,
            "iter_actual_us": float(r["iter_dur_us"]),
            "attn_actual_us": attn_actual_us,
            "attn_theor_bound_us": attn_theor_us,
            "attn_theor_compute_us": attn.c * 1e6,
            "attn_theor_memory_us": attn.m * 1e6,
            "bound_kind": "compute" if attn.c >= attn.m else "memory",
            "ratio_actual_over_theor": ratio,
        })
    return out


def print_decode_pos_compare(rows: list[dict]) -> None:
    print()
    print("=" * 110)
    print(" DECODE POSITION SCAN — actual vs theoretical attn_score")
    print("=" * 110)
    print(f" {'pos':>5}  {'kv_len':>7}  {'iter (ms)':>10}  "
          f"{'attn act (ms)':>14}  {'attn theor (ms)':>16}  "
          f"{'compute':>9}  {'memory':>9}  {'bound':>7}  "
          f"{'actual/theor':>12}")
    print(" " + "-" * 108)
    for r in rows:
        ratio = r["ratio_actual_over_theor"]
        ratio_str = f"{ratio:>10.2f}x " if ratio is not None else f"{'-':>11}"
        print(f" {r['pos']:>5}  {r['kv_len']:>7}  "
              f"{r['iter_actual_us']/1000:>10.3f}  "
              f"{r['attn_actual_us']/1000:>14.3f}  "
              f"{r['attn_theor_bound_us']/1000:>16.3f}  "
              f"{r['attn_theor_compute_us']/1000:>9.3f}  "
              f"{r['attn_theor_memory_us']/1000:>9.3f}  "
              f"{r['bound_kind']:>7}  "
              f"{ratio_str}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_name", help="profile name under out/")
    ap.add_argument("--model", required=True,
                    help="preset model name (see configs/model_specs.py)")
    ap.add_argument("--gpu", default="4090",
                    help="preset gpu name (see configs/hw_specs.py)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override batch size (default: derived from profile name)")
    ap.add_argument("--input-len", type=int, default=None,
                    help="prefill: prompt length; decode: prompt length "
                         "before generation (default: 1 for vllm bench decode, "
                         "derived from profile name for prefill)")
    ap.add_argument("--decode-pos", type=int, default=None,
                    help="decode-only: 1-indexed token position to model "
                         "(default: rep_iter_index from segmented.json)")
    args = ap.parse_args()

    out_dir = ROOT / "out" / args.profile_name
    seg_path = out_dir / "segmented.json"
    bd_path = out_dir / "breakdown.json"
    if not seg_path.exists() or not bd_path.exists():
        sys.exit(f"ERROR: missing {seg_path} or {bd_path}; run the analysis "
                 "pipeline first")
    seg = json.loads(seg_path.read_text())
    bd = json.loads(bd_path.read_text())
    mode = seg["mode"]

    # Workload params: CLI > stem-name autoderive > sensible fallback.
    if mode == "prefill":
        derived_input = parse_prefill_name(args.profile_name)
        batch_size = args.batch_size if args.batch_size is not None else 1
        input_len = (args.input_len if args.input_len is not None
                     else (derived_input if derived_input is not None else 1))
        decode_pos = 1
    else:
        derived = parse_decode_name(args.profile_name)
        batch_size = (args.batch_size if args.batch_size is not None
                      else (derived[0] if derived is not None else 1))
        input_len = args.input_len if args.input_len is not None else 1
        decode_pos = (args.decode_pos if args.decode_pos is not None
                      else int(seg["rep_iter_index"]))

    model = resolve_model(args.model)
    gpu = resolve_gpu(args.gpu)

    print(f"# profile_name = {args.profile_name}")
    print(f"# mode         = {mode}")
    print(f"# model        = {model.name}  ({model.n_layers} layers)")
    print(f"# gpu          = {gpu.name}")
    print(f"# batch_size   = {batch_size}")
    print(f"# input_len    = {input_len}")
    if mode == "decode":
        print(f"# decode_pos   = {decode_pos}  "
              f"(kv_len = {input_len + decode_pos - 1})")

    # Stabilise the predictor's MoE expert sampling.
    random.seed(0)
    expert_latency_cache.clear()

    requests = build_requests(mode, batch_size, input_len, decode_pos)
    theor = compute_theoretical(requests, model, gpu)

    rows = compute_breakdown_rows(bd["totals_us"], theor)
    print_breakdown(rows, gpu.name, model.name)

    decode_pos_compare = None
    if mode == "decode":
        decode_pos_compare = compute_decode_pos_compare(
            out_dir, batch_size, input_len, model, gpu
        )
        if decode_pos_compare:
            print_decode_pos_compare(decode_pos_compare)

    # Totals: full iter actual (incl. others) vs sum of modeled theoretical.
    iter_actual_ms = sum(r["actual_us"] for r in rows) / 1000.0
    sum_modeled_theor_ms = sum(
        r["theor_bound_us"] for r in rows if r["label"] != "others"
    ) / 1000.0

    out_obj = {
        "profile_name": args.profile_name,
        "mode": mode,
        "gpu": gpu.name,
        "model": model.name,
        "n_layers": model.n_layers,
        "batch_size": batch_size,
        "input_len": input_len,
        "decode_pos": decode_pos if mode == "decode" else None,
        "rows": rows,
        "iter_actual_ms": iter_actual_ms,
        "sum_modeled_theor_ms": sum_modeled_theor_ms,
        # Back-compat field names (sum of MODELED ops actual/theor only).
        "sum_actual_ms": sum(
            r["actual_us"] for r in rows if r["label"] != "others"
        ) / 1000.0,
        "sum_theor_ms": sum_modeled_theor_ms,
    }
    if decode_pos_compare is not None:
        out_obj["decode_position_scan"] = decode_pos_compare

    # Write to reports/<model>/<profile_name>/theoretical_latency.json.
    # We use the CLI-supplied model name (rather than the alias-resolved
    # HF key) so the path matches whatever the user typed and what
    # `bench_*.sh` writes for the sweep dir; "/" in HF-style keys gets
    # replaced with "_" so the result is a single subdir.
    model_dirname = args.model.replace("/", "_")
    reports_dir = ROOT / "reports" / model_dirname / args.profile_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "theoretical_latency.json"
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"\n-> wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
