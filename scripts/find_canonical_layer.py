"""Identify the canonical transformer layer pattern via the LLM.

Stage 2 of the pipeline.

Reads `out/<stem>/kernel_flow.parquet` and sends ~100 consecutive kernels
from a steady-state region of the trace (default: trace[1000:1100], or
trace[len/4:len/4+100] for shorter traces) to the LLM.  The LLM returns:

  * the per-layer kernel period P;
  * the canonical layer pattern: P kernel templates in NATURAL ORDER --
    the layer STARTS at the pre-attention norm (pos 0) and ends with
    the last residual / FFN-down kernel before the next layer's
    pre-attn norm.

Output: `out/<stem>/canonical.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perf_agent.llm import LLMConfig, chat_json

ALLOWED_LABELS = [
    "qkv_projection", "attn_score", "o_projection",
    "ffn_up_projection", "ffn_down_projection",
    "moe_router", "moe_routing_aux",
    "norm", "rope_or_qkv_split", "kv_cache_write", "activation",
    "residual_or_misc", "lm_head", "sampling", "embedding",
    "input_prep", "unknown",
]


SYSTEM_PROMPT = f"""You are a GPU performance expert analysing kernel
traces from an LLM inference workload (captured via Nsight Systems).

You will be given ~100 CONSECUTIVE kernels from somewhere in the steady-
state region of a forward-pass trace.  These kernels span several
transformer layers (the same P-kernel pattern repeats one layer after
another).  Your job:

  1. Identify the LAYER PERIOD P -- the number of kernels per
     transformer layer.
  2. Output the CANONICAL LAYER PATTERN: the P kernel templates that
     constitute one transformer layer, in NATURAL ORDER -- the layer
     STARTS at the PRE-ATTENTION NORM (pos 0) and ends with the last
     kernel before the NEXT layer's pre-attention norm.

A pre-norm transformer layer (the modern standard) goes:

    pre-attn norm -> qkv projection -> rope / qkv split -> kv-cache
        write -> attention score -> o projection -> post-attn norm
        -> FFN
where FFN is:
    dense:   ffn_up -> activation -> ffn_down (+ residual reduction)
    MoE:     router-gate -> routing helpers -> ffn_up -> activation
             -> ffn_down -> moe_sum

The layer ENDS with the last residual / reduction kernel right before
the next layer's pre-attn norm.

Hints to recognize common kernels (they vary by engine and model):
  * `Marlin` -- quantized (AWQ/INT4) GEMM (vLLM MoE up/down).
  * `Kernel2` -- templated cuBLAS GEMM (qkv, o, router-gate, lm_head).
  * `ampere_*gemm*` / `*_gemm_*` -- bf16/fp16 GEMMs.
  * `kernel_unified_attention_*`, `flash_fwd_*`, `paged_attention_*`
    -- attention-score kernel.
  * `reshape_and_cache_*kernel*flash*` -- KV-cache writer.
  * `swigluoai_and_mul_kernel`, `act_and_mul_kernel`, `silu_and_mul_*`,
    `triton_poi_fused_mul_silu_*` -- SwiGLU/SiLU activation.
  * `topkGating`, `moe_align_block_size_*` -- MoE routing helpers.
  * `triton_red_fused_*rsqrt*` -- RMSNorm.
  * `triton_poi_fused_*` -- shape ops (rope/qkv split, residuals).
  * `splitKreduce_kernel`, `reduce_kernel`, `elementwise_kernel`,
    `vectorized_elementwise_kernel` -- residual/reduction.

Allowed labels (use EXACTLY one per position):
{ALLOWED_LABELS}

Output STRICT JSON ONLY with this schema:
{{
  "period": <int -- the per-layer kernel count P>,
  "layer_pattern": [
    {{"pos": <int 0..P-1>, "name": "<exact kernel name>",
      "label": "<one of the allowed labels>",
      "confidence": <float 0..1>}}
  ],
  "reasoning": "<3-5 sentences justifying the chosen P and the
                pre-attn-norm boundary>"
}}

Rules:
- `layer_pattern` MUST contain exactly P entries with pos = 0..P-1.
- Use EXACT kernel name strings from the input (do not paraphrase).
- pos 0 MUST be the pre-attention norm (label = "norm").
- pos 1 should be `qkv_projection` (or `rope_or_qkv_split` if there's
  a separate split kernel before qkv -- but pos 1 = qkv is far more
  common)."""


def build_user_prompt(sample: list[dict], total_kernels: int,
                      sample_offset: int) -> str:
    return (
        f"INPUT: {len(sample)} consecutive kernels from a trace of "
        f"{total_kernels:,} total kernels (sample starts at trace "
        f"index {sample_offset}; durations in microseconds).\n\n"
        + json.dumps(sample, indent=2)
        + "\n\nReturn the JSON specified in the system prompt only."
    )


def validate(result: dict) -> list[str]:
    errs: list[str] = []
    period = result.get("period")
    pat = result.get("layer_pattern") or []
    if not isinstance(period, int) or period < 3:
        errs.append(f"bad period: {period!r}")
        return errs
    if len(pat) != period:
        errs.append(f"layer_pattern has {len(pat)} entries, expected "
                    f"{period}")
    seen_pos: set[int] = set()
    for entry in pat:
        p = entry.get("pos")
        if not isinstance(p, int) or p < 0 or p >= period:
            errs.append(f"bad pos: {p!r}")
            continue
        if p in seen_pos:
            errs.append(f"duplicate pos: {p}")
        seen_pos.add(p)
        if entry.get("label") not in ALLOWED_LABELS:
            errs.append(f"bad label at pos={p}: {entry.get('label')!r}")
    by_pos = {e.get("pos"): e for e in pat if isinstance(e.get("pos"), int)}
    if 0 in by_pos and by_pos[0].get("label") != "norm":
        errs.append(f"pos 0 label must be 'norm', got "
                    f"{by_pos[0].get('label')!r}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stem", help="profile stem under out/")
    ap.add_argument("--sample-offset", type=int, default=None,
                    help="trace index where the 100-kernel sample starts "
                         "(default: centered on the trace)")
    ap.add_argument("--sample-size", type=int, default=100,
                    help="number of consecutive kernels to sample "
                         "(default 100)")
    args = ap.parse_args()

    out_dir = ROOT / "out" / args.stem
    flow_path = out_dir / "kernel_flow.parquet"
    if not flow_path.exists():
        print(f"ERROR: {flow_path} not found", file=sys.stderr)
        return 1
    flow = pd.read_parquet(flow_path).reset_index(drop=True)
    n = len(flow)
    print(f"# loaded {n:,} kernels from {flow_path}")

    if args.sample_offset is not None:
        offset = args.sample_offset
    else:
        # Center the sample on the trace so it lands in the steady-state
        # layer-loop region rather than the prologue or epilogue.
        offset = max(0, (n - args.sample_size) // 2)
    end = min(n, offset + args.sample_size)
    sample_df = flow.iloc[offset:end]
    sample = []
    for j, r in enumerate(sample_df.itertuples()):
        sample.append({
            "i": j,
            "name": r.short_name,
            "dur_us": round(r.dur_ns / 1e3, 3),
        })
    print(f"# sample: trace[{offset}:{end}] ({len(sample)} kernels)")

    cfg = LLMConfig.from_env()
    if cfg.api_key.startswith("PLACEHOLDER"):
        print("ERROR: LLM_API_KEY is the placeholder.", file=sys.stderr)
        return 2
    print(f"# model = {cfg.model}")

    user_prompt = build_user_prompt(sample, n, offset)
    result = chat_json(SYSTEM_PROMPT, user_prompt, cfg=cfg, max_tokens=16384)

    errs = validate(result)
    if errs:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        print("\nRaw response:", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 1

    out_obj = {
        "profile_stem": args.stem,
        "sample_offset": offset,
        "sample_size": len(sample),
        "period": int(result["period"]),
        "layer_pattern": sorted(result["layer_pattern"],
                                key=lambda x: x["pos"]),
        "reasoning": result.get("reasoning"),
    }
    out_path = out_dir / "canonical.json"
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"-> wrote {out_path}")

    print(f"\n# Canonical layer pattern (period P = {out_obj['period']}, "
          f"pos 0 = pre-attn norm):")
    for entry in out_obj["layer_pattern"]:
        print(f"  pos {entry['pos']:>2}  {entry['label']:22s}  "
              f"{entry['name']}")
    print(f"\n# Reasoning: {result.get('reasoning', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
