"""Identify the lm_head kernel from the rep iter's last layer + epi+prologue.

Stage 4 of the pipeline.

Reads `out/<profile_name>/segmented.json` and sends two slices to the LLM:
  - `last_layer`: the P kernels of the rep iter's FINAL transformer
                  layer (provided as context so the LLM knows what
                  the layer's tail kernels look like).
  - `epi_prologue`: kernels between this iter's layer-loop and the
                    next iter's layer-loop (= final norm + lm_head +
                    sampling + next-iter input prep, lumped together).

The LLM identifies which kernel in `epi_prologue` is the lm_head -- the
single largest GEMM in the epilogue, which projects the last hidden
state to vocabulary logits.

Output: `out/<profile_name>/lm_head.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze.llm import LLMConfig, chat_json


SYSTEM_PROMPT = """You are a GPU performance expert analysing the
EPILOGUE+PROLOGUE region between consecutive transformer iterations of
an LLM inference workload.

You are given:
  - `last_layer`: the kernels of the FINAL transformer layer of an
    iteration (provided as context so you know what a layer's last
    kernels look like, especially the FFN tail).
  - `epi_prologue`: the kernels executed BETWEEN this iter's layer-loop
    and the next iter's layer-loop.  Conceptually this contains, in
    order:
      * iter K's tail / epilogue: final norm -> lm_head -> sampling
        kernels (vocab softmax + random sampling)
      * iter K+1's prologue: input prep (slot mapping, embedding
        lookup, possibly an embedding-fused first norm, possibly
        even a full layer 0 if its pre-norm has a different name).

Your only job: identify which kernel in `epi_prologue` is the LM_HEAD.

The lm_head is the GEMM that projects from hidden_dim -> vocab_size.
It is the SINGLE LARGEST GEMM in `epi_prologue` (typically one or more
orders of magnitude larger duration than per-layer GEMMs, because
vocab_size >> hidden_dim).

Hints:
  * `Marlin`, `Kernel2`, `ampere_*gemm*`, `*_gemm_*` -- any of these
    can be the lm_head; pick the LARGEST-duration GEMM in
    `epi_prologue`.
  * `cunn_SoftMaxForward`, `distribution_elementwise_grid_stride_kernel`
    -- sampling kernels (NOT the lm_head).
  * `_compute_slot_mapping_kernel` -- paged-KV input prep
    (NOT lm_head).
  * `triton_red_fused_*rsqrt*` -- norm (NOT lm_head).
  * `triton_red_fused_*embedding*rsqrt*` -- embedding-fused norm
    (NOT lm_head).
  * `vectorized_elementwise_kernel`, `unrolled_elementwise_kernel`,
    `splitKreduce_kernel` -- residual / reduction (NOT lm_head).

Output STRICT JSON ONLY with this schema:
{
  "lm_head_i": <int -- the `i` field of the chosen kernel in epi_prologue>,
  "lm_head_name": "<exact kernel name>",
  "confidence": <float 0..1>,
  "reasoning": "<2-4 sentences justifying the pick, comparing durations
                of GEMM candidates in epi_prologue>"
}"""


def build_user_prompt(seg: dict) -> str:
    return (
        f"profile_name = {seg['profile_name']!r}, "
        f"mode = {seg['mode']}, "
        f"num_layers = {seg['num_layers']}, "
        f"period = {seg['period']}, "
        f"rep_iter = {seg['rep_iter_index']}\n\n"
        f"## last_layer (rep iter's final transformer layer; "
        f"{len(seg['last_layer'])} kernels)\n"
        + json.dumps(seg["last_layer"], indent=2)
        + f"\n\n## epi_prologue ({len(seg['epi_prologue'])} kernels "
          "between layer-loops)\n"
        + json.dumps(seg["epi_prologue"], indent=2)
        + "\n\nReturn the JSON specified in the system prompt only."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile_name")
    args = ap.parse_args()

    out_dir = ROOT / "out" / args.profile_name
    seg = json.loads((out_dir / "segmented.json").read_text())

    cfg = LLMConfig.from_env()
    if cfg.api_key.startswith("PLACEHOLDER"):
        print("ERROR: LLM_API_KEY is the placeholder.", file=sys.stderr)
        return 2
    print(f"# model = {cfg.model}")
    print(f"# epi_prologue = {len(seg['epi_prologue'])} kernels")

    user_prompt = build_user_prompt(seg)
    result = chat_json(SYSTEM_PROMPT, user_prompt, cfg=cfg, max_tokens=4096)

    epi_by_i = {r["i"]: r for r in seg["epi_prologue"]}
    lm_i = result.get("lm_head_i")
    if lm_i not in epi_by_i:
        print(f"ERROR: bad lm_head_i {lm_i!r} (epi has indices "
              f"{sorted(epi_by_i)})", file=sys.stderr)
        print("Raw response:", file=sys.stderr)
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 1
    truth = epi_by_i[lm_i]
    if result.get("lm_head_name") and result["lm_head_name"] != truth["name"]:
        print(f"# WARN: lm_head_name {result['lm_head_name']!r} != "
              f"epi[{lm_i}].name {truth['name']!r}; using epi entry's name",
              file=sys.stderr)

    out_obj = {
        "profile_name": args.profile_name,
        "lm_head_i": int(lm_i),
        "lm_head_name": truth["name"],
        "lm_head_dur_us": float(truth["dur_us"]),
        "confidence": result.get("confidence"),
        "reasoning": result.get("reasoning"),
    }
    out_path = out_dir / "lm_head.json"
    out_path.write_text(json.dumps(out_obj, indent=2))
    print(f"-> wrote {out_path}")
    print(f"# lm_head: epi[{lm_i}] = {truth['name']!r}  "
          f"dur = {truth['dur_us']:,.3f} us "
          f"({truth['dur_us']/1000:.3f} ms)")
    print(f"# reasoning: {result.get('reasoning', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
