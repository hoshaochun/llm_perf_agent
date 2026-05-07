# llm_perf_agent

A performance-analysis agent that evaluates **how well a model runs on a
given hardware platform**. The workflow is three steps:

1. **Profile** the model under a sweep of workload settings (input
   lengths, batch sizes, …) → `*.nsys-rep` files.
2. **Analyze** each profile with an LLM-driven pipeline that produces a
   per-operation latency breakdown of the actual workload (Attention /
   FFN / LM head / Other).
3. **Compare** each per-profile breakdown against a roofline (compute-
   vs memory-bound) theoretical model to identify the bottleneck.

A cross-profile summary utility (`summarize_sweep.sh`) ties the
per-profile outputs back together so you can see how each category
scales with the swept variable.

The analysis is **engine- and model-agnostic**: instead of hard-coding
kernel-name patterns, the pipeline uses an LLM to identify (a) the
canonical transformer-layer kernel sequence from a small slice of the
trace, and (b) the lm_head kernel from the inter-iteration epilogue.
Tested on vLLM traces of dense and MoE models (Qwen2.5-7B,
Qwen3-Coder-30B-MoE, gpt-oss-20b).

## Three step scripts + a summary utility + an orchestrator

```
step 1 (profile)     bench_prefill.sh                 bench_decode.sh
                            │  nsys profile vllm bench latency …
                            ▼
                     profile/results/<sweep>/<profile_name>.nsys-rep

step 2 (analyze)     ./analyze_profile.sh <nsys-rep> <mode> <num_layers>
                            │  extract → canonical (LLM) → segment →
                            │  lm_head (LLM) → aggregate
                            ▼
                     out/<profile_name>/breakdown.json (+ canonical,
                                segmented, lm_head, decode_position_scan)

step 3 (compare)     ./compare_profile.sh <nsys-rep> <model> [<gpu>]
                            │  build Request from workload args →
                            │  per-op roofline → ratio + bottleneck
                            ▼
                     out/<profile_name>/theoretical_latency.json

summary (utility)    ./summarize_sweep.sh <sweep_dir>
                            ▼
                     cross-profile latency table

orchestrator         ./run_workflow.sh <prefill|decode> [<model> [<gpu>]] \
                                       [-- <bench-args>...]
                     runs step 1 → loop step 2 → loop step 3 → summary
```

## Quickstart

```bash
# 1. Set your LLM API key (any OpenAI-compatible endpoint: OpenAI,
#    Anthropic, Gemini, ...).  See analyze/llm.py for details.
cat > .env <<EOF
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=...
LLM_MODEL=gemini-2.5-pro
EOF

# 2. Install deps.
uv sync

# 3a. End-to-end (sweep → analyze → compare → summary).
./run_workflow.sh prefill                                # default model+gpu, full sweep
./run_workflow.sh decode  gpt-oss-20b 4090               # explicit model+gpu
./run_workflow.sh prefill -- 1 2 4 8 16 32 64            # custom input_lens
./run_workflow.sh prefill gpt-oss-20b 4090 -- 1 2 4 8

# 3b. Or run the steps individually.
./bench_prefill.sh gpt-oss-20b 1 2 4 8                   # step 1 only
./analyze_profile.sh path/to/foo.nsys-rep prefill 24     # step 2 (one profile)
./compare_profile.sh path/to/foo.nsys-rep gpt-oss-20b    # step 3 (one profile)
./summarize_sweep.sh profile/results/prefill_scan/gpt-oss-20b/  # cross-profile summary
```

`<model>` for step 3 is a key from `configs/model_specs.py` (or one of
the short aliases handled by `theoretical/compare.py`); `<gpu>` is a key
from `configs/hw_specs.py` (default `4090`).

A failure on any single profile during step 2 or step 3 is logged and
the orchestrator continues — partial sweeps still get a final summary
over whatever succeeded.

## Outputs (per profile, under `out/<profile_name>/`)

```
out/<profile_name>/
├── kernel_flow.parquet         # ordered kernel timeline on the dominant stream
├── canonical.json              # canonical layer pattern: P kernels in natural order
├── segmented.json              # rep iter's layer-loop + last-layer + epi+prologue
├── lm_head.json                # LLM-identified lm_head kernel + duration
├── breakdown.json              # per-label totals + per-position means       [step 2]
├── decode_position_scan.json   # (decode mode) attn vs decode position       [step 2]
├── theoretical_latency.json    # actual vs roofline per op + ratio           [step 3]
└── pipeline.log                # full stage-by-stage logs
```

## Repository layout

```
configs/                        # shared specs (used by step 3 and step 1 helpers)
  model_specs.py
  hw_specs.py

profile/                        # step 1 helpers + outputs
  max_output_len.py             # solve max output_len that fits KV cache
  get_n_layers.py
  vram_estimation.py
  results/                      # gitignored; bench scripts write here

analyze/                        # step 2 internals: actual latency breakdown
  llm.py                        # OpenAI-compat LLM client + retries
  extract_kernel_flow.py
  find_canonical_layer.py
  segment_iters.py
  find_lm_head.py
  aggregate_breakdown.py
  decode_position_scan.py       # decode-only attn-vs-position scan

theoretical/                    # step 3 internals: roofline + bottleneck
  predictor.py                  # OperationLatency, matmul/attn/GGEMM models
  compare.py                    # actual-vs-theoretical for one breakdown.json

reports/                        # cross-profile summary internals
  aggregate_scan.py

bench_prefill.sh                # step 1 driver (prefill sweep)
bench_decode.sh                 # step 1 driver (decode sweep)
analyze_profile.sh              # step 2 driver (one profile)
compare_profile.sh              # step 3 driver (one profile)
summarize_sweep.sh              # cross-profile summary utility
run_workflow.sh                 # end-to-end orchestrator (steps 1+2+3 + summary)
```

## Why two LLM calls inside step 2?

- `find_canonical_layer.py` sees ~100 consecutive kernels and identifies
  the per-layer template. This is the only place model architecture
  matters.
- `find_lm_head.py` sees the rep iter's last layer + the kernels between
  layer-loops, and picks the lm_head — robust to engines that reuse
  kernel names across operations (e.g. `Kernel2` is used for qkv,
  o_projection, and lm_head in several vLLM builds).

The LLM prompt is provider-agnostic — set `LLM_BASE_URL` to OpenAI,
Anthropic, or Gemini's OpenAI-compatible endpoint.

## Notes / limitations

- `find_canonical_layer.py` samples 100 kernels from the trace center;
  if your trace is shorter than ~200 kernels (e.g. an extremely short
  prefill) you may need `--sample-offset` to land in the layer-loop
  region.
- `segment_iters.py` requires the natural-start kernel to appear at
  ≥ 70 % match across layer windows. Models where layer 0 differs from
  layers 1..N-1 in more than one position (rare) may need the threshold
  tuned.
- `find_lm_head.py` assumes the lm_head is the largest GEMM in
  `epi_prologue`; this holds for vocab-size ≫ hidden-size models
  (the common case).
- `theoretical/compare.py` uses peak GPU FLOPS / bandwidth from
  `configs/hw_specs.py`. For sharper roofline estimates, plug
  microbenchmark data via the `bench_data` argument in `predictor.py`.
