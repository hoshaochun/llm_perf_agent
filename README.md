# llm_perf_agent

A performance-analysis agent that evaluates **how well a model runs on a
given hardware platform**. The workflow is:

1. **Profile** the model under several workload settings (input length,
   batch size, …) → `*.nsys-rep` files.
2. **Analyze** each profile with an LLM-driven pipeline that produces a
   per-operation latency breakdown (Attention / FFN / LM head / Other).
3. **Compare** the actual numbers against a roofline (compute- vs
   memory-bound) theoretical model to identify the bottleneck per
   operation and across the workload sweep.

The analysis is **engine- and model-agnostic**: instead of hard-coding
kernel-name patterns, the pipeline uses an LLM to identify (a) the
canonical transformer-layer kernel sequence from a small slice of the
trace, and (b) the lm_head kernel from the inter-iteration epilogue.
Tested on vLLM traces of dense and MoE models (Qwen2.5-7B,
Qwen3-Coder-30B-MoE, gpt-oss-20b).

## Pipeline

```
                                       ┌── (sweep over a workload variable) ──┐
                                       │                                       │
[profile/]  bench_prefill.sh / bench_decode.sh
                │   nsys profile vllm bench latency …
                ▼
         profile/results/<sweep>/<stem>.nsys-rep
                │
                ▼
[analyze/]  run_pipeline.sh  (one nsys-rep → per-op latency breakdown)
   1. extract_kernel_flow.py   nsys-rep → kernel_flow.parquet
   2. find_canonical_layer.py  LLM identifies P + canonical layer (100-kernel sample)
   3. segment_iters.py         canonical-template scan → per-iter (layer-loop + epi+prologue)
   4. find_lm_head.py          LLM picks lm_head from rep iter's last_layer + epi+prologue
   5. aggregate_breakdown.py   Σ per-label durations across N layers + lm_head
                │
                ▼
         out/<stem>/breakdown.json
                │
                ▼
[theoretical/]  compare.py
                roofline bound (compute vs memory) per operation +
                actual / theoretical ratio → bottleneck
                │
                ▼
[reports/]  aggregate_scan.py
            cross-profile summary across the whole sweep
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

# 3. Analyze one nsys-rep (with optional theoretical comparison).
./run_pipeline.sh path/to/profile.nsys-rep <prefill|decode> <num_layers> \
                  [<model> [<gpu>]]
```

If `<model>` is supplied (a key from `configs/model_specs.py`, or one of
the short aliases handled by `theoretical/compare.py`) the pipeline also
runs the roofline analysis using `<gpu>` (a key from
`configs/hw_specs.py`, default `4090`).

## End-to-end sweep with vLLM

The `profile/` directory ties the pipeline to a vLLM profiling sweep:

```bash
# Sweep prefill across input lengths.
./profile/bench_prefill.sh                       # default sweep (1..8192)
./profile/bench_prefill.sh 1 2 4 8 16 32 64      # custom subset

# Sweep decode across batch sizes (max output_len fitted to VRAM).
./profile/bench_decode.sh
```

Each loop iteration: (a) `nsys profile vllm bench latency …`, (b) the
analysis pipeline on the freshly-written `.nsys-rep`. After the sweep,
`reports/aggregate_scan.py` prints a cross-profile summary table with
per-category latency vs the swept variable.

## Outputs (per profile)

```
out/<stem>/
├── kernel_flow.parquet         # ordered kernel timeline on the dominant stream
├── canonical.json              # canonical layer pattern: P kernels in natural order
├── segmented.json              # rep iter's layer-loop + last-layer + epi+prologue
├── lm_head.json                # LLM-identified lm_head kernel + duration
├── breakdown.json              # per-label totals + per-position means
├── theoretical_latency.json    # (when --model given) actual vs roofline per op
├── decode_position_scan.json   # (decode mode) attn vs decode position
└── pipeline.log                # full stage-by-stage logs
```

## Repository layout

```
configs/                        # shared specs (used by every stage)
  model_specs.py
  hw_specs.py

profile/                        # step 1: capture nsys profiles
  bench_prefill.sh
  bench_decode.sh
  max_output_len.py             # solve max output_len that fits KV cache
  get_n_layers.py
  vram_estimation.py

analyze/                        # step 2: per-profile actual latency breakdown
  llm.py                        # OpenAI-compat LLM client + retries
  extract_kernel_flow.py
  find_canonical_layer.py
  segment_iters.py
  find_lm_head.py
  aggregate_breakdown.py
  decode_position_scan.py       # decode-only attn-vs-position scan

theoretical/                    # step 3: roofline + bottleneck analysis
  predictor.py                  # OperationLatency, matmul/attn/GGEMM models
  run_predictor.py              # standalone multi-request workload simulator
  compare.py                    # actual-vs-theoretical for one breakdown.json

reports/                        # cross-profile summary
  aggregate_scan.py

run_pipeline.sh                 # end-to-end wrapper for one profile
```

## Why two LLM calls?

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
