# llm_perf_agent

LLM-driven latency-breakdown tool for LLM-inference workloads. Takes an
Nsight Systems profile (`*.nsys-rep`) of a transformer forward pass and
produces a per-category latency breakdown (Attention / FFN / LM head /
Other) along with a canonical per-layer kernel pattern.

The pipeline is **engine- and model-agnostic**: instead of hard-coding
kernel-name patterns, it asks an LLM to identify (a) the canonical
transformer-layer kernel sequence from a small sample of the trace, and
(b) the lm_head kernel from the inter-iteration epilogue. Tested against
vLLM traces of dense and MoE models (Qwen2.5-7B, Qwen3-Coder-30B-MoE,
gpt-oss-20b).

## Pipeline

```
nsys-rep
    │
    ▼
[1] extract_kernel_flow.py     nsys-rep → kernel_flow.parquet
[2] find_canonical_layer.py    LLM identifies P + canonical layer (100-kernel sample)
[3] segment_iters.py           canonical-template scan → per-iter (layer-loop + epi+prologue)
[4] find_lm_head.py            LLM picks lm_head from rep iter's last_layer + epi+prologue
[5] aggregate_breakdown.py     Σ per-label durations across N layers + lm_head
    │
    ▼
breakdown.json + printed table
```

For decode mode, a bonus stage (`decode_position_scan.py`) sweeps
attention-score latency across decode positions 1, 2, 4, … to expose
how attention compute scales with KV-cache size.

## Quickstart

```bash
# 1. Set your LLM API key (any OpenAI-compatible endpoint works:
#    OpenAI, Anthropic, Gemini, …).  See perf_agent/llm.py for details.
cat > .env <<EOF
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=...
LLM_MODEL=gemini-2.5-pro
EOF

# 2. Install deps.
uv sync

# 3. Analyze one nsys-rep.
./run_pipeline.sh path/to/profile.nsys-rep <prefill|decode> <num_layers>
```

The terminal output shows the canonical layer pattern, the identified
lm_head kernel + duration, and a category breakdown in milliseconds.

## End-to-end with vLLM benchmarks

The `benchmarks/` directory ties the pipeline to a vLLM profiling sweep:

```bash
# Sweep prefill across input lengths and analyze each profile.
./benchmarks/bench_prefill.sh                       # default sweep (1..8192)
./benchmarks/bench_prefill.sh 1 2 4 8 16 32 64      # custom subset

# Sweep decode across batch sizes (max output_len fitted to VRAM).
./benchmarks/bench_decode.sh
```

Each loop iteration: (a) `nsys profile vllm bench latency …`,
(b) the 5-stage pipeline on the freshly-written `.nsys-rep`. After the
sweep, `scripts/aggregate_scan.py` prints a cross-profile summary table
with per-category latency vs the swept variable.

## Outputs (per profile)

```
out/<stem>/
├── kernel_flow.parquet      # ordered kernel timeline on the dominant stream
├── canonical.json           # canonical layer pattern: P kernels in natural order
├── segmented.json           # rep iter's layer-loop + last-layer + epi+prologue
├── lm_head.json             # LLM-identified lm_head kernel + duration
├── breakdown.json           # per-label totals + per-position means
└── pipeline.log             # full stage-by-stage logs
```

## Why two LLM calls?

- Stage 2 sees ~100 consecutive kernels and identifies the per-layer
  template. This is the only place model architecture matters.
- Stage 4 sees the rep iter's last layer + the kernels between
  layer-loops, and picks the lm_head — robust to engines that reuse
  kernel names across operations (e.g. `Kernel2` is used for qkv,
  o_projection, and lm_head in several vLLM builds).

The LLM prompt is provider-agnostic — set `LLM_BASE_URL` to OpenAI,
Anthropic, or Gemini's OpenAI-compatible endpoint.

## Repository layout

```
perf_agent/                  # LLM client package (chat_json + retries)
scripts/                     # 5 pipeline stages + aggregate_scan
  extract_kernel_flow.py
  find_canonical_layer.py
  segment_iters.py
  find_lm_head.py
  aggregate_breakdown.py
  decode_position_scan.py    # decode-only bonus
  aggregate_scan.py          # cross-profile summary
benchmarks/                  # vLLM sweep + helpers
  bench_prefill.sh
  bench_decode.sh
  max_output_len.py
  get_n_layers.py
  model_specs.py
  vram_estimation.py
run_pipeline.sh              # end-to-end wrapper for one profile
```

## Notes / limitations

- Stage 2 samples 100 kernels from the trace center; if your trace is
  shorter than ~200 kernels (e.g. an extremely short prefill) you may
  need `--sample-offset` to land in the layer-loop region.
- Stage 3 segmentation requires the natural-start kernel to appear at
  ≥ 70 % match across layer windows. Models where layer 0 differs from
  layers 1..N-1 in more than one position (rare) may need the threshold
  tuned.
- Stage 4 assumes the lm_head is the largest GEMM in `epi_prologue`;
  this holds for vocab-size ≫ hidden-size models (the common case).
