#!/usr/bin/env bash
# Step 1 of the workflow: capture nsys profiles for a prefill sweep.
#
# For each input_len we capture one nsys profile (output_len=1, batch=1).
# Profiles land in profile/results/prefill_scan/<model_name>/.
#
# This script ONLY profiles. Pair it with analyze_profile.sh (step 2) and
# summarize_sweep.sh (step 3), or run all three via run_workflow.sh.
#
# Usage:
#   ./bench_prefill.sh <model_name> [input_len ...]
#
# Examples:
#   ./bench_prefill.sh gpt-oss-20b                       # default sweep (1..8192)
#   ./bench_prefill.sh gpt-oss-20b 1 2 4 8 16 32 64      # custom subset
#   ./bench_prefill.sh Qwen2.5-Coder-7B-Instruct
#
# Model presets live in configs/model_specs.py; the alias table in
# theoretical/compare.py / profile/max_output_len.py maps short names
# (gpt-oss-20b, Qwen2.5-Coder-7B-Instruct, ...) to HuggingFace keys.
# MODEL_HOME (default /mnt/llm_team/silicon_mind) can still be set via env.
set -uo pipefail

usage() {
    sed -n '3,21p' "$0" >&2
    exit 1
}
[[ $# -ge 1 ]] || usage

model_name="$1"; shift
model_home="${MODEL_HOME:-/mnt/llm_team/silicon_mind}"

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

out_dir="profile/results/prefill_scan/${model_name}"
mkdir -p "$out_dir"
echo "Model    : $model_name"
echo "Out dir  : $out_dir"

if [[ $# -gt 0 ]]; then
    input_lens=("$@")
else
    input_lens=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
fi

for input_len in "${input_lens[@]}"; do
    output_name="$out_dir/prefill_in${input_len}"
    echo
    echo "=== Profiling prefill input_len=$input_len ==="
    nsys profile \
        --trace-fork-before-exec=true \
        --capture-range=cudaProfilerApi \
        -o "$output_name" \
        -f true \
        --cuda-graph-trace=node \
        vllm bench latency --model $model_home/$model_name \
            --profiler-config.profiler cuda \
            --num-iters-warmup 1 \
            --num-iters 1 \
            --max-num-batched-tokens 16384 \
            --input-len "$input_len" \
            --output-len 1 \
            --batch-size 1 \
            --profile
done
