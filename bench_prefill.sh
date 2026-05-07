#!/usr/bin/env bash
# Step 1 of the workflow: capture nsys profiles for a prefill sweep.
#
# For each input_len we capture one nsys profile (output_len=1, batch=1).
# Profiles land in profile/results/prefill_scan/<model_name>/.
#
# This script ONLY profiles. Pair it with analyze_profile.sh (step 2) and
# summarize_sweep.sh (step 3), or run all three via run_workflow.sh.
#
# Sweep variable: pass custom values as positional args, e.g.
#   ./bench_prefill.sh 1 2 4 8 16 32 64
#
# Model: override via env var, e.g.
#   MODEL_NAME=Qwen2.5-Coder-7B-Instruct ./bench_prefill.sh
set -uo pipefail

model_home="${MODEL_HOME:-/mnt/llm_team/silicon_mind}"
model_name="${MODEL_NAME:-gpt-oss-20b}"
# Available presets in configs/model_specs.py (and the alias table in
# theoretical/compare.py / profile/max_output_len.py):
#   gpt-oss-20b
#   Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit
#   Qwen2.5-Coder-7B-Instruct

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
