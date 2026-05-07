#!/usr/bin/env bash
# Step 1 of the workflow: capture nsys profiles for a decode sweep.
#
# For each batch_size we (a) solve the largest output_len that fits in
# KV-cache VRAM via max_output_len.py, (b) capture one nsys profile.
# Profiles land in profile/results/decode_scan/<model_name>/.
#
# This script ONLY profiles. Pair it with analyze_profile.sh (step 2) and
# summarize_sweep.sh (step 3), or run all three via run_workflow.sh.
#
# Model: override via env var, e.g.
#   MODEL_NAME=Qwen2.5-Coder-7B-Instruct ./bench_decode.sh
set -uo pipefail

model_home="${MODEL_HOME:-/mnt/llm_team/silicon_mind}"
model_name="${MODEL_NAME:-gpt-oss-20b}"

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

out_dir="profile/results/decode_scan/${model_name}"
mkdir -p "$out_dir"
echo "Model    : $model_name"
echo "Out dir  : $out_dir"

batch_sizes=(1 2 4 8 16 32 64 128 256)
max_output_len=16384

for batch_size in "${batch_sizes[@]}"; do
    output_len=$(python3 profile/max_output_len.py \
        --model "$model_name" \
        --batch-size "$batch_size" \
        --input-len 1 \
        --max-output-len "$max_output_len")

    output_name="$out_dir/decode_bs${batch_size}_out${output_len}"
    echo
    echo "=== Profiling decode batch_size=$batch_size output_len=$output_len ==="
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
            --input-len 1 \
            --output-len "$output_len" \
            --batch-size "$batch_size" \
            --profile
done
