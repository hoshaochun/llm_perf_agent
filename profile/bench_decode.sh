#!/usr/bin/env bash
# Sweep decode batch sizes -> nsys profile -> latency-breakdown pipeline.
#
# For each batch_size we:
#   1. solve max output_len via max_output_len.py;
#   2. capture an nsys profile (one decode run);
#   3. run the analysis pipeline (run_pipeline.sh) on the freshly-written
#      .nsys-rep to get a per-iter latency breakdown in out/<stem>/.
#
# A pipeline failure on a single profile does NOT abort the sweep -- we
# log the failure and continue, then summarize whatever succeeded at the
# end via reports/aggregate_scan.py.
set -uo pipefail

model_home="/mnt/llm_team/silicon_mind"
model_name="gpt-oss-20b"
# model_name="Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"
# model_name="Qwen2.5-Coder-7B-Instruct"

# GPU preset (matches a key in configs/hw_specs.py:PRESET_GPUS).
gpu="4090"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

n_layers=$(python3 profile/get_n_layers.py "$model_name")
echo "Model    : $model_name"
echo "n_layers : $n_layers"

out_dir="profile/results/decode_scan/${model_name}"
mkdir -p "$out_dir"

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

    echo
    echo "=== Analyzing $output_name.nsys-rep ==="
    if ! ./run_pipeline.sh "$output_name.nsys-rep" decode "$n_layers" \
            "$model_name" "$gpu"; then
        echo "WARN: pipeline failed for $output_name; continuing sweep" >&2
    fi
done

echo
echo "=== Cross-profile summary ==="
uv run python reports/aggregate_scan.py "$out_dir"
