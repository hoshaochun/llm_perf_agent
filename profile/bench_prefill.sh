#!/usr/bin/env bash
# Sweep prefill input lengths -> nsys profile -> latency-breakdown pipeline.
#
# For each input_len we:
#   1. capture an nsys profile (one prefill run, output_len=1);
#   2. run the analysis pipeline (run_pipeline.sh) on the freshly-written
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

out_dir="profile/results/prefill_scan/${model_name}"
mkdir -p "$out_dir"

# Sweep variable: pass custom values as positional args, e.g.
#   ./profile/bench_prefill.sh 1 2 4 8 16 32 64
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

    echo
    echo "=== Analyzing $output_name.nsys-rep ==="
    if ! ./run_pipeline.sh "$output_name.nsys-rep" prefill "$n_layers" \
            "$model_name" "$gpu"; then
        echo "WARN: pipeline failed for $output_name; continuing sweep" >&2
    fi
done

echo
echo "=== Cross-profile summary ==="
uv run python reports/aggregate_scan.py "$out_dir"
