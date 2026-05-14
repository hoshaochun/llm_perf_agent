#!/usr/bin/env bash
# Step 1 of the workflow: capture nsys profiles for a decode sweep.
#
# Sweeps (batch_size x input_len); output_len is fixed at 200.  We measure
# mid-decode latency, so one profile corresponds to one decode position
# (input_len = the KV-cache length the decode step sees).  Profiles land
# in profile/results/decode_scan/<model_name>/decode_bs<B>_in<I>.nsys-rep.
#
# This script ONLY profiles. Pair it with analyze_profile.sh (step 2) and
# summarize_sweep.sh (step 3), or run all three via run_workflow.sh.
#
# Usage:
#   ./bench_decode.sh <model_name> [batch_size ...]
#
# Examples:
#   ./bench_decode.sh gpt-oss-20b                        # default batch sweep (1..128)
#   ./bench_decode.sh gpt-oss-20b 1 2 4 8                # custom batch subset
#   ./bench_decode.sh Qwen2.5-Coder-7B-Instruct
#
# input_lens are hard-coded (1, 2, ..., 8192).  Per-case KV-cache fit
# check uses max_output_len.py against TOTAL_VRAM_GB (default 141, H200);
# cases that wouldn't fit are skipped.  MODEL_HOME
# (default /mnt/llm_team/silicon_mind) is also env-overridable.
set -uo pipefail

usage() {
    sed -n '3,23p' "$0" >&2
    exit 1
}
[[ $# -ge 1 ]] || usage

model_name="$1"; shift
model_home="${MODEL_HOME:-/mnt/llm_team/silicon_mind}"
total_vram_gb="${TOTAL_VRAM_GB:-141}"   # H200 default; override via env

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

out_dir="profile/results/decode_scan/${model_name}"
mkdir -p "$out_dir"
echo "Model    : $model_name"
echo "Out dir  : $out_dir"
echo "VRAM     : $total_vram_gb GB (for KV-cache fit check)"

if [[ $# -gt 0 ]]; then
    batch_sizes=("$@")
else
    batch_sizes=(1 2 4 8 16 32 64 128)
fi
input_lens=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
output_len=200
MAX_ATTEMPTS=3

# Probe whether max_output_len.py recognizes this model.  If yes, we skip
# (bs, input_len) pairs whose KV cache wouldn't fit.  If no (e.g. preset
# not yet added), fall back to letting every case through.
if python3 profile/max_output_len.py \
        --model "$model_name" --batch-size 1 --input-len 1 \
        --max-output-len 1 --total-vram-gb "$total_vram_gb" \
        >/dev/null 2>&1; then
    kv_check_enabled=1
    echo "KV check : enabled"
else
    kv_check_enabled=0
    echo "KV check : disabled ($model_name not recognized by max_output_len.py)"
fi

# Returns 0 iff the model fits weights + KV(bs, input_len + output_len) in
# total_vram_gb (with the same accounting as max_output_len.py).
check_kv_fit() {
    local batch_size="$1" input_len="$2" fit
    fit=$(python3 profile/max_output_len.py \
        --model "$model_name" \
        --batch-size "$batch_size" \
        --input-len "$input_len" \
        --max-output-len "$output_len" \
        --total-vram-gb "$total_vram_gb" 2>/dev/null) || return 1
    [[ "$fit" == "$output_len" ]]
}

# nsys occasionally fails to import its own capture ("Importer error /
# Importation failed / Wrong event order ..."), producing an unusable
# .nsys-rep.  Retry the test case a few times when that happens.
profile_one() {
    local batch_size="$1" input_len="$2" output_name="$3"
    local log_file attempt nsys_status
    log_file=$(mktemp)
    for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
        echo
        if (( attempt == 1 )); then
            echo "=== Profiling decode batch_size=$batch_size input_len=$input_len ==="
        else
            echo "=== Retry $attempt/$MAX_ATTEMPTS for batch_size=$batch_size input_len=$input_len ==="
            rm -f "${output_name}.nsys-rep"
        fi
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
                --output-len "$output_len" \
                --batch-size "$batch_size" \
                --profile 2>&1 | tee "$log_file"
        nsys_status=${PIPESTATUS[0]}
        if (( nsys_status == 0 )) \
            && ! grep -qE "Importer error|Importation failed" "$log_file"; then
            rm -f "$log_file"
            return 0
        fi
        echo "WARN: nsys import/profile failed (attempt $attempt/$MAX_ATTEMPTS)" >&2
    done
    rm -f "$log_file"
    echo "ERROR: gave up on bs=$batch_size in=$input_len after $MAX_ATTEMPTS attempts" >&2
    return 1
}

for batch_size in "${batch_sizes[@]}"; do
    for input_len in "${input_lens[@]}"; do
        if (( kv_check_enabled )) && ! check_kv_fit "$batch_size" "$input_len"; then
            echo
            echo "=== SKIP bs=$batch_size in=$input_len (KV cache won't fit in $total_vram_gb GB) ==="
            continue
        fi
        output_name="$out_dir/decode_bs${batch_size}_in${input_len}"
        profile_one "$batch_size" "$input_len" "$output_name" || true
    done
done
