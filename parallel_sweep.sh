#!/usr/bin/env bash
# Run step 2 (analyze) + step 3 (compare) on a sweep dir, with WORKERS
# profiles processed in parallel.  Each (profile -> analyze -> compare)
# pipeline is one unit of work; up to WORKERS units run concurrently.
# When the pool is full new profiles wait for a slot.  Per-profile stdout
# and stderr land in logs/<gpu>/<mode>_<model>_workers/<profile>.log.
# At the end the summary table is written under reports/.
#
# Usage:
#   ./parallel_sweep.sh <sweep_dir> <prefill|decode> <num_layers> <gpu> <model> [<workers>]
#
# Examples:
#   ./parallel_sweep.sh profile/results/h200/decode_scan/gpt-oss-20b decode 24 h200 gpt-oss-20b 4
#   ./parallel_sweep.sh profile/results/h200/decode_scan/Qwen3-32B decode 64 h200 Qwen3-32B 6
#
# Workers count tradeoffs:
#   - Each unit makes 2 LLM calls in step 2 (canonical layer + lm_head).
#     N workers + M parallel sweeps = up to N*M concurrent LLM calls.
#     Watch your provider's rate limits.
#   - Each unit's nsys export is CPU+disk heavy; very high WORKERS can
#     saturate disk I/O.  4-6 is usually a good starting point.
set -uo pipefail

usage() {
    sed -n '3,21p' "$0" >&2
    exit 1
}
[[ $# -ge 5 && $# -le 6 ]] || usage

SWEEP_DIR="$1"
MODE="$2"
NUM_LAYERS="$3"
GPU="$4"
MODEL="$5"
WORKERS="${6:-4}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

[[ -d "$SWEEP_DIR" ]] || {
    echo "ERROR: $SWEEP_DIR is not a directory" >&2; exit 1; }
[[ "$MODE" == "prefill" || "$MODE" == "decode" ]] || {
    echo "ERROR: mode must be 'prefill' or 'decode'" >&2; exit 1; }
[[ "$NUM_LAYERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: num_layers must be a positive integer" >&2; exit 1; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: workers must be a positive integer" >&2; exit 1; }

MODEL_DIRNAME="${MODEL//\//_}"

shopt -s nullglob
profiles=("$SWEEP_DIR"/*.nsys-rep)
(( ${#profiles[@]} > 0 )) || {
    echo "ERROR: no *.nsys-rep in $SWEEP_DIR" >&2; exit 1; }

logs_dir="logs/$GPU/${MODE}_${MODEL_DIRNAME}_workers"
mkdir -p "$logs_dir"

echo "=========================================================================="
echo " Parallel sweep:  $SWEEP_DIR"
echo " Mode/Layers   :  $MODE / $NUM_LAYERS"
echo " GPU/Model     :  $GPU / $MODEL"
echo " Profiles      :  ${#profiles[@]}"
echo " Workers       :  $WORKERS"
echo " Per-profile logs in $logs_dir/"
echo "=========================================================================="

process_one() {
    local profile="$1"
    local profile_name
    profile_name="$(basename "$profile" .nsys-rep)"
    local log="$logs_dir/$profile_name.log"
    : > "$log"
    if ./analyze_profile.sh "$profile" "$MODE" "$NUM_LAYERS" "$GPU" "$MODEL" \
            >> "$log" 2>&1 \
        && ./compare_profile.sh "$profile" "$MODEL" "$GPU" \
            >> "$log" 2>&1; then
        printf "[%s] OK   %s\n" "$(date +%H:%M:%S)" "$profile_name"
        return 0
    fi
    printf "[%s] FAIL %s  (log: %s)\n" "$(date +%H:%M:%S)" "$profile_name" "$log"
    return 1
}

n_ok=0
n_fail=0
active=0
for profile in "${profiles[@]}"; do
    while (( active >= WORKERS )); do
        if wait -n; then n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); fi
        active=$((active - 1))
    done
    process_one "$profile" &
    active=$((active + 1))
done
while (( active > 0 )); do
    if wait -n; then n_ok=$((n_ok + 1)); else n_fail=$((n_fail + 1)); fi
    active=$((active - 1))
done

echo
echo "Parallel sweep done. $n_ok succeeded, $n_fail failed (of ${#profiles[@]})."

# Cross-profile summary after both steps.
summary_path="reports/$GPU/$MODEL_DIRNAME/${MODE}_summary.txt"
mkdir -p "$(dirname "$summary_path")"
./summarize_sweep.sh "$SWEEP_DIR" "$GPU" "$MODEL" > "$summary_path" 2>&1
echo "Summary -> $summary_path"

[[ $n_fail -eq 0 ]]
