#!/usr/bin/env bash
# Step 3 of the workflow: compare actual latency breakdown(s) against
# the theoretical roofline (compute / memory bound) per operation, and
# identify the bottleneck.
#
# Usage:
#   ./compare_profile.sh <path> <model> [<gpu>]
#
# <path> can be either:
#   - a single .nsys-rep file   -> compare that one profile
#   - a directory               -> compare every *.nsys-rep in it
#                                  (typically a step-1 sweep dir)
#
# Examples:
#   ./compare_profile.sh profile/results/decode_scan/gpt-oss-20b/decode_bs1_out16384.nsys-rep \
#                        gpt-oss-20b
#   ./compare_profile.sh profile/results/decode_scan/gpt-oss-20b \
#                        gpt-oss-20b 4090
#
# Requires that step 2 (analyze_profile.sh) has already run on each
# profile (the comparison reads breakdown.json + segmented.json).
#
# Writes out/<profile_name>/theoretical_latency.json per profile.
#
# <model> must be a key (or short alias) recognised by
# theoretical/compare.py (see configs/model_specs.py).
# <gpu> defaults to "4090"; see configs/hw_specs.py for presets.
set -uo pipefail

usage() {
    sed -n '3,29p' "$0" >&2
    exit 1
}
[[ $# -ge 2 && $# -le 3 ]] || usage

PATH_ARG="$1"
MODEL="$2"
GPU="${3:-4090}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

shopt -s nullglob
if [[ -f "$PATH_ARG" ]]; then
    profiles=("$PATH_ARG")
elif [[ -d "$PATH_ARG" ]]; then
    profiles=("$PATH_ARG"/*.nsys-rep)
    [[ ${#profiles[@]} -gt 0 ]] || {
        echo "ERROR: no *.nsys-rep in $PATH_ARG" >&2; exit 1; }
else
    echo "ERROR: $PATH_ARG is neither a .nsys-rep file nor a directory" >&2
    exit 1
fi

compare_one() {
    local profile="$1"
    local profile_name out_dir
    profile_name="$(basename "$profile" .nsys-rep)"
    out_dir="out/$profile_name"
    if [[ ! -f "$out_dir/breakdown.json" || ! -f "$out_dir/segmented.json" ]]; then
        echo "WARN: $out_dir is missing breakdown.json / segmented.json;"
        echo "      run analyze_profile.sh first."
        return 1
    fi

    echo
    echo "=== Comparing $profile_name (model=$MODEL, gpu=$GPU) ==="
    uv run python theoretical/compare.py "$profile_name" \
        --model "$MODEL" --gpu "$GPU"
}

n_ok=0
n_fail=0
for profile in "${profiles[@]}"; do
    if compare_one "$profile"; then
        n_ok=$((n_ok + 1))
    else
        n_fail=$((n_fail + 1))
        echo "WARN: compare failed for $profile; continuing" >&2
    fi
done

echo
echo "Step 3 (compare) done. $n_ok succeeded, $n_fail failed (of ${#profiles[@]})."
[[ $n_fail -eq 0 ]]
