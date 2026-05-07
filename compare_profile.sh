#!/usr/bin/env bash
# Step 3 of the workflow: compare ONE profile's actual latency breakdown
# against the theoretical roofline (compute / memory bound) per
# operation, and identify the bottleneck.
#
# Usage:
#   ./compare_profile.sh <profile.nsys-rep> <model> [<gpu>]
# Example:
#   ./compare_profile.sh profile/results/decode_scan/gpt-oss-20b/decode_bs1_out16384.nsys-rep \
#                        gpt-oss-20b
#   ./compare_profile.sh profile/results/decode_example2.nsys-rep \
#                        Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit 4090
#
# Requires that step 2 (analyze_profile.sh) has already run on this
# profile, since the comparison reads:
#   out/<profile_name>/breakdown.json
#   out/<profile_name>/segmented.json
#
# Writes:
#   out/<profile_name>/theoretical_latency.json
#
# <model> must be a key (or short alias) recognised by
# theoretical/compare.py (see configs/model_specs.py).
# <gpu> defaults to "4090"; see configs/hw_specs.py for presets.
set -euo pipefail

usage() {
    sed -n '3,21p' "$0" >&2
    exit 1
}
[[ $# -ge 2 && $# -le 3 ]] || usage

PROFILE="$1"
MODEL="$2"
GPU="${3:-4090}"

[[ -f "$PROFILE" ]] || { echo "ERROR: profile not found: $PROFILE" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PROFILE_NAME="$(basename "$PROFILE" .nsys-rep)"
OUT_DIR="out/$PROFILE_NAME"
[[ -f "$OUT_DIR/breakdown.json" && -f "$OUT_DIR/segmented.json" ]] || {
    echo "ERROR: $OUT_DIR is missing breakdown.json / segmented.json;" >&2
    echo "       run analyze_profile.sh first." >&2
    exit 1
}

echo "Profile : $PROFILE_NAME"
echo "Model   : $MODEL"
echo "GPU     : $GPU"
echo

uv run python theoretical/compare.py "$PROFILE_NAME" \
    --model "$MODEL" --gpu "$GPU"
