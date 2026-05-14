#!/usr/bin/env bash
# Step 3 of the workflow: aggregate per-profile breakdowns from a sweep
# directory into a cross-profile summary table.
#
# Usage:
#   ./summarize_sweep.sh <sweep_dir> <gpu> <model>
#
# Example:
#   ./summarize_sweep.sh profile/results/4090/decode_scan/gpt-oss-20b 4090 gpt-oss-20b
#   ./summarize_sweep.sh profile/results/h200/prefill_scan/gpt-oss-20b h200 gpt-oss-20b
#
# Reads every `*.nsys-rep` in <sweep_dir> and looks up its
# `out/<gpu>/<model>/<profile_name>/breakdown.json` (and
# theoretical_latency.json when present), then prints one row per swept
# value with the per-category breakdown.
set -euo pipefail

[[ $# -eq 3 ]] || {
    echo "usage: $0 <sweep_dir> <gpu> <model>" >&2
    exit 1
}

sweep_dir="$1"
gpu="$2"
model="$3"
[[ -d "$sweep_dir" ]] || {
    echo "ERROR: sweep dir not found: $sweep_dir" >&2
    exit 1
}

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

uv run python reports/aggregate_scan.py "$sweep_dir" --gpu "$gpu" --model "$model"
