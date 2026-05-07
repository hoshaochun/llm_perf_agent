#!/usr/bin/env bash
# Step 3 of the workflow: aggregate per-profile breakdowns from a sweep
# directory into a cross-profile summary table.
#
# Usage:
#   ./summarize_sweep.sh <sweep_dir>
#
# Example:
#   ./summarize_sweep.sh profile/results/decode_scan/gpt-oss-20b
#   ./summarize_sweep.sh profile/results/prefill_scan/gpt-oss-20b
#
# Reads every `*.nsys-rep` in <sweep_dir> and looks up its
# `out/<profile_name>/breakdown.json` (and theoretical_latency.json when
# present), then prints one row per swept value with the per-category
# breakdown.
set -euo pipefail

[[ $# -eq 1 ]] || {
    echo "usage: $0 <sweep_dir>" >&2
    exit 1
}

sweep_dir="$1"
[[ -d "$sweep_dir" ]] || {
    echo "ERROR: sweep dir not found: $sweep_dir" >&2
    exit 1
}

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

uv run python reports/aggregate_scan.py "$sweep_dir"
