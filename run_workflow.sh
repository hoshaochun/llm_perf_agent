#!/usr/bin/env bash
# End-to-end performance-evaluation workflow.  Runs the three steps in
# sequence:
#
#   step 1 (profile)    bench_<mode>.sh
#                       -> profile/results/<mode>_scan/<model>/*.nsys-rep
#   step 2 (analyze)    analyze_profile.sh, looped over each .nsys-rep
#                       -> out/<profile_name>/{breakdown,theoretical_latency,...}.json
#   step 3 (summarize)  summarize_sweep.sh
#                       -> cross-profile latency table
#
# Usage:
#   ./run_workflow.sh <prefill|decode> [<model> [<gpu>]] [-- <bench-args>...]
#
# Examples:
#   ./run_workflow.sh prefill                                     # default model+gpu, full sweep
#   ./run_workflow.sh prefill gpt-oss-20b                         # custom model
#   ./run_workflow.sh prefill gpt-oss-20b 4090                    # custom gpu too
#   ./run_workflow.sh prefill -- 1 2 4 8 16 32 64                 # custom subset of input_lens
#   ./run_workflow.sh prefill gpt-oss-20b 4090 -- 1 2 4 8 16      # everything overridden
#
# A failure in step 2 on a single profile is logged but does not abort
# the workflow -- whatever succeeded is summarised in step 3.
set -uo pipefail

usage() {
    sed -n '3,21p' "$0" >&2
    exit 1
}

[[ $# -ge 1 ]] || usage
MODE="$1"; shift
[[ "$MODE" == "prefill" || "$MODE" == "decode" ]] || {
    echo "ERROR: mode must be 'prefill' or 'decode'" >&2; exit 1; }

# Optional positional args before the `--` separator: model, gpu.
MODEL="${MODEL:-gpt-oss-20b}"
GPU="${GPU:-4090}"
if [[ $# -gt 0 && "$1" != "--" ]]; then MODEL="$1"; shift; fi
if [[ $# -gt 0 && "$1" != "--" ]]; then GPU="$1";   shift; fi
if [[ $# -gt 0 && "$1" == "--" ]];  then shift; fi
# Anything left in $@ is forwarded to the bench script.

repo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_root"

n_layers=$(python3 profile/get_n_layers.py "$MODEL")

echo "=========================================================================="
echo " Workflow : $MODE"
echo " Model    : $MODEL  ($n_layers layers)"
echo " GPU      : $GPU"
echo " Bench args: ${*:-<none>}"
echo "=========================================================================="

# -------- Step 1: profile sweep --------
echo
echo "==== Step 1: profiling ===="
MODEL_NAME="$MODEL" "./bench_${MODE}.sh" "$@"

sweep_dir="profile/results/${MODE}_scan/${MODEL}"
[[ -d "$sweep_dir" ]] || {
    echo "ERROR: expected sweep dir $sweep_dir not found after profiling" >&2
    exit 1
}

# -------- Step 2: per-profile analysis --------
echo
echo "==== Step 2: analyze each profile in $sweep_dir ===="
shopt -s nullglob
profiles=("$sweep_dir"/*.nsys-rep)
if [[ ${#profiles[@]} -eq 0 ]]; then
    echo "ERROR: no .nsys-rep files in $sweep_dir" >&2
    exit 1
fi
for profile_path in "${profiles[@]}"; do
    echo
    echo "--- $(basename "$profile_path") ---"
    if ! ./analyze_profile.sh "$profile_path" "$MODE" "$n_layers" \
            "$MODEL" "$GPU"; then
        echo "WARN: analyze_profile failed for $profile_path; continuing" >&2
    fi
done

# -------- Step 3: cross-profile summary --------
echo
echo "==== Step 3: summarize ===="
./summarize_sweep.sh "$sweep_dir"
