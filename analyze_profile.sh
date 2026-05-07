#!/usr/bin/env bash
# Step 2 of the workflow: analyze the actual latency breakdown of one
# profile or a whole sweep of profiles.
#
# Usage:
#   ./analyze_profile.sh <path> <prefill|decode> <num_layers>
#
# <path> can be either:
#   - a single .nsys-rep file   -> analyze that one profile
#   - a directory               -> analyze every *.nsys-rep in it
#                                  (typically a step-1 sweep dir)
#
# Examples:
#   ./analyze_profile.sh profile/results/examples/decode_example3.nsys-rep decode 28
#   ./analyze_profile.sh profile/results/decode_scan/gpt-oss-20b decode 24
#
# This step ONLY measures the actual workload. Pair with
# compare_profile.sh (step 3) for the theoretical comparison.
#
# Pipeline stages (per profile):
#   1. extract_kernel_flow.py   -- nsys-rep -> kernel_flow.parquet
#   2. find_canonical_layer.py  -- LLM identifies P + canonical layer
#   3. segment_iters.py         -- canonical-template scan -> per-iter
#   4. find_lm_head.py          -- LLM picks lm_head from rep iter
#   5. aggregate_breakdown.py   -- sums per-label durations + lm_head
#   [+] decode_position_scan.py -- (decode) attn vs decode position
#
# Per-profile artefacts written to out/<profile_name>/, full logs in
# out/<profile_name>/pipeline.log.  A failure on one profile is logged
# but does not abort the rest.
set -uo pipefail

usage() {
    sed -n '3,30p' "$0" >&2
    exit 1
}
[[ $# -eq 3 ]] || usage

PATH_ARG="$1"
MODE="$2"
NUM_LAYERS="$3"

[[ "$MODE" == "prefill" || "$MODE" == "decode" ]] || {
    echo "ERROR: mode must be 'prefill' or 'decode'" >&2; exit 1; }
[[ "$NUM_LAYERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: num_layers must be a positive integer" >&2; exit 1; }

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

# Globals consumed by run_stage / analyze_one.
LOG=""

run_stage() {
    local label="$1"; shift
    printf "  %-50s " "$label"
    if "$@" >> "$LOG" 2>&1; then
        echo "OK"
        return 0
    fi
    echo "FAIL"
    return 1
}

print_report() {
    local profile_name="$1"
    local mode="$2"
    uv run python - "$profile_name" <<'PY'
import json, sys
from pathlib import Path

profile_name = sys.argv[1]
out = Path("out") / profile_name
canonical = json.loads((out / "canonical.json").read_text())
seg = json.loads((out / "segmented.json").read_text())
lmh = json.loads((out / "lm_head.json").read_text())
agg = json.loads((out / "breakdown.json").read_text())

P = canonical["period"]
N = seg["num_layers"]
labs = {int(e["pos"]): e for e in canonical["layer_pattern"]}
mean_us = {int(k): v for k, v in agg["per_pos_mean_us"].items()}

print()
print("=" * 78)
print(f" CANONICAL LAYER PATTERN  (one transformer layer, {P} kernels)")
print("=" * 78)
print(f" {'pos':>3}  {'mean (us)':>11}  {'operation':<22}  kernel name")
print(" " + "-" * 76)
for pos in range(P):
    e = labs[pos]
    m = mean_us.get(pos, 0.0)
    print(f" {pos:>3}  {m:>11.3f}  {e['label']:<22}  {e['name']}")

print()
print("=" * 78)
print(" LM HEAD")
print("=" * 78)
print(f" {'pos':>3}  {'dur (us)':>11}  {'operation':<22}  kernel name")
print(" " + "-" * 76)
print(f" {0:>3}  {lmh['lm_head_dur_us']:>11.3f}  "
      f"{'lm_head':<22}  {lmh['lm_head_name']}")

print()
print("=" * 78)
print(f" LATENCY BREAKDOWN  (rep iter = {seg['rep_iter_index']}, "
      f"{N} layers x P={P})")
print("=" * 78)
totals_ms = {k: v / 1000.0 for k, v in agg["totals_us"].items()}
grand_ms = agg["sum_labeled_us"] / 1000.0
groups = [
    ("Attention (qkv + attn + o)", ["qkv_projection", "attn_score", "o_projection"]),
    ("FFN (up + down)",            ["ffn_up_projection", "ffn_down_projection"]),
    ("LM head",                    ["lm_head"]),
]
used = set()
for title, labels in groups:
    sub = sum(totals_ms.get(l, 0.0) for l in labels)
    used.update(labels)
    print(f"\n  {title}: {sub:>10,.3f} ms   ({sub/grand_ms*100:5.1f}% of total)")
    for l in labels:
        v = totals_ms.get(l, 0.0)
        if v > 0:
            print(f"    {l:<28s}  {v:>10,.3f} ms   ({v/grand_ms*100:5.2f}%)")

other_labels = sorted(
    [l for l in totals_ms if l not in used and totals_ms[l] > 0],
    key=lambda l: -totals_ms[l]
)
if other_labels:
    sub = sum(totals_ms[l] for l in other_labels)
    print(f"\n  Other: {sub:>10,.3f} ms   ({sub/grand_ms*100:5.1f}% of total)")
    for l in other_labels:
        v = totals_ms[l]
        print(f"    {l:<28s}  {v:>10,.3f} ms   ({v/grand_ms*100:5.2f}%)")

print(f"\n  {'Total labelled':<30s}  {grand_ms:>10,.3f} ms")
print()
print("=" * 78)
print(f" Artefacts: {out}/")
print(f"   canonical.json              (LLM-identified layer pattern + labels)")
print(f"   segmented.json              (rep iter's layer-loop + epi_prologue)")
print(f"   lm_head.json                (LLM-identified lm_head kernel)")
print(f"   breakdown.json              (latency totals by category)")
PY

    if [[ "$mode" == "decode" ]]; then
        uv run python - "$profile_name" <<'PY'
import json, sys
from pathlib import Path
profile_name = sys.argv[1]
scan = json.loads(
    (Path("out") / profile_name / "decode_position_scan.json").read_text()
)
print()
print("=" * 78)
print(" ATTENTION-SCORE LATENCY vs DECODE POSITION (KV-cache size)")
print("=" * 78)
print(f" {'pos':>5}  {'iter_latency (ms)':>20}  "
      f"{'attn_score_latency (ms)':>25}  {'attn %':>8}")
print(" " + "-" * 76)
for r in scan["rows"]:
    iter_ms = r["iter_dur_us"] / 1000.0
    attn_ms = r["sum_attn_us"] / 1000.0
    pct = (attn_ms / iter_ms * 100) if iter_ms > 0 else 0.0
    print(f" {r['pos']:>5}  {iter_ms:>20.3f}  {attn_ms:>25.3f}  "
          f"{pct:>7.2f}%")
PY
    fi
}

analyze_one() {
    local profile="$1"
    local profile_abs profile_name out_dir
    profile_abs="$(cd "$(dirname "$profile")" && pwd)/$(basename "$profile")"
    profile_name="$(basename "$profile" .nsys-rep)"
    out_dir="out/$profile_name"
    mkdir -p "$out_dir"
    LOG="$out_dir/pipeline.log"
    : > "$LOG"
    # Drop any stale theoretical output from a prior compare_profile run;
    # this run will only repopulate breakdown / canonical / segmented /
    # lm_head, so the theoretical file would otherwise be inconsistent.
    rm -f "$out_dir/theoretical_latency.json"

    echo
    echo "=== Analyzing $profile_name ==="

    run_stage "[1/5] extract kernel flow" \
        uv run python analyze/extract_kernel_flow.py "$profile_abs" || return 1
    run_stage "[2/5] LLM identifies canonical layer pattern" \
        uv run python analyze/find_canonical_layer.py "$profile_name" || return 1
    run_stage "[3/5] segment iters using canonical pattern" \
        uv run python analyze/segment_iters.py "$profile_name" \
            --mode "$MODE" --num-layers "$NUM_LAYERS" || return 1
    run_stage "[4/5] LLM identifies lm_head" \
        uv run python analyze/find_lm_head.py "$profile_name" || return 1
    run_stage "[5/5] aggregate latency breakdown" \
        uv run python analyze/aggregate_breakdown.py "$profile_name" || return 1
    if [[ "$MODE" == "decode" ]]; then
        run_stage "[+] decode-position attention scan" \
            uv run python analyze/decode_position_scan.py "$profile_name" || return 1
    fi

    print_report "$profile_name" "$MODE" || return 1
    return 0
}

n_ok=0
n_fail=0
for profile in "${profiles[@]}"; do
    if analyze_one "$profile"; then
        n_ok=$((n_ok + 1))
    else
        n_fail=$((n_fail + 1))
        echo "WARN: analyze failed for $profile; continuing" >&2
        if [[ -f "$LOG" ]]; then
            echo "Last 40 lines of log:" >&2
            tail -40 "$LOG" >&2
        fi
    fi
done

echo
echo "Step 2 (analyze) done. $n_ok succeeded, $n_fail failed (of ${#profiles[@]})."
[[ $n_fail -eq 0 ]]
