#!/usr/bin/env bash
# Step 2 of the workflow: analyze ONE Nsight Systems profile -> per-layer
# kernel labels + latency breakdown (+ optional theoretical comparison).
#
# Usage:
#   ./analyze_profile.sh <profile.nsys-rep> <prefill|decode> <num_layers> \
#                        [<model> [<gpu>]]
# Example:
#   ./analyze_profile.sh profile/results/decode_example3.nsys-rep decode 28
#   ./analyze_profile.sh profile/results/decode_scan/gpt-oss-20b/decode_bs1_out16384.nsys-rep \
#                        decode 24 gpt-oss-20b 4090
#
# Pipeline stages:
#   1. extract_kernel_flow.py    -- nsys-rep -> kernel_flow.parquet
#   2. find_canonical_layer.py   -- LLM identifies P + canonical layer pattern
#                                   from a 100-kernel sample
#   3. segment_iters.py          -- canonical-template scan splits the trace
#                                   into per-iter (layer-loop + epi+prologue);
#                                   emits one rep iter
#   4. find_lm_head.py           -- LLM picks lm_head from rep iter's
#                                   last_layer + epi+prologue
#   5. aggregate_breakdown.py    -- sums per-label durations + lm_head
#   [+] theoretical/compare.py   -- (when <model> given) actual vs roofline
#                                   per operation
#   [+] decode_position_scan.py  -- (decode mode) attn vs decode position
#
# Final output (printed to stdout):
#   1. canonical layer pattern (per-position labels + mean duration)
#   2. lm_head kernel + duration
#   3. latency breakdown grouped into Attention / FFN / LM head / Other
#   4. (when <model> given) actual vs theoretical roofline table
#
# Intermediate artefacts written to out/<profile_name>/ ; full logs in
# out/<profile_name>/pipeline.log.
set -euo pipefail

usage() {
    sed -n '3,28p' "$0" >&2
    exit 1
}
[[ $# -ge 3 && $# -le 5 ]] || usage

PROFILE="$1"
MODE="$2"
NUM_LAYERS="$3"
MODEL="${4:-}"
GPU="${5:-4090}"

[[ -f "$PROFILE" ]] || { echo "ERROR: profile not found: $PROFILE" >&2; exit 1; }
[[ "$MODE" == "prefill" || "$MODE" == "decode" ]] || {
    echo "ERROR: mode must be 'prefill' or 'decode'" >&2; exit 1; }
[[ "$NUM_LAYERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: num_layers must be a positive integer" >&2; exit 1; }

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Resolve absolute path so extract_kernel_flow.py can find it.
PROFILE_ABS="$(cd "$(dirname "$PROFILE")" && pwd)/$(basename "$PROFILE")"
PROFILE_NAME="$(basename "$PROFILE" .nsys-rep)"
OUT_DIR="out/$PROFILE_NAME"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/pipeline.log"
: > "$LOG"

run_stage() {
    local label="$1"; shift
    printf "  %-50s " "$label"
    if "$@" >> "$LOG" 2>&1; then
        echo "OK"
    else
        echo "FAIL"
        echo
        echo "Pipeline failed. Tail of log:"
        tail -40 "$LOG" >&2
        exit 1
    fi
}

echo "Profile : $PROFILE_ABS"
echo "Mode    : $MODE"
echo "Layers  : $NUM_LAYERS"
echo

run_stage "[1/5] extract kernel flow" \
    uv run python analyze/extract_kernel_flow.py "$PROFILE_ABS"
run_stage "[2/5] LLM identifies canonical layer pattern" \
    uv run python analyze/find_canonical_layer.py "$PROFILE_NAME"
run_stage "[3/5] segment iters using canonical pattern" \
    uv run python analyze/segment_iters.py "$PROFILE_NAME" \
        --mode "$MODE" --num-layers "$NUM_LAYERS"
run_stage "[4/5] LLM identifies lm_head" \
    uv run python analyze/find_lm_head.py "$PROFILE_NAME"
run_stage "[5/5] aggregate latency breakdown" \
    uv run python analyze/aggregate_breakdown.py "$PROFILE_NAME"

if [[ -n "$MODEL" ]]; then
    run_stage "[+] theoretical roofline analysis" \
        uv run python theoretical/compare.py "$PROFILE_NAME" \
            --model "$MODEL" --gpu "$GPU"
fi

if [[ "$MODE" == "decode" ]]; then
    run_stage "[+] decode-position attention scan" \
        uv run python analyze/decode_position_scan.py "$PROFILE_NAME"
fi

# ----------------------- final structured output ----------------------------
uv run python - <<PY
import json
from pathlib import Path

profile_name = "$PROFILE_NAME"
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

theor_path = out / "theoretical_latency.json"
if theor_path.exists():
    th = json.loads(theor_path.read_text())
    print()
    print("=" * 102)
    print(f" ACTUAL vs THEORETICAL  (gpu={th['gpu']}, model={th['model']}"
          + (f", batch={th['batch_size']}, input_len={th['input_len']}"
             + (f", decode_pos={th['decode_pos']}" if th.get('decode_pos') else "")
             ) + ")")
    print("=" * 102)
    print(f" {'operation':<22}  {'actual (ms)':>11}  {'theor (ms)':>10}  "
          f"{'compute':>9}  {'memory':>9}  {'bound':>7}  {'a/t':>7}")
    print(" " + "-" * 100)
    for r in th["rows"]:
        print(f" {r['label']:<22}  "
              f"{r['actual_us']/1000:>11.3f}  "
              f"{r['theor_bound_us']/1000:>10.3f}  "
              f"{r['theor_compute_us']/1000:>9.3f}  "
              f"{r['theor_memory_us']/1000:>9.3f}  "
              f"{r['bound_kind']:>7}  "
              f"{r['ratio_actual_over_theor']:>6.2f}x")
    print(" " + "-" * 100)
    sa = th['sum_actual_ms']; st = th['sum_theor_ms']
    print(f" {'TOTAL (these ops)':<22}  "
          f"{sa:>11.3f}  {st:>10.3f}  {'':>9}  {'':>9}  {'':>7}  "
          f"{(sa/st if st>0 else 0):>6.2f}x")

print()
print("=" * 78)
print(f" Artefacts: {out}/")
print(f"   canonical.json              (LLM-identified layer pattern + labels)")
print(f"   segmented.json              (rep iter's layer-loop + epi_prologue)")
print(f"   lm_head.json                (LLM-identified lm_head kernel)")
print(f"   breakdown.json              (latency totals by category)")
if theor_path.exists():
    print(f"   theoretical_latency.json   (actual vs roofline per op)")
PY

if [[ "$MODE" == "decode" ]]; then
    uv run python - <<PY
import json
from pathlib import Path
scan = json.loads(
    (Path("out") / "$PROFILE_NAME" / "decode_position_scan.json").read_text()
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
