"""Aggregate latency breakdowns across a sweep of profiles.

Reads every `*.nsys-rep` in <scan_dir> and looks up its
`out/<stem>/breakdown.json`.  Prints a cross-profile table:

  - decode sweep (stems like `decode_bs<B>_out<O>`):
      one row per batch_size with iter-latency + per-category breakdown.
  - prefill sweep (stems like `prefill_in<L>`):
      one row per input_len.

Usage:
    python reports/aggregate_scan.py <scan_dir>

E.g.   python reports/aggregate_scan.py profile/results/decode_scan/gpt-oss-20b/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


CAT_LABELS = {
    "Attention": ["qkv_projection", "attn_score", "o_projection"],
    "FFN":       ["ffn_up_projection", "ffn_down_projection"],
    "LM Head":   ["lm_head"],
}


def parse_decode(stem: str) -> tuple[int, int] | None:
    m = re.match(r"decode_bs(\d+)_out(\d+)$", stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_prefill(stem: str) -> int | None:
    m = re.match(r"prefill_in(\d+)$", stem)
    return int(m.group(1)) if m else None


def categorize(totals_us: dict[str, float]) -> dict:
    used: set[str] = set()
    out: dict[str, float] = {}
    for cat, labels in CAT_LABELS.items():
        out[cat] = sum(totals_us.get(l, 0.0) for l in labels)
        used.update(labels)
    out["Other"] = sum(v for k, v in totals_us.items() if k not in used)
    return out


def load_rows(scan_dir: Path) -> tuple[str, list[dict]]:
    profiles = sorted(scan_dir.glob("*.nsys-rep"))
    if not profiles:
        sys.exit(f"ERROR: no .nsys-rep files in {scan_dir}")

    stems = [p.stem for p in profiles]
    is_decode = any(s.startswith("decode") for s in stems)
    is_prefill = any(s.startswith("prefill") for s in stems)
    if is_decode and is_prefill:
        sys.exit("ERROR: scan dir mixes decode and prefill profiles")
    mode = "decode" if is_decode else "prefill"

    rows: list[dict] = []
    for p in profiles:
        stem = p.stem
        bd_path = ROOT / "out" / stem / "breakdown.json"
        if not bd_path.exists():
            print(f"# WARN: no breakdown.json for {stem} (pipeline failed?)",
                  file=sys.stderr)
            continue
        bd = json.loads(bd_path.read_text())
        totals_us = bd["totals_us"]
        cats_us = categorize(totals_us)
        total_us = bd["sum_labeled_us"]
        row = {
            "stem": stem,
            "total_ms": total_us / 1000.0,
            "attn_ms":   cats_us["Attention"] / 1000.0,
            "ffn_ms":    cats_us["FFN"] / 1000.0,
            "lmh_ms":    cats_us["LM Head"] / 1000.0,
            "other_ms":  cats_us["Other"] / 1000.0,
            "attn_score_ms": totals_us.get("attn_score", 0.0) / 1000.0,
        }
        if mode == "decode":
            parsed = parse_decode(stem)
            if parsed:
                row["batch_size"], row["output_len"] = parsed
        else:
            il = parse_prefill(stem)
            if il is not None:
                row["input_len"] = il
        rows.append(row)

    if mode == "decode":
        rows.sort(key=lambda r: r.get("batch_size", 0))
    else:
        rows.sort(key=lambda r: r.get("input_len", 0))
    return mode, rows


def fmt_pct(num_ms: float, denom_ms: float) -> str:
    return f"{(num_ms / denom_ms * 100):>5.1f}%" if denom_ms > 0 else "    -"


def print_decode_table(rows: list[dict]) -> None:
    print("=" * 102)
    print(" CROSS-PROFILE LATENCY SUMMARY  (decode sweep)")
    print("=" * 102)
    print(f" {'batch':>5}  {'out_len':>7}  {'iter (ms)':>10}  "
          f"{'attn (ms)':>10}  {'ffn (ms)':>10}  {'lmh (ms)':>10}  "
          f"{'other (ms)':>11}  {'attn%':>6}  {'ffn%':>6}  {'lmh%':>6}")
    print(" " + "-" * 100)
    for r in rows:
        bs = r.get("batch_size", "?")
        ol = r.get("output_len", "?")
        t = r["total_ms"]
        print(f" {bs:>5}  {ol:>7}  {t:>10.3f}  "
              f"{r['attn_ms']:>10.3f}  {r['ffn_ms']:>10.3f}  "
              f"{r['lmh_ms']:>10.3f}  {r['other_ms']:>11.3f}  "
              f"{fmt_pct(r['attn_ms'], t)}  "
              f"{fmt_pct(r['ffn_ms'], t)}  "
              f"{fmt_pct(r['lmh_ms'], t)}")


def print_prefill_table(rows: list[dict]) -> None:
    print("=" * 92)
    print(" CROSS-PROFILE LATENCY SUMMARY  (prefill sweep)")
    print("=" * 92)
    print(f" {'input':>5}  {'iter (ms)':>10}  "
          f"{'attn (ms)':>10}  {'ffn (ms)':>10}  {'lmh (ms)':>10}  "
          f"{'other (ms)':>11}  {'attn%':>6}  {'ffn%':>6}  {'lmh%':>6}")
    print(" " + "-" * 90)
    for r in rows:
        il = r.get("input_len", "?")
        t = r["total_ms"]
        print(f" {il:>5}  {t:>10.3f}  "
              f"{r['attn_ms']:>10.3f}  {r['ffn_ms']:>10.3f}  "
              f"{r['lmh_ms']:>10.3f}  {r['other_ms']:>11.3f}  "
              f"{fmt_pct(r['attn_ms'], t)}  "
              f"{fmt_pct(r['ffn_ms'], t)}  "
              f"{fmt_pct(r['lmh_ms'], t)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scan_dir", help="dir containing the *.nsys-rep sweep")
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    if not scan_dir.exists():
        sys.exit(f"ERROR: scan dir not found: {scan_dir}")

    mode, rows = load_rows(scan_dir)
    if not rows:
        sys.exit("ERROR: no rows to summarize (no breakdown.json files found)")

    print()
    if mode == "decode":
        print_decode_table(rows)
    else:
        print_prefill_table(rows)
    print()
    print(f" Source: {scan_dir}/  ({len(rows)} profiles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
