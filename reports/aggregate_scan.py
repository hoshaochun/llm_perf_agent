"""Aggregate latency breakdowns across a sweep of profiles.

Reads every `*.nsys-rep` in <scan_dir> and looks up its
`out/<profile_name>/breakdown.json` (and theoretical_latency.json when
present), then prints two cross-profile tables:

  Section 1: ITERATION LATENCY (one row per profile)
      end-to-end iter latency, theoretical bound, ratio, and the
      identified bottleneck category + bound type (compute / memory).

  Section 2: PER-CATEGORY BREAKDOWN (one row per profile)
      for each of Attention / FFN / LM Head / Other:
          actual ms / theoretical ms / actual÷theor / % of iter

Theoretical figures come from `theoretical_latency.json`; if that file
is missing for a profile the theor + ratio cells fall back to `-`.

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
CATS = ["Attention", "FFN", "LM Head", "Other"]


def parse_decode(profile_name: str) -> tuple[int, int] | None:
    m = re.match(r"decode_bs(\d+)_out(\d+)$", profile_name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_prefill(profile_name: str) -> int | None:
    m = re.match(r"prefill_in(\d+)$", profile_name)
    return int(m.group(1)) if m else None


def categorize_actual(totals_us: dict[str, float]) -> dict[str, float]:
    """Sum actual durations into the four reporting categories."""
    used: set[str] = set()
    out: dict[str, float] = {}
    for cat, labels in CAT_LABELS.items():
        out[cat] = sum(totals_us.get(l, 0.0) for l in labels)
        used.update(labels)
    out["Other"] = sum(v for k, v in totals_us.items() if k not in used)
    return out


def categorize_theor(theor_rows: list[dict]
                     ) -> tuple[dict[str, float | None], dict[str, str | None]]:
    """Sum theoretical bounds + tag each category compute/memory-bound.

    Returns (theor_us_per_cat, bound_kind_per_cat). 'Other' is always
    (None, None) since the predictor doesn't model misc kernels.
    """
    by_label = {row["label"]: row for row in theor_rows}
    theor_us = {cat: None for cat in CATS}
    bound_kind = {cat: None for cat in CATS}
    for cat, labels in CAT_LABELS.items():
        t_us = 0.0
        c_us = 0.0
        m_us = 0.0
        has = False
        for lab in labels:
            r = by_label.get(lab)
            if r is None:
                continue
            t_us += float(r["theor_bound_us"])
            c_us += float(r["theor_compute_us"])
            m_us += float(r["theor_memory_us"])
            has = True
        if has:
            theor_us[cat] = t_us
            bound_kind[cat] = "compute" if c_us >= m_us else "memory"
    return theor_us, bound_kind


def find_bottleneck(cats_actual_ms: dict[str, float],
                    cats_bound: dict[str, str | None],
                    iter_actual_ms: float) -> str:
    """Bottleneck = category with the highest actual ms (= where time is
    spent), qualified by its compute/memory bound type."""
    cat = max(cats_actual_ms, key=lambda c: cats_actual_ms[c])
    pct = (cats_actual_ms[cat] / iter_actual_ms * 100) if iter_actual_ms > 0 else 0.0
    bound = cats_bound.get(cat)
    if bound:
        return f"{cat} ({bound}-bound, {pct:.1f}%)"
    return f"{cat} ({pct:.1f}%)"


def load_rows(scan_dir: Path) -> tuple[str, list[dict]]:
    profiles = sorted(scan_dir.glob("*.nsys-rep"))
    if not profiles:
        sys.exit(f"ERROR: no .nsys-rep files in {scan_dir}")

    profile_names = [p.stem for p in profiles]
    is_decode = any(s.startswith("decode") for s in profile_names)
    is_prefill = any(s.startswith("prefill") for s in profile_names)
    if is_decode and is_prefill:
        sys.exit("ERROR: scan dir mixes decode and prefill profiles")
    mode = "decode" if is_decode else "prefill"

    rows: list[dict] = []
    for p in profiles:
        profile_name = p.stem
        bd_path = ROOT / "out" / profile_name / "breakdown.json"
        if not bd_path.exists():
            print(f"# WARN: no breakdown.json for {profile_name} (pipeline failed?)",
                  file=sys.stderr)
            continue
        bd = json.loads(bd_path.read_text())

        th_path = ROOT / "out" / profile_name / "theoretical_latency.json"
        th = json.loads(th_path.read_text()) if th_path.exists() else None

        totals_us = bd["totals_us"]
        iter_actual_us = float(bd["sum_labeled_us"])
        cats_actual_us = categorize_actual(totals_us)

        if th is not None:
            cats_theor_us, cats_bound = categorize_theor(th["rows"])
            # iter theoretical = sum of categorical theoretical bounds + the
            # actual time spent in 'Other' (which we don't roofline).  This
            # gives "fastest the iter could plausibly run" assuming misc
            # overhead is unavoidable.
            iter_theor_us = (
                sum(v for v in cats_theor_us.values() if v is not None)
                + cats_actual_us["Other"]
            )
        else:
            cats_theor_us = {cat: None for cat in CATS}
            cats_bound    = {cat: None for cat in CATS}
            iter_theor_us = None

        bottleneck = find_bottleneck(
            {c: v / 1000.0 for c, v in cats_actual_us.items()},
            cats_bound,
            iter_actual_us / 1000.0,
        )

        row = {
            "profile_name":   profile_name,
            "iter_actual_ms": iter_actual_us / 1000.0,
            "iter_theor_ms":  (iter_theor_us / 1000.0) if iter_theor_us else None,
            "cats_actual_ms": {c: v / 1000.0 for c, v in cats_actual_us.items()},
            "cats_theor_ms":  {c: (v / 1000.0 if v else None)
                               for c, v in cats_theor_us.items()},
            "cats_bound":     cats_bound,
            "bottleneck":     bottleneck,
            "has_theor":      th is not None,
        }
        if mode == "decode":
            parsed = parse_decode(profile_name)
            if parsed:
                row["batch_size"], row["output_len"] = parsed
        else:
            il = parse_prefill(profile_name)
            if il is not None:
                row["input_len"] = il
        rows.append(row)

    rows.sort(key=lambda r: r.get("batch_size" if mode == "decode" else "input_len", 0))
    return mode, rows


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


def _row_index(row: dict, mode: str) -> str:
    if mode == "decode":
        bs = row.get("batch_size", "?")
        ol = row.get("output_len", "?")
        return f" {bs:>5}  {ol:>7}"
    return f" {row.get('input_len', '?'):>5}"


def _index_header(mode: str) -> str:
    if mode == "decode":
        return f" {'batch':>5}  {'out_len':>7}"
    return f" {'input':>5}"


def print_iter_table(rows: list[dict], mode: str) -> None:
    title = f"{mode.upper()} SWEEP — ITERATION LATENCY  (actual vs theoretical roofline)"
    print()
    print("=" * 100)
    print(f" {title}")
    print("=" * 100)
    head_idx = _index_header(mode)
    print(f"{head_idx}  {'iter (ms)':>10}  {'theor (ms)':>10}  "
          f"{'a/t':>7}  bottleneck")
    print(" " + "-" * 98)
    for r in rows:
        idx = _row_index(r, mode)
        a = r["iter_actual_ms"]
        t = r["iter_theor_ms"]
        if t and t > 0:
            t_str = f"{t:>10.3f}"
            r_str = f"{a/t:>5.2f}x"
        else:
            t_str = f"{'-':>10}"
            r_str = f"{'-':>6}"
        print(f"{idx}  {a:>10.3f}  {t_str}  {r_str}  {r['bottleneck']}")


def _fmt_cat_cell(act_ms: float, theor_ms: float | None,
                  iter_ms: float) -> str:
    pct = (act_ms / iter_ms * 100) if iter_ms > 0 else 0.0
    if theor_ms is not None and theor_ms > 0:
        ratio = act_ms / theor_ms
        return f"{act_ms:>7.2f} {theor_ms:>7.2f} {ratio:>4.2f}x {pct:>5.1f}%"
    return f"{act_ms:>7.2f} {'-':>7} {'-':>5} {pct:>5.1f}%"


def print_category_table(rows: list[dict], mode: str) -> None:
    title = (f"{mode.upper()} SWEEP — PER-CATEGORY BREAKDOWN  "
             "(actual ms / theor ms / actual÷theor / % of iter)")
    print()
    print("=" * 132)
    print(f" {title}")
    print("=" * 132)
    head_idx = _index_header(mode)
    cell_w = 28
    cat_hdr = "  ".join(f"{cat:^{cell_w}}" for cat in CATS)
    print(f"{head_idx}  {cat_hdr}")
    pad = " " * len(head_idx)
    sub = "  ".join(
        f"{'act':>7} {'theor':>7} {'r':>5} {'%':>6}" for _ in CATS
    )
    print(f"{pad}  {sub}")
    print(" " + "-" * 130)
    for r in rows:
        idx = _row_index(r, mode)
        iter_ms = r["iter_actual_ms"]
        cells = []
        for cat in CATS:
            cells.append(_fmt_cat_cell(
                r["cats_actual_ms"][cat],
                r["cats_theor_ms"][cat],
                iter_ms,
            ))
        print(f"{idx}  " + "  ".join(cells))


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

    print_iter_table(rows, mode)
    print_category_table(rows, mode)

    print()
    n_with_theor = sum(1 for r in rows if r["has_theor"])
    print(f" Source: {scan_dir}/  ({len(rows)} profiles, "
          f"{n_with_theor} with theoretical roofline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
