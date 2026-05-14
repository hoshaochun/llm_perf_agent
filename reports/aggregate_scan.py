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


def parse_decode(profile_name: str) -> dict | None:
    """Parse decode profile names.

    Two conventions:
      `decode_bs<B>_in<I>`  -- long-input method, one KV-size per profile.
      `decode_bs<B>_out<O>` -- legacy long-output method.

    Returns a dict with `batch_size` and one of `input_len`/`output_len`.
    """
    m = re.match(r"decode_bs(\d+)_in(\d+)$", profile_name)
    if m:
        return {"batch_size": int(m.group(1)), "input_len": int(m.group(2))}
    m = re.match(r"decode_bs(\d+)_out(\d+)$", profile_name)
    if m:
        return {"batch_size": int(m.group(1)), "output_len": int(m.group(2))}
    return None


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


def _find_theoretical_json(gpu: str, model: str, profile_name: str
                           ) -> Path | None:
    """Locate `theoretical_latency.json` for a profile in the
    reports/<gpu>/<model>/<profile_name>/ layout."""
    direct = (ROOT / "reports" / gpu / model / profile_name
              / "theoretical_latency.json")
    return direct if direct.exists() else None


def load_rows(gpu: str, model: str, scan_dir: Path) -> tuple[str, list[dict]]:
    profiles = sorted(scan_dir.glob("*.nsys-rep"))
    if not profiles:
        sys.exit(f"ERROR: no .nsys-rep files in {scan_dir}")

    profile_names = [p.stem for p in profiles]
    is_decode = any(s.startswith("decode") for s in profile_names)
    is_prefill = any(s.startswith("prefill") for s in profile_names)
    if is_decode and is_prefill:
        sys.exit("ERROR: scan dir mixes decode and prefill profiles")
    mode = "decode" if is_decode else "prefill"

    model_dirname = model.replace("/", "_")
    rows: list[dict] = []
    for p in profiles:
        profile_name = p.stem
        bd_path = ROOT / "out" / gpu / model_dirname / profile_name / "breakdown.json"
        if not bd_path.exists():
            print(f"# WARN: no breakdown.json for {profile_name} (pipeline failed?)",
                  file=sys.stderr)
            continue
        bd = json.loads(bd_path.read_text())

        th_path = _find_theoretical_json(gpu, model_dirname, profile_name)
        th = json.loads(th_path.read_text()) if th_path is not None else None

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

        # Per-op attn_score detail (used by the attn-vs-decode-position table).
        attn_actual_us = float(totals_us.get("attn_score", 0.0))
        attn_theor_us = None
        attn_bound_kind = None
        if th is not None:
            for r in th["rows"]:
                if r["label"] == "attn_score":
                    attn_theor_us = float(r["theor_bound_us"])
                    attn_bound_kind = r.get("bound_kind")
                    break

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
            "attn_actual_ms": attn_actual_us / 1000.0,
            "attn_theor_ms":  (attn_theor_us / 1000.0) if attn_theor_us else None,
            "attn_bound":     attn_bound_kind,
        }
        if mode == "decode":
            parsed = parse_decode(profile_name) or {}
            if "batch_size" in parsed:
                row["batch_size"] = parsed["batch_size"]
            if "input_len" in parsed:
                row["input_len"] = parsed["input_len"]
            if "output_len" in parsed:
                row["output_len"] = parsed["output_len"]
        else:
            il = parse_prefill(profile_name)
            if il is not None:
                row["input_len"] = il
        rows.append(row)

    # Pick the sweep variable: input_len (new long-input decode method,
    # prefill) > output_len (legacy decode) > batch_size (legacy decode).
    if mode == "decode":
        has_input = any("input_len" in r for r in rows)
        has_output = any("output_len" in r for r in rows)
        if has_input:
            sweep_key = "input_len"
        elif has_output:
            sweep_key = "output_len"
        else:
            sweep_key = "batch_size"
    else:
        sweep_key = "input_len"
    rows.sort(key=lambda r: r.get(sweep_key, 0))
    for r in rows:
        r["_sweep_key"] = sweep_key
    return mode, rows


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


_SWEEP_KEY_HEADER = {
    "batch_size": "batch",
    "input_len":  "input",
    "output_len": "out_len",
}


def _sweep_key(rows: list[dict]) -> str:
    return rows[0].get("_sweep_key", "batch_size") if rows else "batch_size"


def _row_index(row: dict, mode: str, sweep_key: str) -> str:
    if mode == "decode":
        bs = row.get("batch_size", "?")
        if sweep_key == "input_len":
            il = row.get("input_len", "?")
            return f" {bs:>5}  {il:>7}"
        ol = row.get("output_len", "?")
        return f" {bs:>5}  {ol:>7}"
    return f" {row.get('input_len', '?'):>5}"


def _index_header(mode: str, sweep_key: str) -> str:
    if mode == "decode":
        sweep_lbl = _SWEEP_KEY_HEADER.get(sweep_key, "swept")
        return f" {'batch':>5}  {sweep_lbl:>7}"
    return f" {'input':>5}"


def _group_decode_by_kv(rows: list[dict]) -> list[tuple[int, list[dict]]]:
    """Group decode rows by input_len (= kv_cache_len), batch_size inner."""
    by_kv: dict[int, list[dict]] = {}
    for r in rows:
        kv = int(r.get("input_len", r.get("output_len", 0)))
        by_kv.setdefault(kv, []).append(r)
    out = []
    for kv in sorted(by_kv):
        group = sorted(by_kv[kv], key=lambda r: r.get("batch_size", 0))
        out.append((kv, group))
    return out


def _fmt_iter_row(r: dict) -> str:
    bs = r.get("batch_size", "?")
    a = r["iter_actual_ms"]
    t = r["iter_theor_ms"]
    if t and t > 0:
        t_str = f"{t:>10.3f}"
        r_str = f"{a/t:>5.2f}x"
    else:
        t_str = f"{'-':>10}"
        r_str = f"{'-':>6}"
    return f"   {bs:>5}  {a:>10.3f}  {t_str}  {r_str}  {r['bottleneck']}"


def print_iter_table(rows: list[dict], mode: str) -> None:
    title = f"{mode.upper()} SWEEP — ITERATION LATENCY  (actual vs theoretical roofline)"
    print()
    print("=" * 100)
    print(f" {title}")
    print("=" * 100)
    if mode == "decode":
        for kv, group in _group_decode_by_kv(rows):
            print()
            print(f"  kv_cache_len = {kv}")
            print(f"   {'batch':>5}  {'iter (ms)':>10}  {'theor (ms)':>10}  "
                  f"{'a/t':>7}  bottleneck")
            print("   " + "-" * 96)
            for r in group:
                print(_fmt_iter_row(r))
        return
    sk = _sweep_key(rows)
    head_idx = _index_header(mode, sk)
    print(f"{head_idx}  {'iter (ms)':>10}  {'theor (ms)':>10}  "
          f"{'a/t':>7}  bottleneck")
    print(" " + "-" * 98)
    for r in rows:
        idx = _row_index(r, mode, sk)
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


def _fmt_cat_row(r: dict) -> str:
    bs = r.get("batch_size", "?")
    iter_ms = r["iter_actual_ms"]
    cells = []
    for cat in CATS:
        cells.append(_fmt_cat_cell(
            r["cats_actual_ms"][cat],
            r["cats_theor_ms"][cat],
            iter_ms,
        ))
    return f"   {bs:>5}  " + "  ".join(cells)


def print_category_table(rows: list[dict], mode: str) -> None:
    title = (f"{mode.upper()} SWEEP — PER-CATEGORY BREAKDOWN  "
             "(actual ms / theor ms / actual÷theor / % of iter)")
    print()
    print("=" * 132)
    print(f" {title}")
    print("=" * 132)
    cell_w = 28
    cat_hdr = "  ".join(f"{cat:^{cell_w}}" for cat in CATS)
    sub = "  ".join(
        f"{'act':>7} {'theor':>7} {'r':>5} {'%':>6}" for _ in CATS
    )
    if mode == "decode":
        for kv, group in _group_decode_by_kv(rows):
            print()
            print(f"  kv_cache_len = {kv}")
            print(f"   {'batch':>5}  {cat_hdr}")
            print(f"   {'':>5}  {sub}")
            print("   " + "-" * 128)
            for r in group:
                print(_fmt_cat_row(r))
        return
    sk = _sweep_key(rows)
    head_idx = _index_header(mode, sk)
    print(f"{head_idx}  {cat_hdr}")
    pad = " " * len(head_idx)
    print(f"{pad}  {sub}")
    print(" " + "-" * 130)
    for r in rows:
        idx = _row_index(r, mode, sk)
        iter_ms = r["iter_actual_ms"]
        cells = []
        for cat in CATS:
            cells.append(_fmt_cat_cell(
                r["cats_actual_ms"][cat],
                r["cats_theor_ms"][cat],
                iter_ms,
            ))
        print(f"{idx}  " + "  ".join(cells))


def print_attn_table(rows: list[dict], mode: str) -> None:
    """attn_score actual vs theoretical, per (batch_size, decode_position).

    Replaces the old in-trace decode_position_scan: with the long-input
    decode method each profile IS one (batch, decode_position) data
    point, so the scan-across-positions table is a cross-profile view.
    Decode position is taken from input_len (new `decode_bs<B>_in<I>`
    convention).  Rows are sorted by (batch_size, input_len).
    """
    if mode != "decode":
        return
    print()
    print("=" * 100)
    print(" DECODE — ATTN_SCORE LATENCY vs DECODE POSITION  "
          "(one row per profile)")
    print("=" * 100)
    print(f" {'batch':>5}  {'input':>7}  "
          f"{'iter (ms)':>10}  {'attn act (ms)':>14}  "
          f"{'attn theor (ms)':>16}  {'a/t':>8}  {'bound':>7}")
    print(" " + "-" * 98)
    # Sort by batch_size then input_len for a coherent 2D sweep view.
    sorted_rows = sorted(
        rows,
        key=lambda r: (r.get("batch_size", 0), r.get("input_len", 0)),
    )
    for r in sorted_rows:
        bs = r.get("batch_size", "?")
        il = r.get("input_len", r.get("output_len", "?"))
        iter_ms = r["iter_actual_ms"]
        a_ms = r["attn_actual_ms"]
        t_ms = r["attn_theor_ms"]
        if t_ms is not None and t_ms > 0:
            t_str = f"{t_ms:>16.3f}"
            r_str = f"{a_ms/t_ms:>6.2f}x"
            bound = r.get("attn_bound") or "-"
        else:
            t_str = f"{'-':>16}"
            r_str = f"{'-':>7}"
            bound = "-"
        print(f" {bs:>5}  {il:>7}  {iter_ms:>10.3f}  {a_ms:>14.3f}  "
              f"{t_str}  {r_str}  {bound:>7}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scan_dir", help="dir containing the *.nsys-rep sweep")
    ap.add_argument("--gpu", required=True,
                    help="GPU name; reads out/<gpu>/<model>/ and reports/<gpu>/<model>/")
    ap.add_argument("--model", required=True,
                    help="model name; reads out/<gpu>/<model>/ and reports/<gpu>/<model>/")
    args = ap.parse_args()

    scan_dir = Path(args.scan_dir)
    if not scan_dir.exists():
        sys.exit(f"ERROR: scan dir not found: {scan_dir}")

    mode, rows = load_rows(args.gpu, args.model, scan_dir)
    if not rows:
        sys.exit("ERROR: no rows to summarize (no breakdown.json files found)")

    print_iter_table(rows, mode)
    print_category_table(rows, mode)
    print_attn_table(rows, mode)

    print()
    n_with_theor = sum(1 for r in rows if r["has_theor"])
    print(f" Source: {scan_dir}/  ({len(rows)} profiles, "
          f"{n_with_theor} with theoretical roofline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
