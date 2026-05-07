"""Extract the overall kernel execution flow from a Nsight Systems profile.

Usage:
    uv run python analyze/extract_kernel_flow.py path/to/profile.nsys-rep

Steps:
  1. nsys export --type sqlite (run lazily; reuses existing .sqlite if fresh).
  2. Identify the dominant compute stream (highest cumulative kernel time).
  3. Pull the ordered kernel timeline on that stream, joined to short and
     demangled names.
  4. Persist to out/<profile-stem>/kernel_flow.parquet and print a summary.

Output columns: start, end, dur_ns, short_name, demangled_name.
Rows are in execution order on the dominant stream.
"""
from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def ensure_sqlite(profile: Path) -> Path:
    """Export the profile to sqlite next to itself if missing or stale."""
    sqlite_path = profile.with_suffix(".sqlite")
    fresh = (
        sqlite_path.exists()
        and sqlite_path.stat().st_mtime >= profile.stat().st_mtime
    )
    if fresh:
        print(f"# reusing existing sqlite: {sqlite_path}")
        return sqlite_path
    print(f"# nsys export {profile.name} -> {sqlite_path.name}")
    subprocess.run(
        ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
         "-o", str(sqlite_path), str(profile)],
        check=True,
    )
    return sqlite_path


def extract_flow(sqlite_path: Path) -> tuple[pd.DataFrame, int]:
    con = sqlite3.connect(sqlite_path)
    streams = pd.read_sql(
        """
        SELECT streamId, COUNT(*) AS n_kernels,
               SUM(end - start) AS total_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY streamId ORDER BY total_ns DESC
        """,
        con,
    )
    print("\n# Streams (top 5 by cumulative kernel time)")
    streams_view = streams.copy()
    streams_view["total_ms"] = streams_view["total_ns"] / 1e6
    print(streams_view.head(5).drop(columns=["total_ns"]).to_string(index=False))
    dominant = int(streams.iloc[0]["streamId"])

    df = pd.read_sql(
        f"""
        SELECT k.start, k.end, (k.end - k.start) AS dur_ns,
               sn.value AS short_name, dn.value AS demangled_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds sn ON k.shortName     = sn.id
        LEFT JOIN StringIds dn ON k.demangledName = dn.id
        WHERE k.streamId = {dominant}
        ORDER BY k.start
        """,
        con,
    )
    return df, dominant


def print_summary(df: pd.DataFrame, dominant: int) -> None:
    span_ms = (df["end"].max() - df["start"].min()) / 1e6
    print(f"\n# Kernel flow on stream {dominant}: "
          f"{len(df):,} kernels over {span_ms:.2f} ms")

    print("\n# Top 15 kernel names by cumulative duration:")
    top = (
        df.groupby("short_name", as_index=False)
          .agg(n=("dur_ns", "size"),
               total_us=("dur_ns", lambda s: round(s.sum() / 1e3, 2)),
               mean_us =("dur_ns", lambda s: round(s.mean() / 1e3, 3)))
          .sort_values("total_us", ascending=False)
          .head(15)
    )
    print(top.to_string(index=False))

    head_n, tail_n = 15, 5
    t0 = int(df["start"].iloc[0])
    print(f"\n# First {head_n} kernels (t relative to first kernel start):")
    for r in df.head(head_n).itertuples():
        print(f"  t={ (r.start - t0)/1e3:9.2f}us  dur={r.dur_ns/1e3:7.2f}us  "
              f"{r.short_name}")
    if len(df) > head_n + tail_n:
        print(f"\n... ({len(df) - head_n - tail_n:,} kernels in between) ...\n")
        print(f"# Last {tail_n} kernels:")
        for r in df.tail(tail_n).itertuples():
            print(f"  t={ (r.start - t0)/1e3:9.2f}us  dur={r.dur_ns/1e3:7.2f}us  "
                  f"{r.short_name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile", type=Path, help="path to a .nsys-rep file")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: out/<profile-stem>/)")
    args = ap.parse_args()

    profile = args.profile.resolve()
    if not profile.exists():
        print(f"ERROR: profile not found: {profile}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (ROOT / "out" / profile.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"# profile : {profile}")
    print(f"# out_dir : {out_dir}")

    sqlite_path = ensure_sqlite(profile)
    df, dominant = extract_flow(sqlite_path)
    out_path = out_dir / "kernel_flow.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n-> wrote {out_path} ({len(df):,} rows)")

    print_summary(df, dominant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
