"""Check an upload file against sample_submission.csv before it goes to Kaggle.

Same rows in the same order, same task per row, no blanks, queue_pred in {0,1},
finite non-negative values. Also prints per-task value summaries so an all-zero
task (which would score 0) is easy to spot.

    python -I submit/check_submission.py <submission.csv> [--template <sample_submission.csv>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
import os
TEMPLATE = Path(os.environ.get("TFB_REL", ROOT / "data" / "kaggle_public")) / "sample_submission.csv"
COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
VALUE_COLUMN = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    args = ap.parse_args()

    sub = pl.read_csv(args.submission, infer_schema_length=0)
    tpl = pl.read_csv(args.template, columns=["submission_id", "task"], infer_schema_length=0)
    problems = []

    if sub.columns != COLUMNS:
        problems.append(f"columns {sub.columns} != {COLUMNS}")
    if sub.height != tpl.height:
        problems.append(f"rows {sub.height:,} != template {tpl.height:,}")
    else:
        if not sub["submission_id"].equals(tpl["submission_id"]):
            problems.append("submission_id order differs from the template")
        if not sub["task"].equals(tpl["task"]):
            problems.append("task column differs from the template")
    if sub["submission_id"].n_unique() != sub.height:
        problems.append("duplicate submission_id")

    blanks = {c: sub[c].is_null().sum() + (sub[c] == "").sum() for c in sub.columns}
    blanks = {c: n for c, n in blanks.items() if n}
    if blanks:
        problems.append(f"blank cells: {blanks}")

    num = sub.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in COLUMNS[2:]])
    for c in COLUMNS[2:]:
        bad = num.filter(~pl.col(c).is_finite() | (pl.col(c) < 0)).height
        if bad:
            problems.append(f"{c}: {bad:,} non-finite or negative values")
    q = num.filter(pl.col("task") == "queue")
    nonbinary = q.filter(~pl.col("queue_pred").is_in([0.0, 1.0])).height
    if nonbinary:
        problems.append(f"queue_pred: {nonbinary:,} values not in {{0,1}}")

    print(f"file: {args.submission}  rows: {sub.height:,}  size: {args.submission.stat().st_size / 1e6:.1f} MB")
    for task, cols in VALUE_COLUMN.items():
        part = num.filter(pl.col("task") == task)
        for c in cols:
            s = part[c]
            print(f"  {task:5s} {c:10s} n={part.height:>9,}  mean={s.mean():10.3f}  "
                  f"min={s.min():9.3f}  max={s.max():10.3f}  zeros={(s == 0).sum():>9,}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("OK: matches the template and passes the evaluator's format rules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
