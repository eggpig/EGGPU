#!/usr/bin/env python3
"""Build a reproducible category-level estimate from staged EGGPU results.

This script is intentionally separate from the full benchmark runner.  It is
used when a complete full run predates later targeted optimizations.  The output
is an estimate: it combines the complete baseline rows from one full benchmark
with newer EGGPU rows from targeted runs and micro-checks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd


CATEGORIES = {
    "Ranking/Centrality": ["PageRank", "BC", "Closeness"],
    "Connectivity/Core": ["LCC", "WCC", "SCC", "KCore"],
    "Paths/Trees": ["MST", "BFS", "Dijkstra", "BellmanFord", "SSSP"],
    "Structural Holes": ["EffectiveSize", "Efficiency", "Constraint", "Hierarchy"],
}

DEFAULT_REPLACE_FUNCS = {"BFS", "SSSP", "KCore", "SCC"}


def geomean(values):
    xs = [float(v) for v in values if float(v) > 0 and math.isfinite(float(v))]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def read_metric_csv(result_dir: Path, metric: str) -> pd.DataFrame:
    path = result_dir / f"results_{metric}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return pd.read_csv(path)


def parse_micro_jsonl(path: Path, function: str | None = None):
    if not path or not path.exists():
        return []
    rows = []
    dataset = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("RUN dataset="):
            m = re.search(r"RUN dataset=([^ ]+)", line)
            dataset = m.group(1) if m else None
            continue
        if "RESULT_JSON " not in line:
            continue
        payload = line.split("RESULT_JSON ", 1)[1]
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if obj.get("metric") not in {"e2e", "kernel"}:
            continue
        if function is not None and obj.get("function") != function:
            continue
        rows.append(
            {
                "dataset": dataset,
                "function": obj.get("function"),
                "metric": obj.get("metric"),
                "seconds": float(obj.get("seconds")),
                "source": str(path),
            }
        )
    return rows


def parse_repeat_median(path: Path, dataset: str, function: str):
    if not path or not path.exists():
        return []
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        if "RESULT_JSON " not in line:
            continue
        try:
            obj = json.loads(line.split("RESULT_JSON ", 1)[1])
        except Exception:
            continue
        if obj.get("function") != function or obj.get("metric") not in {"e2e", "kernel"}:
            continue
        values.setdefault(obj["metric"], []).append(float(obj["seconds"]))
    return [
        {
            "dataset": dataset,
            "function": function,
            "metric": metric,
            "seconds": statistics.median(seconds),
            "source": str(path),
        }
        for metric, seconds in values.items()
        if seconds
    ]


def merged_metric(args, metric: str) -> pd.DataFrame:
    full = read_metric_csv(args.full_result, metric)
    target = read_metric_csv(args.weak_target, metric)
    replace_funcs = set(args.replace_funcs)
    df = pd.concat(
        [
            full[~full["function"].isin(replace_funcs)],
            target[target["function"].isin(replace_funcs)],
        ],
        ignore_index=True,
    )

    micro_rows = []
    micro_rows += parse_micro_jsonl(args.constraint_micro, "Constraint")
    micro_rows += parse_repeat_median(args.constraint_repeat, "ca-HepTh", "Constraint")
    micro_rows += parse_micro_jsonl(args.scc_micro, "SCC")
    micro_rows += parse_micro_jsonl(args.kcore_micro, "KCore")

    for row in micro_rows:
        if row["metric"] != metric:
            continue
        mask = (
            (df["baseline"] == "EGGPU")
            & (df["dataset"] == row["dataset"])
            & (df["function"] == row["function"])
            & (df["metric"] == metric)
        )
        if not bool(mask.any()):
            continue
        df.loc[mask, "seconds"] = row["seconds"]
        df.loc[mask, "status"] = "ok"
        if "is_timeout" in df.columns:
            df.loc[mask, "is_timeout"] = False
        df.loc[mask, "notes"] = (
            df.loc[mask, "notes"].fillna("").astype(str)
            + f"; category-estimate override from {row['source']}"
        )
    return df


def summarize_metric(args, metric: str):
    df = merged_metric(args, metric)
    timeout_col = df["is_timeout"].astype(bool) if "is_timeout" in df.columns else False
    ok = df[(df["status"] == "ok") & df["seconds"].notna() & (~timeout_col)].copy()
    best = ok.groupby(["dataset", "function"], as_index=False)["seconds"].min()
    best = best.rename(columns={"seconds": "pair_best_seconds"})
    ok = ok.merge(best, on=["dataset", "function"], how="left")
    ok["ratio_to_pair_best"] = ok["seconds"] / ok["pair_best_seconds"]

    rows = []
    details = []
    for category, funcs in CATEGORIES.items():
        sub = ok[ok["function"].isin(funcs)].copy()
        total_pairs = len(set(zip(sub["dataset"], sub["function"])))
        for baseline, group in sub.groupby("baseline"):
            coverage_pairs = len(set(zip(group["dataset"], group["function"])))
            ratios = group["ratio_to_pair_best"].tolist()
            rows.append(
                {
                    "metric": metric,
                    "category": category,
                    "baseline": baseline,
                    "coverage_pairs": coverage_pairs,
                    "total_pairs": total_pairs,
                    "coverage": f"{coverage_pairs}/{total_pairs}",
                    "geomean_ratio": geomean(ratios),
                    "arithmetic_ratio": float(pd.Series(ratios).mean()),
                    "total_seconds": float(group["seconds"].sum()),
                }
            )
            details.append(group.assign(metric_for_summary=metric, category=category))

    summary = pd.DataFrame(rows)
    rank_rows = []
    for (metric_name, category), group in summary.groupby(["metric", "category"]):
        ordered = group.sort_values(["geomean_ratio", "coverage_pairs"], ascending=[True, False]).copy()
        ordered["rank"] = range(1, len(ordered) + 1)
        rank_rows.append(ordered)
    summary = pd.concat(rank_rows, ignore_index=True)
    detail_df = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return summary, detail_df


def write_markdown(out_path: Path, summary: pd.DataFrame, args):
    lines = [
        "# Category Estimate",
        "",
        "This is a reproducible merged estimate, not a replacement for a final full benchmark.",
        "",
        "Inputs:",
        f"- full result: `{args.full_result}`",
        f"- weak target: `{args.weak_target}`",
        f"- Constraint micro: `{args.constraint_micro}`",
        f"- Constraint repeat: `{args.constraint_repeat}`",
        f"- SCC micro: `{args.scc_micro}`",
        f"- KCore micro: `{args.kcore_micro}`",
        "",
        "Score is geometric mean of `baseline_time / pair_best_time`; lower is better.",
        "",
        "| Metric | Category | Baseline | Coverage | Score | Rank |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in summary.sort_values(["metric", "category", "rank"]).itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.category} | {row.baseline} | {row.coverage} | "
            f"{row.geomean_ratio:.3f} | {int(row.rank)} |"
        )
    lines.append("")
    eggpu = summary[summary["baseline"] == "EGGPU"].sort_values(["metric", "category"])
    lines.append("EGGPU rows:")
    lines.append("")
    lines.append("| Metric | Category | Coverage | Score | Rank |")
    lines.append("|---|---|---:|---:|---:|")
    for row in eggpu.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.category} | {row.coverage} | "
            f"{row.geomean_ratio:.3f} | {int(row.rank)} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full-result",
        type=Path,
        required=True,
        help="Complete full benchmark result directory. Required to avoid accidentally using stale historical results.",
    )
    ap.add_argument(
        "--weak-target",
        type=Path,
        required=True,
        help="Targeted result directory for functions replaced in this estimate.",
    )
    ap.add_argument(
        "--constraint-micro",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--constraint-repeat",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--scc-micro",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--kcore-micro",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--replace-funcs",
        nargs="*",
        default=sorted(DEFAULT_REPLACE_FUNCS),
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.full_result / "category_estimate_current"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    details = []
    for metric in ("e2e", "kernel"):
        summary, detail = summarize_metric(args, metric)
        summaries.append(summary)
        details.append(detail)

    summary = pd.concat(summaries, ignore_index=True)
    detail = pd.concat(details, ignore_index=True)
    summary.to_csv(out_dir / "category_estimate_summary.csv", index=False)
    detail.to_csv(out_dir / "category_estimate_details.csv", index=False)
    write_markdown(out_dir / "CATEGORY_ESTIMATE.md", summary, args)

    print(f"Wrote {out_dir / 'category_estimate_summary.csv'}")
    print(f"Wrote {out_dir / 'CATEGORY_ESTIMATE.md'}")
    eggpu = summary[summary["baseline"] == "EGGPU"].sort_values(["metric", "category"])
    print(eggpu[["metric", "category", "coverage", "geomean_ratio", "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
