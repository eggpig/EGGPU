#!/usr/bin/env python3
"""Summarize EGGPU dataset-function pairs that are not pair-level fastest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_rows(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        candidate = input_path / "category_estimate_details.csv"
        if candidate.exists():
            input_path = candidate
        else:
            frames = []
            for metric in ("e2e", "kernel"):
                metric_path = input_path / f"results_{metric}.csv"
                if metric_path.exists():
                    frames.append(pd.read_csv(metric_path))
            if not frames:
                raise SystemExit(f"no result CSV found under {input_path}")
            return pd.concat(frames, ignore_index=True)
    return pd.read_csv(input_path)


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[(df["status"] == "ok") & df["seconds"].notna() & (df["seconds"] > 0)].copy()
    details = []
    summaries = []
    detail_columns = [
        "dataset",
        "function",
        "eggpu_seconds",
        "best_seconds",
        "best_baseline",
        "best_baseline_seconds",
        "metric",
        "ratio_to_best",
        "is_pair_sota",
    ]
    summary_columns = [
        "metric",
        "function",
        "total_pairs",
        "sota_pairs",
        "non_sota_pairs",
        "worst_ratio_to_best",
        "mean_ratio_to_best",
        "main_non_sota_datasets",
    ]
    for metric in ("e2e", "kernel"):
        d = df[df["metric"] == metric].copy()
        eggpu = d[d["baseline"] == "EGGPU"][["dataset", "function", "seconds"]].rename(
            columns={"seconds": "eggpu_seconds"}
        )
        best = d.groupby(["dataset", "function"], as_index=False)["seconds"].min().rename(
            columns={"seconds": "best_seconds"}
        )
        best_baseline = (
            d.sort_values("seconds")
            .groupby(["dataset", "function"], as_index=False)
            .first()[["dataset", "function", "baseline", "seconds"]]
            .rename(columns={"baseline": "best_baseline", "seconds": "best_baseline_seconds"})
        )
        merged = eggpu.merge(best, on=["dataset", "function"]).merge(
            best_baseline, on=["dataset", "function"]
        )
        merged["metric"] = metric
        merged["ratio_to_best"] = merged["eggpu_seconds"] / merged["best_seconds"]
        merged["is_pair_sota"] = merged["ratio_to_best"] <= 1.0005
        details.append(merged)

        for function, group in merged.groupby("function"):
            losses = group[~group["is_pair_sota"]]
            summaries.append(
                {
                    "metric": metric,
                    "function": function,
                    "total_pairs": int(len(group)),
                    "sota_pairs": int(group["is_pair_sota"].sum()),
                    "non_sota_pairs": int(len(losses)),
                    "worst_ratio_to_best": float(group["ratio_to_best"].max()),
                    "mean_ratio_to_best": float(group["ratio_to_best"].mean()),
                    "main_non_sota_datasets": ", ".join(
                        losses.sort_values("ratio_to_best", ascending=False)
                        .head(5)["dataset"]
                        .tolist()
                    ),
                }
            )

    details_df = (
        pd.concat(details, ignore_index=True)
        if details
        else pd.DataFrame(columns=detail_columns)
    )
    if details_df.empty:
        details_df = pd.DataFrame(columns=detail_columns)
    summary_df = pd.DataFrame(summaries, columns=summary_columns)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["metric", "non_sota_pairs", "worst_ratio_to_best"],
            ascending=[True, False, False],
        )
    return details_df, summary_df


def write_markdown(out_dir: Path, details: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# EGGPU Non-SOTA Pair Summary",
        "",
        "A pair is counted as SOTA when EGGPU is within 0.05% of the fastest successful baseline for the same dataset and function.",
        "",
    ]
    for metric in ("e2e", "kernel"):
        if summary.empty:
            sdf = pd.DataFrame(columns=summary.columns)
            total_pairs = 0
        else:
            sdf = summary[(summary["metric"] == metric) & (summary["non_sota_pairs"] > 0)]
            total_pairs = int(summary[summary["metric"] == metric]["total_pairs"].sum())
        non_sota = int(sdf["non_sota_pairs"].sum())
        lines.extend(
            [
                f"## {metric.upper()}",
                "",
                f"EGGPU is not pair-level fastest on {non_sota}/{total_pairs} pairs.",
                "",
                "| Function | Non-SOTA pairs | Worst ratio | Main datasets |",
                "|---|---:|---:|---|",
            ]
        )
        for row in sdf.itertuples(index=False):
            lines.append(
                f"| {row.function} | {row.non_sota_pairs}/{row.total_pairs} | "
                f"{row.worst_ratio_to_best:.2f}x | {row.main_non_sota_datasets} |"
            )
        lines.append("")

        ddf = details[(details["metric"] == metric) & (~details["is_pair_sota"])].sort_values(
            "ratio_to_best", ascending=False
        )
        lines.extend(
            [
                f"### Worst {metric.upper()} pairs",
                "",
                "| Function | Dataset | EGGPU seconds | Best baseline | Best seconds | Ratio |",
                "|---|---|---:|---|---:|---:|",
            ]
        )
        for row in ddf.head(20).itertuples(index=False):
            lines.append(
                f"| {row.function} | {row.dataset} | {row.eggpu_seconds:.6g} | "
                f"{row.best_baseline} | {row.best_seconds:.6g} | {row.ratio_to_best:.2f}x |"
            )
        lines.append("")

    (out_dir / "EGGPU_NON_SOTA_PAIRS.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="category_estimate_details.csv, category dir, or full result dir")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    input_path = args.input.resolve()
    out_dir = args.out_dir or (input_path if input_path.is_dir() else input_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(input_path)
    details, summary = summarize(df)
    details.to_csv(out_dir / "eggpu_pair_sota_details.csv", index=False)
    summary.to_csv(out_dir / "eggpu_pair_sota_summary.csv", index=False)
    write_markdown(out_dir, details, summary)
    print(f"Wrote non-SOTA summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
