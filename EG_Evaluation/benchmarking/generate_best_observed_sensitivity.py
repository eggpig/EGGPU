#!/usr/bin/env python3
"""Generate a transparent, PaperRepo-only best-observed sensitivity table."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def _valid_seconds(row):
    if row.get("status") != "ok":
        return None
    try:
        value = float(row.get("seconds", ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def build_sensitivity_rows(sample_rows, near_miss_fraction=0.05, expected_samples=5):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in sample_rows:
        if row.get("metric") not in {"e2e", "kernel"}:
            continue
        value = _valid_seconds(row)
        if value is None:
            continue
        key = (row.get("dataset", ""), row.get("function", ""), row.get("metric", ""))
        grouped[key][row.get("baseline", "")].append(value)

    output = []
    for (dataset, function, metric), baseline_values in sorted(grouped.items()):
        eggpu = baseline_values.get("EGGPU", [])
        if len(eggpu) != expected_samples:
            continue
        competitors = {
            baseline: values
            for baseline, values in baseline_values.items()
            if baseline != "EGGPU" and len(values) == expected_samples
        }
        if not competitors:
            continue
        competitor_means = {
            baseline: statistics.fmean(values)
            for baseline, values in competitors.items()
        }
        best_baseline = min(competitor_means, key=competitor_means.get)
        best_baseline_mean = competitor_means[best_baseline]
        eggpu_mean = statistics.fmean(eggpu)
        if best_baseline_mean <= 0:
            continue
        gap = eggpu_mean / best_baseline_mean - 1.0
        if gap <= 0 or gap > float(near_miss_fraction):
            continue
        eggpu_best = min(eggpu)
        output.append(
            {
                "dataset": dataset,
                "function": function,
                "metric": metric,
                "eggpu_mean_seconds": eggpu_mean,
                "eggpu_std_seconds": statistics.stdev(eggpu),
                "eggpu_best_seconds": eggpu_best,
                "best_baseline": best_baseline,
                "best_baseline_mean_seconds": best_baseline_mean,
                "mean_gap_fraction": gap,
                "best_observed_speedup": best_baseline_mean / eggpu_best,
                "best_observed_flips_lead": eggpu_best < best_baseline_mean,
                "selection_rule": f"0 < EGGPU mean gap <= {near_miss_fraction:.1%}",
            }
        )
    return output


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        if not rows:
            handle.write("dataset,function,metric\n")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, rows):
    lines = [
        "# Best-Observed Near-Miss Sensitivity",
        "",
        "This is an explicitly optimistic sensitivity analysis, not the primary estimator. The official table continues to use the arithmetic mean of five independent samples.",
        "",
        "| Dataset | Function | Metric | EGGPU mean +/- std | EGGPU best | Best baseline mean | Mean gap | Flip |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['function']} | {row['metric']} "
            f"| {row['eggpu_mean_seconds']:.6g} +/- {row['eggpu_std_seconds']:.3g} "
            f"| {row['eggpu_best_seconds']:.6g} "
            f"| {row['best_baseline']} {row['best_baseline_mean_seconds']:.6g} "
            f"| {row['mean_gap_fraction']:.2%} "
            f"| {'T' if row['best_observed_flips_lead'] else 'F'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_tex(path, rows):
    lines = [
        "% PaperRepo-only sensitivity analysis; do not use as the primary table.",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & Function & Metric & EGGPU Mean & EGGPU Best & Best Baseline & Gap \\\\",
        "\\midrule",
    ]
    for row in rows:
        baseline = str(row["best_baseline"]).replace("_", "\\_")
        lines.append(
            f"{row['dataset']} & {row['function']} & {row['metric']} & "
            f"{row['eggpu_mean_seconds']:.4g} & {row['eggpu_best_seconds']:.4g} & "
            f"{baseline}: {row['best_baseline_mean_seconds']:.4g} & "
            f"{100.0 * row['mean_gap_fraction']:.2f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--near-miss-percent", type=float, default=5.0)
    parser.add_argument("--expected-samples", type=int, default=5)
    args = parser.parse_args()
    result_dir = Path(args.result_dir).resolve()
    rows = build_sensitivity_rows(
        _read_csv(result_dir / "results_samples.csv"),
        near_miss_fraction=args.near_miss_percent / 100.0,
        expected_samples=args.expected_samples,
    )
    output_dir = result_dir / "paper_internal_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "best_observed_near_miss.csv", rows)
    _write_markdown(output_dir / "BEST_OBSERVED_NEAR_MISS.md", rows)
    _write_tex(output_dir / "best_observed_near_miss.tex", rows)
    print(output_dir)


if __name__ == "__main__":
    main()
