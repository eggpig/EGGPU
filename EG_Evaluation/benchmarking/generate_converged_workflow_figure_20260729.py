#!/usr/bin/env python3
"""Generate the submission-facing same-graph workflow figure.

This visualization intentionally reports the controlled EGGPU comparison only.
It removes the earlier cross-system cumulative panel, whose final separation was
dominated by Closeness and therefore did not isolate graph-state reuse.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_figure(
    self_control_path: Path,
    cumulative_path: Path,
    output_path: Path,
) -> None:
    per_call = [
        row for row in read_rows(self_control_path) if row["metric"] == "e2e"
    ]
    per_call.sort(key=lambda row: int(row["call_position"]))
    cumulative = read_rows(cumulative_path)
    retained = sorted(
        (row for row in cumulative if row["baseline"] == "EGGPU"),
        key=lambda row: int(row["call_position"]),
    )
    rebuilt = sorted(
        (row for row in cumulative if row["baseline"] == "EGGPU-isolated"),
        key=lambda row: int(row["call_position"]),
    )

    functions = [row["function"] for row in per_call]
    reuse = np.asarray([float(row["reuse_seconds"]) for row in per_call])
    isolated = np.asarray([float(row["isolated_seconds"]) for row in per_call])
    ratios = np.asarray([float(row["isolated_over_reuse"]) for row in per_call])
    cumulative_reuse = np.asarray(
        [float(row["cumulative_geomean_seconds"]) for row in retained]
    )
    cumulative_rebuilt = np.asarray(
        [float(row["cumulative_geomean_seconds"]) for row in rebuilt]
    )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.4,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.6,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    blue = "#2878B5"
    orange = "#D95F02"
    grid = "#D7DCE2"

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
    x = np.arange(len(functions))
    width = 0.36

    axes[0].bar(
        x - width / 2,
        reuse,
        width,
        color=blue,
        edgecolor="white",
        linewidth=0.4,
        label="Retained state",
    )
    axes[0].bar(
        x + width / 2,
        isolated,
        width,
        color=orange,
        edgecolor="white",
        linewidth=0.4,
        label="Rebuild per call",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.005, 1.55)
    axes[0].set_xticks(x, functions)
    axes[0].set_ylabel("Per-call E2E latency (s)")
    axes[0].set_title("(a) Per-function public-call latency")
    axes[0].grid(axis="y", color=grid, linewidth=0.55, which="both")
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper right", frameon=False, ncol=1)
    for idx, (iso, ratio) in enumerate(zip(isolated, ratios, strict=True)):
        axes[0].text(
            idx,
            iso * 1.12,
            f"{ratio:.1f}$\\times$",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#333333",
        )

    axes[1].plot(
        x,
        cumulative_reuse,
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        color=blue,
        label="Retained state",
    )
    axes[1].plot(
        x,
        cumulative_rebuilt,
        marker="s",
        markersize=4.2,
        linewidth=1.8,
        color=orange,
        label="Rebuild per call",
    )
    axes[1].fill_between(
        x,
        cumulative_reuse,
        cumulative_rebuilt,
        color=blue,
        alpha=0.08,
        linewidth=0,
    )
    axes[1].set_xticks(x, functions)
    axes[1].set_ylabel("Cumulative E2E latency (s)")
    axes[1].set_title("(b) Cumulative workflow latency")
    axes[1].set_ylim(0, 2.25)
    axes[1].grid(axis="y", color=grid, linewidth=0.55)
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="upper left", frameon=False)
    final_ratio = cumulative_rebuilt[-1] / cumulative_reuse[-1]
    axes[1].annotate(
        f"{final_ratio:.2f}$\\times$",
        xy=(x[-1], cumulative_rebuilt[-1]),
        xytext=(x[-1] - 0.9, cumulative_rebuilt[-1] - 0.08),
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.075, right=0.995, top=0.89, bottom=0.18, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-control", required=True, type=Path)
    parser.add_argument("--cumulative", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    generate_figure(args.self_control, args.cumulative, args.output)


if __name__ == "__main__":
    main()
