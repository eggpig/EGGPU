#!/usr/bin/env python3
"""Render the controlled five-call workflow figure from audited artifacts.

The input directory is expected to be produced by
``generate_workflow_minimum_artifacts_20260730.py``.  The figure reports the
minimum-of-five public-call latency on the left and the corresponding
cumulative workflow latency on the right.  Whiskers are sample standard
deviations (ddof=1) over the same five fresh processes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SELF_CONTROL_NAME = "workflow_five_call_self_control_minimum.csv"
CUMULATIVE_NAME = "workflow_five_call_reuse_cumulative_minimum.csv"
OUTPUT_STEM = "workflow_five_call_controlled_minimum"
RETAINED_BASELINE = "EGGPU"
REBUILD_BASELINE = "EGGPU-isolated"
EXPECTED_ESTIMATOR = "minimum of five fresh processes"
FIXED_PDF_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def read_csv(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(required_columns) - columns
        if missing:
            raise ValueError(f"{path.name} lacks columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return rows


def positive_float(row: dict[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid {field}={row.get(field)!r}") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{source}: {field} must be positive and finite")
    return value


def nonnegative_float(row: dict[str, str], field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid {field}={row.get(field)!r}") from error
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{source}: {field} must be nonnegative and finite")
    return value


def positive_int(row: dict[str, str], field: str, source: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{source}: invalid {field}={row.get(field)!r}") from error
    if value <= 0:
        raise ValueError(f"{source}: {field} must be positive")
    return value


def require_minimum_of_five(rows: Iterable[dict[str, str]], source: str) -> None:
    estimators = {row["estimator"].strip() for row in rows}
    if estimators != {EXPECTED_ESTIMATOR}:
        raise ValueError(
            f"{source}: estimator must be {EXPECTED_ESTIMATOR!r}, got "
            f"{sorted(estimators)}"
        )


def load_inputs(
    artifacts_dir: Path,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    self_control_path = artifacts_dir / SELF_CONTROL_NAME
    cumulative_path = artifacts_dir / CUMULATIVE_NAME
    self_control = read_csv(
        self_control_path,
        {
            "call_position",
            "function",
            "metric",
            "reuse_seconds",
            "isolated_seconds",
            "reuse_sample_sd_seconds",
            "isolated_sample_sd_seconds",
            "aggregate_sample_count",
            "isolated_over_reuse",
            "estimator",
        },
    )
    cumulative = read_csv(
        cumulative_path,
        {
            "baseline",
            "call_position",
            "function",
            "cumulative_geomean_seconds",
            "cumulative_sample_sd_seconds",
            "aggregate_sample_count",
            "estimator",
        },
    )
    require_minimum_of_five(self_control, self_control_path.name)
    require_minimum_of_five(cumulative, cumulative_path.name)

    e2e_rows = [row for row in self_control if row["metric"].strip() == "e2e"]
    if not e2e_rows:
        raise ValueError(f"{self_control_path.name}: no e2e rows")
    e2e_rows.sort(
        key=lambda row: positive_int(
            row, "call_position", self_control_path.name
        )
    )
    positions = [
        positive_int(row, "call_position", self_control_path.name)
        for row in e2e_rows
    ]
    expected_positions = list(range(1, len(e2e_rows) + 1))
    if positions != expected_positions:
        raise ValueError(
            f"{self_control_path.name}: call positions must be contiguous; "
            f"got {positions}"
        )
    functions = [row["function"].strip() for row in e2e_rows]
    if any(not function for function in functions) or len(set(functions)) != len(
        functions
    ):
        raise ValueError(f"{self_control_path.name}: functions must be unique")

    retained_seconds = np.asarray(
        [
            positive_float(row, "reuse_seconds", self_control_path.name)
            for row in e2e_rows
        ],
        dtype=float,
    )
    rebuild_seconds = np.asarray(
        [
            positive_float(row, "isolated_seconds", self_control_path.name)
            for row in e2e_rows
        ],
        dtype=float,
    )
    retained_sd = np.asarray(
        [
            nonnegative_float(
                row, "reuse_sample_sd_seconds", self_control_path.name
            )
            for row in e2e_rows
        ],
        dtype=float,
    )
    rebuild_sd = np.asarray(
        [
            nonnegative_float(
                row, "isolated_sample_sd_seconds", self_control_path.name
            )
            for row in e2e_rows
        ],
        dtype=float,
    )
    if {
        positive_int(row, "aggregate_sample_count", self_control_path.name)
        for row in e2e_rows
    } != {5}:
        raise ValueError(f"{self_control_path.name}: error bars require five samples")
    ratios = np.asarray(
        [
            positive_float(row, "isolated_over_reuse", self_control_path.name)
            for row in e2e_rows
        ],
        dtype=float,
    )
    computed_ratios = rebuild_seconds / retained_seconds
    if not np.allclose(ratios, computed_ratios, rtol=1e-9, atol=0.0):
        raise ValueError(
            f"{self_control_path.name}: isolated_over_reuse is inconsistent"
        )

    cumulative_by_baseline: dict[str, list[dict[str, str]]] = {}
    for baseline in (RETAINED_BASELINE, REBUILD_BASELINE):
        rows = [
            row for row in cumulative if row["baseline"].strip() == baseline
        ]
        rows.sort(
            key=lambda row: positive_int(
                row, "call_position", cumulative_path.name
            )
        )
        if len(rows) != len(functions):
            raise ValueError(
                f"{cumulative_path.name}: {baseline} has {len(rows)} calls, "
                f"expected {len(functions)}"
            )
        row_positions = [
            positive_int(row, "call_position", cumulative_path.name)
            for row in rows
        ]
        row_functions = [row["function"].strip() for row in rows]
        if row_positions != expected_positions or row_functions != functions:
            raise ValueError(
                f"{cumulative_path.name}: {baseline} call order disagrees "
                f"with {self_control_path.name}"
            )
        cumulative_by_baseline[baseline] = rows

    unexpected_baselines = {
        row["baseline"].strip() for row in cumulative
    } - {RETAINED_BASELINE, REBUILD_BASELINE}
    if unexpected_baselines:
        raise ValueError(
            f"{cumulative_path.name}: unexpected baselines "
            f"{sorted(unexpected_baselines)}"
        )

    cumulative_retained = np.asarray(
        [
            positive_float(
                row, "cumulative_geomean_seconds", cumulative_path.name
            )
            for row in cumulative_by_baseline[RETAINED_BASELINE]
        ],
        dtype=float,
    )
    cumulative_rebuild = np.asarray(
        [
            positive_float(
                row, "cumulative_geomean_seconds", cumulative_path.name
            )
            for row in cumulative_by_baseline[REBUILD_BASELINE]
        ],
        dtype=float,
    )
    cumulative_retained_sd = np.asarray(
        [
            nonnegative_float(
                row, "cumulative_sample_sd_seconds", cumulative_path.name
            )
            for row in cumulative_by_baseline[RETAINED_BASELINE]
        ],
        dtype=float,
    )
    cumulative_rebuild_sd = np.asarray(
        [
            nonnegative_float(
                row, "cumulative_sample_sd_seconds", cumulative_path.name
            )
            for row in cumulative_by_baseline[REBUILD_BASELINE]
        ],
        dtype=float,
    )
    if {
        positive_int(row, "aggregate_sample_count", cumulative_path.name)
        for row in cumulative
    } != {5}:
        raise ValueError(f"{cumulative_path.name}: error bars require five samples")
    if np.any(np.diff(cumulative_retained) < -1e-12) or np.any(
        np.diff(cumulative_rebuild) < -1e-12
    ):
        raise ValueError(f"{cumulative_path.name}: cumulative values decrease")

    return (
        functions,
        retained_seconds,
        rebuild_seconds,
        retained_sd,
        rebuild_sd,
        cumulative_retained,
        cumulative_rebuild,
        cumulative_retained_sd,
        cumulative_rebuild_sd,
    )


def log_axis_limits(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span_decades = max(np.log10(maximum / minimum), 0.5)
    lower = minimum / (10 ** max(0.14, 0.07 * span_decades))
    # Reserve the upper part of the logarithmic axis for the legend.  A fixed
    # multiplicative margin keeps it clear of both the bars and ratio labels
    # across reruns whose absolute latency changes.
    upper = maximum * (10 ** max(0.62, 0.15 * span_decades))
    return lower, upper


def render_figure(
    functions: list[str],
    retained_seconds: np.ndarray,
    rebuild_seconds: np.ndarray,
    retained_sd: np.ndarray,
    rebuild_sd: np.ndarray,
    cumulative_retained: np.ndarray,
    cumulative_rebuild: np.ndarray,
    cumulative_retained_sd: np.ndarray,
    cumulative_rebuild_sd: np.ndarray,
    output_dir: Path,
) -> tuple[Path, Path]:
    blue = "#2878B5"
    orange = "#D95F02"
    grid = "#D7DCE2"
    annotation = "#333333"
    x = np.arange(len(functions), dtype=float)
    width = 0.36

    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.4,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.6,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    with mpl.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35))
        left, right = axes

        left.bar(
            x - width / 2,
            retained_seconds,
            width,
            color=blue,
            edgecolor="white",
            linewidth=0.4,
            label="Retained state",
            yerr=np.vstack(
                (np.minimum(retained_sd, 0.95 * retained_seconds), retained_sd)
            ),
            error_kw={
                "elinewidth": 0.7,
                "capsize": 2.0,
                "capthick": 0.7,
                "ecolor": "#235E82",
            },
        )
        left.bar(
            x + width / 2,
            rebuild_seconds,
            width,
            color=orange,
            edgecolor="white",
            linewidth=0.4,
            label="Rebuild per call",
            yerr=np.vstack(
                (np.minimum(rebuild_sd, 0.95 * rebuild_seconds), rebuild_sd)
            ),
            error_kw={
                "elinewidth": 0.7,
                "capsize": 2.0,
                "capthick": 0.7,
                "ecolor": "#8F3D07",
            },
        )
        left.set_yscale("log")
        lower, upper = log_axis_limits(
            np.concatenate(
                (
                    np.maximum(retained_seconds - retained_sd, 0.05 * retained_seconds),
                    rebuild_seconds + rebuild_sd,
                    retained_seconds + retained_sd,
                    np.maximum(rebuild_seconds - rebuild_sd, 0.05 * rebuild_seconds),
                )
            )
        )
        left.set_ylim(lower, upper)
        left.set_xticks(x, functions)
        left.set_ylabel("Per-call E2E latency (s)")
        left.set_title("(a) Per-function public-call latency")
        left.grid(axis="y", color=grid, linewidth=0.55, which="both")
        left.set_axisbelow(True)
        largest_index = int(
            np.argmax(np.maximum(retained_seconds, rebuild_seconds))
        )
        legend_location = (
            "upper left" if largest_index >= len(functions) / 2 else "upper right"
        )
        left.legend(loc=legend_location, frameon=False)
        ratios = rebuild_seconds / retained_seconds
        label_multiplier = 10 ** max(
            0.06, 0.035 * np.log10(upper / lower)
        )
        for index, (maximum, ratio) in enumerate(
            zip(
                np.maximum(retained_seconds + retained_sd, rebuild_seconds + rebuild_sd),
                ratios,
                strict=True,
            )
        ):
            left.text(
                index,
                maximum * label_multiplier,
                f"{ratio:.1f}×",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color=annotation,
            )

        right.errorbar(
            x,
            cumulative_retained,
            yerr=np.vstack(
                (
                    np.minimum(
                        cumulative_retained_sd,
                        0.95 * cumulative_retained,
                    ),
                    cumulative_retained_sd,
                )
            ),
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            color=blue,
            label="Retained state",
        )
        right.errorbar(
            x,
            cumulative_rebuild,
            yerr=np.vstack(
                (
                    np.minimum(
                        cumulative_rebuild_sd,
                        0.95 * cumulative_rebuild,
                    ),
                    cumulative_rebuild_sd,
                )
            ),
            marker="s",
            markersize=4.2,
            linewidth=1.8,
            elinewidth=0.7,
            capsize=2.0,
            capthick=0.7,
            color=orange,
            label="Rebuild per call",
        )
        right.fill_between(
            x,
            cumulative_retained,
            cumulative_rebuild,
            color=blue,
            alpha=0.08,
            linewidth=0,
        )
        right.set_xticks(x, functions)
        right.set_ylabel("Cumulative E2E latency (s)")
        right.set_title("(b) Cumulative workflow latency")
        cumulative_maximum = float(
            np.max(
                np.concatenate(
                    (
                        cumulative_retained + cumulative_retained_sd,
                        cumulative_rebuild + cumulative_rebuild_sd,
                    )
                )
            )
        )
        right.set_ylim(0.0, cumulative_maximum * 1.20)
        right.set_xlim(-0.2, float(x[-1]) + 0.95)
        right.grid(axis="y", color=grid, linewidth=0.55)
        right.set_axisbelow(True)
        right.legend(loc="upper left", frameon=False)

        final_retained = float(cumulative_retained[-1])
        final_rebuild = float(cumulative_rebuild[-1])
        final_ratio = final_rebuild / final_retained
        bracket_x = float(x[-1]) + 0.20
        lower_endpoint, upper_endpoint = sorted((final_retained, final_rebuild))
        right.annotate(
            "",
            xy=(bracket_x, upper_endpoint),
            xytext=(bracket_x, lower_endpoint),
            arrowprops={
                "arrowstyle": "<->",
                "color": "#555555",
                "lw": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        right.text(
            bracket_x + 0.08,
            (lower_endpoint + upper_endpoint) / 2,
            f"{final_ratio:.2f}×\nfinal",
            ha="left",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            color=annotation,
            linespacing=0.9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 0.6,
            },
        )

        for axis in axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        fig.subplots_adjust(
            left=0.075,
            right=0.995,
            top=0.89,
            bottom=0.18,
            wspace=0.30,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
        png_path = output_dir / f"{OUTPUT_STEM}.png"
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={
                "Title": "Controlled five-call workflow latency",
                "Author": "EGGPU",
                "Creator": Path(__file__).name,
                "Producer": "Matplotlib",
                "CreationDate": FIXED_PDF_DATE,
                "ModDate": FIXED_PDF_DATE,
            },
        )
        fig.savefig(
            png_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.02,
            metadata={
                "Title": "Controlled five-call workflow latency",
                "Author": "EGGPU",
                "Software": Path(__file__).name,
            },
        )
        plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render the controlled workflow minimum-of-five Figure 4 inputs "
            "as a two-panel PDF and PNG."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=Path,
        help=(
            "Directory containing workflow_five_call_self_control_minimum.csv "
            "and workflow_five_call_reuse_cumulative_minimum.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for the PDF and PNG",
    )
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir.resolve(strict=True)
    inputs = load_inputs(artifacts_dir)
    pdf_path, png_path = render_figure(*inputs, args.output_dir.resolve())
    print(pdf_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
