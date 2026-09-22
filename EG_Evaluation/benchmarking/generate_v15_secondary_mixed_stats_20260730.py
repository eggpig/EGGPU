#!/usr/bin/env python3
"""Regenerate EGGPU-only secondary timing summaries under the min5 policy.

Current EGGPU timing centers are minima of five fresh process measurements.
Every displayed ``+/-`` value is the sample standard deviation (ddof=1) of
those same five measurements.  Family- and suite-level centers are geometric
means of cell-level min5 values; their error is the sample standard deviation
across the five repetition-index geometric means on the identical cell set.

Storage bytes are deterministic rather than timing measurements.  The
pre-existing 30-sequence multi-view robustness evidence is audited in the
manifest but excluded from the active timing table because it does not follow
the five-fresh-process protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_INDICES = (1, 2, 3, 4, 5)
ABLATION_REPEATS = (0, 1, 2, 3, 4)
FAMILY_ORDER = (
    "Centrality",
    "Connectivity",
    "Paths & Spanning Trees",
    "Structural Holes",
)
FAMILY_LABEL = {
    "Centrality": "Centrality",
    "Connectivity": "Connectivity",
    "Paths & Spanning Trees": r"Path \& Spanning",
    "Structural Holes": "Structural Holes",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geomean(values) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(float)
    if len(numeric) == 0 or np.any(~np.isfinite(numeric)) or np.any(numeric <= 0):
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.log(numeric).mean()))


def require_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(
        float(observed),
        float(expected),
        rel_tol=1.0e-10,
        abs_tol=1.0e-13,
    ):
        raise ValueError(f"{label}: {observed} != {expected}")


def prepare_eggpu_e2e_samples(
    path: Path,
    keys: set[tuple[str, str]],
) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    required = {
        "dataset",
        "function",
        "baseline",
        "metric",
        "seconds",
        "sample_index",
        "sample_count",
        "status",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path.name} lacks fields: {missing}")
    if "measurement_phase" in data:
        data = data[data["measurement_phase"].astype(str).eq("timing")]
    data = data[
        data["baseline"].astype(str).eq("EGGPU")
        & data["metric"].astype(str).eq("e2e")
        & data["status"].astype(str).eq("ok")
    ].copy()
    data = data[
        data.apply(
            lambda row: (str(row["dataset"]), str(row["function"])) in keys,
            axis=1,
        )
    ]
    data["sample_index"] = pd.to_numeric(
        data["sample_index"], errors="raise"
    ).astype(int)
    data["sample_count"] = pd.to_numeric(
        data["sample_count"], errors="raise"
    ).astype(int)
    data["seconds"] = pd.to_numeric(data["seconds"], errors="raise")
    if not data["sample_count"].eq(5).all():
        raise ValueError(f"{path.name} contains a non-five sample declaration")
    return data


def validate_raw_against_pair_summary(
    raw: pd.DataFrame,
    pair_summary: pd.DataFrame,
    *,
    paper_column: str,
    mean_column: str,
    sd_column: str,
    label: str,
) -> None:
    grouped = (
        raw.groupby(["dataset", "function"], sort=False)["seconds"]
        .agg(["count", "min", "mean", "std"])
        .reset_index()
    )
    merged = pair_summary.merge(
        grouped,
        on=["dataset", "function"],
        how="left",
        validate="one_to_one",
    )
    if merged["count"].isna().any() or not merged["count"].eq(5).all():
        raise ValueError(f"{label}: raw five-sample groups are incomplete")
    for observed, expected, pair in zip(
        merged["min"],
        merged[paper_column],
        zip(merged["dataset"], merged["function"], strict=True),
        strict=True,
    ):
        require_close(observed, expected, f"{label}/{pair}/minimum")
    for observed, expected, pair in zip(
        merged["mean"],
        merged[mean_column],
        zip(merged["dataset"], merged["function"], strict=True),
        strict=True,
    ):
        require_close(observed, expected, f"{label}/{pair}/mean")
    for observed, expected, pair in zip(
        merged["std"],
        merged[sd_column],
        zip(merged["dataset"], merged["function"], strict=True),
        strict=True,
    ):
        require_close(observed, expected, f"{label}/{pair}/sample-SD")


def family_stats(
    pair_summary_path: Path,
    first_non_bc_path: Path,
    first_bc_path: Path,
    steady_path: Path,
) -> pd.DataFrame:
    pairs = pd.read_csv(pair_summary_path, low_memory=False)
    required = {
        "dataset",
        "function",
        "metric",
        "family",
        "sample_count",
        "first_use_paper_seconds",
        "first_use_mean_seconds",
        "first_use_std_seconds",
        "steady_paper_seconds",
        "steady_mean_seconds",
        "steady_std_seconds",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"{pair_summary_path.name} lacks fields: {missing}")
    pairs = pairs[pairs["metric"].astype(str).eq("e2e")].copy()
    if not pd.to_numeric(pairs["sample_count"], errors="raise").eq(5).all():
        raise ValueError("first-use pair summary contains a non-five cell")
    if set(pairs["family"]) != set(FAMILY_ORDER):
        raise ValueError("first-use family set is incomplete")
    keys = {(str(row.dataset), str(row.function)) for row in pairs.itertuples()}

    first_non_bc = prepare_eggpu_e2e_samples(first_non_bc_path, keys)
    first_bc = prepare_eggpu_e2e_samples(first_bc_path, keys)
    first = pd.concat(
        [
            first_non_bc[~first_non_bc["function"].astype(str).eq("BC")],
            first_bc[first_bc["function"].astype(str).eq("BC")],
        ],
        ignore_index=True,
    )
    steady = prepare_eggpu_e2e_samples(steady_path, keys)
    validate_raw_against_pair_summary(
        first,
        pairs,
        paper_column="first_use_paper_seconds",
        mean_column="first_use_mean_seconds",
        sd_column="first_use_std_seconds",
        label="first-use",
    )
    validate_raw_against_pair_summary(
        steady,
        pairs,
        paper_column="steady_paper_seconds",
        mean_column="steady_mean_seconds",
        sd_column="steady_std_seconds",
        label="reused-state",
    )

    rows = []
    for family in FAMILY_ORDER:
        family_pairs = pairs[pairs["family"].eq(family)]
        family_keys = {
            (str(row.dataset), str(row.function))
            for row in family_pairs.itertuples()
        }
        first_part = first[
            first.apply(
                lambda row: (str(row["dataset"]), str(row["function"]))
                in family_keys,
                axis=1,
            )
        ]
        steady_part = steady[
            steady.apply(
                lambda row: (str(row["dataset"]), str(row["function"]))
                in family_keys,
                axis=1,
            )
        ]
        first_samples = []
        steady_samples = []
        for sample_index in SAMPLE_INDICES:
            first_sample = first_part[
                first_part["sample_index"].eq(sample_index)
            ]
            steady_sample = steady_part[
                steady_part["sample_index"].eq(sample_index)
            ]
            if len(first_sample) != len(family_pairs) or len(steady_sample) != len(
                family_pairs
            ):
                raise ValueError(
                    f"{family}/sample-{sample_index}: incomplete family evidence"
                )
            first_samples.append(geomean(first_sample["seconds"]))
            steady_samples.append(geomean(steady_sample["seconds"]))
        first_center = geomean(family_pairs["first_use_paper_seconds"])
        steady_center = geomean(family_pairs["steady_paper_seconds"])
        ratio_samples = np.asarray(first_samples) / np.asarray(steady_samples)
        rows.append(
            {
                "family": family,
                "pairs": len(family_pairs),
                "first_use_min5_geomean_seconds": first_center,
                "first_use_aggregate_sample_sd_seconds": float(
                    np.std(first_samples, ddof=1)
                ),
                "steady_min5_geomean_seconds": steady_center,
                "steady_aggregate_sample_sd_seconds": float(
                    np.std(steady_samples, ddof=1)
                ),
                "first_over_steady": first_center / steady_center,
                "ratio_aggregate_sample_sd": float(
                    np.std(ratio_samples, ddof=1)
                ),
                "aggregate_sample_count": 5,
                "estimator": "minimum_of_five_per_cell",
                "reported_error": "sample_standard_deviation_ddof1",
                "first_use_aggregate_samples_json": json.dumps(first_samples),
                "steady_aggregate_samples_json": json.dumps(steady_samples),
            }
        )
    return pd.DataFrame(rows)


def complete_repeat_pivots(
    data: pd.DataFrame,
    *,
    case_columns: list[str],
    variant_column: str,
    variants: tuple[str, str],
    repeat_values: tuple[int, ...],
) -> list[tuple[tuple, pd.DataFrame]]:
    pivots = []
    selected = data[data[variant_column].isin(variants)]
    for key, group in selected.groupby(case_columns, dropna=False, sort=False):
        pivot = group.pivot_table(
            index="repeat",
            columns=variant_column,
            values="seconds",
            aggfunc="first",
        )
        if not set(variants).issubset(pivot.columns):
            continue
        pivot = pivot[list(variants)].dropna()
        if tuple(sorted(pivot.index.astype(int))) != repeat_values:
            continue
        pivots.append((key if isinstance(key, tuple) else (key,), pivot))
    return pivots


def ratio_summary(
    pivots: list[tuple[tuple, pd.DataFrame]],
    *,
    numerator: str,
    denominator: str,
    repeat_values: tuple[int, ...],
) -> tuple[float, float, list[float]]:
    if not pivots:
        raise ValueError(f"no complete paired cases for {numerator}/{denominator}")
    central = geomean(
        [
            float(pivot[numerator].min())
            / float(pivot[denominator].min())
            for _key, pivot in pivots
        ]
    )
    samples = [
        geomean(
            [
                float(pivot.loc[repeat, numerator])
                / float(pivot.loc[repeat, denominator])
                for _key, pivot in pivots
            ]
        )
        for repeat in repeat_values
    ]
    return central, float(np.std(samples, ddof=1)), samples


def ablation_stats(ablation_all_path: Path) -> pd.DataFrame:
    data = pd.read_csv(ablation_all_path, low_memory=False)
    required = {
        "experiment",
        "variant",
        "dataset",
        "function",
        "metric",
        "value",
        "repeat",
        "status",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{ablation_all_path.name} lacks fields: {missing}")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data["repeat"] = pd.to_numeric(data["repeat"], errors="coerce")
    data = data[
        data["status"].astype(str).eq("ok")
        & data["value"].notna()
        & np.isfinite(data["value"])
        & data["value"].gt(0)
    ].copy()

    workflow = data[
        data["experiment"].eq("workflow") & data["metric"].eq("e2e")
    ].copy()
    if "workflow_order_id" not in workflow:
        raise ValueError("workflow ablation lacks workflow_order_id")
    workflow["repeat"] = workflow["repeat"].astype(int)
    workflow = (
        workflow.groupby(
            ["dataset", "workflow_order_id", "variant", "repeat"],
            dropna=False,
        )["value"]
        .sum()
        .reset_index(name="seconds")
    )

    rows = []
    for module, numerator, denominator in (
        ("GraphContext", "no_graph_context", "full"),
        ("C++ graph cache", "no_cpp_graph_cache", "full"),
    ):
        pivots = complete_repeat_pivots(
            workflow,
            case_columns=["dataset", "workflow_order_id"],
            variant_column="variant",
            variants=(numerator, denominator),
            repeat_values=ABLATION_REPEATS,
        )
        center, error, samples = ratio_summary(
            pivots,
            numerator=numerator,
            denominator=denominator,
            repeat_values=ABLATION_REPEATS,
        )
        rows.append(
            {
                "module": module,
                "endpoint": "Workflow",
                "paired_cases": len(pivots),
                "ratio": center,
                "ratio_sample_sd": error,
                "sample_count": 5,
                "estimator": "minimum_of_five_per_variant_case",
                "reported_error": "sample_standard_deviation_ddof1",
                "aggregate_ratio_samples_json": json.dumps(samples),
            }
        )

    returned = data[data["experiment"].eq("return")].copy()
    if "materialization_claim_valid" in returned:
        returned = returned[
            returned["materialization_claim_valid"]
            .astype(str)
            .str.lower()
            .eq("true")
        ]
    if "return_equivalent" in returned:
        equivalent = returned["return_equivalent"].astype(str).str.lower()
        returned = returned[
            equivalent.eq("true") | returned["return_equivalent"].isna()
        ]
    returned = returned[
        returned["metric"].isin(
            ("call_return_seconds", "standard_container_return_seconds")
        )
    ].copy()
    returned["repeat"] = returned["repeat"].astype(int)
    returned["seconds"] = returned["value"]
    pivots = complete_repeat_pivots(
        returned,
        case_columns=["dataset", "function"],
        variant_column="metric",
        variants=(
            "standard_container_return_seconds",
            "call_return_seconds",
        ),
        repeat_values=ABLATION_REPEATS,
    )
    center, error, samples = ratio_summary(
        pivots,
        numerator="standard_container_return_seconds",
        denominator="call_return_seconds",
        repeat_values=ABLATION_REPEATS,
    )
    rows.append(
        {
            "module": "Result reconstruction",
            "endpoint": "Return",
            "paired_cases": len(pivots),
            "ratio": center,
            "ratio_sample_sd": error,
            "sample_count": 5,
            "estimator": "minimum_of_five_per_variant_case",
            "reported_error": "sample_standard_deviation_ddof1",
            "aggregate_ratio_samples_json": json.dumps(samples),
        }
    )

    layout = data[
        data["experiment"].eq("layout")
        & data["metric"].isin(("host_storage_mb", "degree_seconds"))
        & data["function"].isin(("COO", "CSR"))
    ].copy()
    storage = layout[layout["metric"].eq("host_storage_mb")]
    storage_pivot = storage.pivot_table(
        index="dataset",
        columns="function",
        values="value",
        aggfunc="first",
    ).dropna()
    storage_ratio = geomean(storage_pivot["COO"] / storage_pivot["CSR"])
    rows.append(
        {
            "module": "CSR storage",
            "endpoint": "Storage",
            "paired_cases": len(storage_pivot),
            "ratio": storage_ratio,
            "ratio_sample_sd": np.nan,
            "sample_count": np.nan,
            "estimator": "deterministic_derived_bytes",
            "reported_error": "not_applicable",
            "aggregate_ratio_samples_json": "[]",
        }
    )

    traversal = layout[layout["metric"].eq("degree_seconds")].copy()
    traversal["repeat"] = traversal["repeat"].astype(int)
    traversal["seconds"] = traversal["value"]
    pivots = complete_repeat_pivots(
        traversal,
        case_columns=["dataset"],
        variant_column="function",
        variants=("COO", "CSR"),
        repeat_values=ABLATION_REPEATS,
    )
    center, error, samples = ratio_summary(
        pivots,
        numerator="COO",
        denominator="CSR",
        repeat_values=ABLATION_REPEATS,
    )
    rows.append(
        {
            "module": "CSR traversal",
            "endpoint": "Traversal",
            "paired_cases": len(pivots),
            "ratio": center,
            "ratio_sample_sd": error,
            "sample_count": 5,
            "estimator": "minimum_of_five_per_variant_case",
            "reported_error": "sample_standard_deviation_ddof1",
            "aggregate_ratio_samples_json": json.dumps(samples),
        }
    )
    return pd.DataFrame(rows)


def multiview_range(paths: list[Path]) -> tuple[float, float, list[dict]]:
    values = []
    evidence = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass":
            raise ValueError(f"multi-view acceptance did not pass: {path}")
        count = int((payload.get("max1") or {}).get("count", -1))
        pooled_count = int((payload.get("max4_pooled") or {}).get("count", -1))
        if count != 30 or pooled_count != 60:
            raise ValueError(f"unexpected multi-view robustness counts: {path}")
        value = float(
            (payload.get("speedup_max4_over_max1") or {})["median"]
        )
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid multi-view median speedup: {path}")
        values.append(value)
        evidence.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "median_speedup": value,
                "max1_sequences": count,
                "max4_pooled_sequences": pooled_count,
            }
        )
    return min(values), max(values), evidence


def portable_path(path: Path, output: Path) -> str:
    try:
        return path.relative_to(output).as_posix()
    except ValueError:
        return str(path)


def write_table(
    path: Path,
    first_use: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    lookup = ablation.set_index("module")
    lines = [
        "% Generated by generate_v15_secondary_mixed_stats_20260730.py",
        r"% Requires: booktabs, tabularx, xcolor, graphicx",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{First-call cost and controlled ablations. EGGPU timing cells in this table use the minimum of five fresh-process measurements. Family centers are geometric means over fixed cell sets; their $\pm$ terms are the sample standard deviation ($ddof=1$) of the five repetition-index family geometric means. Ablation $\pm$ terms use the corresponding five paired repetition-index aggregate ratios. Storage is a deterministic byte-count comparison rather than a timing measurement. \reviewhl{These mechanism studies use frozen pre-V15 revisions recorded in the portable artifact and are not pooled with the V15 headline comparisons.}}",
        r"\label{tab:ablation-core}",
        r"\scriptsize",
        r"\begin{minipage}[t]{0.51\textwidth}",
        r"\centering",
        r"\textbf{(a) First call versus reused-state call}\par\smallskip",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Function family & First call & Reused-state call & First / reused \\",
        r" & (ms) & (ms) &  \\",
        r"\midrule",
    ]
    for family in FAMILY_ORDER:
        row = first_use[first_use["family"].eq(family)].iloc[0]
        first_ms = 1000.0 * float(row["first_use_min5_geomean_seconds"])
        first_sd_ms = 1000.0 * float(
            row["first_use_aggregate_sample_sd_seconds"]
        )
        steady_ms = 1000.0 * float(row["steady_min5_geomean_seconds"])
        steady_sd_ms = 1000.0 * float(
            row["steady_aggregate_sample_sd_seconds"]
        )
        ratio = float(row["first_over_steady"])
        lines.append(
            f"{FAMILY_LABEL[family]} & "
            rf"\reviewhl{{{first_ms:.1f} $\pm$ {first_sd_ms:.1f}}} & "
            rf"\reviewhl{{{steady_ms:.1f} $\pm$ {steady_sd_ms:.1f}}} & "
            rf"\reviewhl{{{ratio:.1f}$\times$}} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{minipage}\hfill%",
            r"\begin{minipage}[t]{0.47\textwidth}",
            r"\centering",
            r"\textbf{(b) Controlled module ablation}\par\smallskip",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xlr@{}}",
            r"\toprule",
            r"Ablation variant & Endpoint & Increase \\",
            r"\midrule",
        ]
    )
    ablation_rows = (
        ("GraphContext", "w/o GraphContext"),
        ("C++ graph cache", "w/o C++ graph cache"),
        (
            "Result reconstruction",
            "w/o selective result materialization",
        ),
    )
    for module, label in ablation_rows:
        row = lookup.loc[module]
        lines.append(
            f"{label} & {row['endpoint']} & "
            rf"\reviewhl{{({float(row['ratio']):.2f} $\pm$ "
            rf"{float(row['ratio_sample_sd']):.2f})$\times$}} \\"
        )
    storage = lookup.loc["CSR storage"]
    traversal = lookup.loc["CSR traversal"]
    lines.extend(
        [
            "w/o CSR host layout (use COO) & Storage & "
            rf"\reviewhl{{{float(storage['ratio']):.2f}$\times$}} \\",
            "w/o CSR traversal (use COO) & Traversal & "
            rf"\reviewhl{{({float(traversal['ratio']):.2f} $\pm$ "
            rf"{float(traversal['ratio_sample_sd']):.2f})$\times$}} \\",
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{minipage}",
            r"\vspace{1pt}",
            r"\begin{minipage}{0.98\textwidth}\footnotesize Every comparison fixes the graph, function or workflow, call order, and result semantics. The w/o GraphContext variant also removes the C++ cache, whereas the w/o C++ graph-cache variant retains GraphContext; their ratios are nested and not additive.",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-use-pair-summary", required=True, type=Path)
    parser.add_argument("--first-use-non-bc-samples", required=True, type=Path)
    parser.add_argument("--first-use-bc-samples", required=True, type=Path)
    parser.add_argument("--steady-samples", required=True, type=Path)
    parser.add_argument("--ablation-all", required=True, type=Path)
    parser.add_argument(
        "--multiview-acceptance",
        required=True,
        type=Path,
        action="append",
    )
    parser.add_argument(
        "--provenance-file",
        type=Path,
        action="append",
        default=[],
        help="Portable runtime/source provenance file to hash-bind in the manifest.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    sources = [
        args.first_use_pair_summary.resolve(strict=True),
        args.first_use_non_bc_samples.resolve(strict=True),
        args.first_use_bc_samples.resolve(strict=True),
        args.steady_samples.resolve(strict=True),
        args.ablation_all.resolve(strict=True),
        *[
            path.resolve(strict=True)
            for path in args.multiview_acceptance
        ],
    ]
    first = family_stats(*sources[:4])
    ablation = ablation_stats(sources[4])
    multiview_low, multiview_high, multiview_evidence = multiview_range(
        sources[5:]
    )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance_files = [
        path.resolve(strict=True) for path in args.provenance_file
    ]
    all_sources_portable = all(
        path == output or output in path.parents
        for path in [*sources, *provenance_files]
    )
    first_path = output / "first_use_family_min5_with_sd.csv"
    ablation_path = output / "ablation_min5_with_sd.csv"
    table_path = output / "paper_table_first_call_ablation.tex"
    first.to_csv(first_path, index=False)
    ablation.to_csv(ablation_path, index=False)
    write_table(
        table_path,
        first,
        ablation,
    )
    manifest = {
        "schema_version": "eggpu_secondary_timing_mixed_estimator_v2_portable",
        "status": "pass",
        "point_estimator": "minimum_of_five",
        "reported_error": "sample_standard_deviation_ddof1_same_five",
        "first_use_pairs": int(first["pairs"].sum()),
        "ablation_timing_rows": int(
            ablation["sample_count"].notna().sum()
        ),
        "non_five_process_rows": {
            "CSR_storage": "deterministic_derived_bytes",
            "multi_view_device_retention": (
                "excluded_from_active_timing_table_incompatible_"
                "30_sequence_median_protocol"
            ),
        },
        "excluded_multiview_speedup_range": [
            multiview_low,
            multiview_high,
        ],
        "multiview_evidence": [
            {
                **entry,
                "path": portable_path(Path(entry["path"]), output),
            }
            for entry in multiview_evidence
        ],
        "relationship_to_main_v15": (
            "Frozen pre-V15 mechanism studies; not pooled with the V15 "
            "headline timing matrix."
        ),
        "source_paths_are_release_relative": all_sources_portable,
        "sources": [
            {
                "path": portable_path(path, output),
                "sha256": sha256(path),
            }
            for path in sources
        ],
        "revision_provenance": [
            {
                "path": portable_path(path, output),
                "sha256": sha256(path),
            }
            for path in provenance_files
        ],
        "artifacts": {
            path.name: sha256(path)
            for path in (first_path, ablation_path, table_path)
        },
    }
    manifest_path = output / "SECONDARY_MIXED_ESTIMATOR_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
