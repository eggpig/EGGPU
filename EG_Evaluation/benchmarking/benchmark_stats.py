#!/usr/bin/env python3
"""Statistical aggregation for independent benchmark samples."""

import math
import statistics
from collections import Counter, defaultdict


GROUP_FIELDS = (
    "dataset_size",
    "graph_type",
    "dataset",
    "function",
    "baseline",
    "metric",
    "unit",
    "metric_family",
    "measurement_scope",
    "timer_kind",
    "measurement_window",
    "semantic",
    "estimator_kind",
    "sample_sources",
    "source_policy",
    "source_seed",
    "source_nodes_sha",
)

STATIC_UNAVAILABLE_STATUSES = {"skipped", "unsupported"}


def _float_or_none(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _number(value):
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.12g}"


def _t_critical_95(df):
    # Two-sided 95% Student-t critical values. Normal approximation is adequate
    # beyond the table because benchmark repeat counts are small in practice.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        25: 2.060,
        30: 2.042,
    }
    if df in table:
        return table[df]
    if df < 25:
        return table[20]
    if df < 30:
        return table[25]
    return 1.96


def _group_key(row):
    return tuple(str(row.get(field, "")) for field in GROUP_FIELDS)


def _ordered_unique(values):
    seen = set()
    result = []
    for value in values:
        value = str(value or "")
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def aggregate_sample_rows(sample_rows, expected_samples):
    """Aggregate raw independent samples without hiding incomplete groups."""

    expected_samples = max(1, int(expected_samples))
    grouped = defaultdict(list)
    for row in sample_rows:
        grouped[_group_key(row)].append(dict(row))

    aggregated = []
    for key in sorted(grouped):
        group = sorted(
            grouped[key],
            key=lambda row: int(row.get("sample_index") or 0),
        )
        first = dict(group[0])
        values = []
        for row in group:
            if row.get("status") != "ok":
                continue
            raw_value = row.get("value")
            if raw_value in (None, ""):
                raw_value = row.get("seconds")
            value = _float_or_none(raw_value)
            if value is not None:
                values.append(value)

        statuses = Counter(str(row.get("status", "failed")) for row in group)
        n_total = len(group)
        n_valid = len(values)
        if n_valid == expected_samples and n_total == expected_samples:
            status = "ok"
            publishable = True
        elif n_valid == 0 and statuses and set(statuses) <= STATIC_UNAVAILABLE_STATUSES:
            status = "unsupported" if statuses.get("unsupported") else "skipped"
            publishable = False
        else:
            status = "incomplete"
            publishable = False

        mean = statistics.fmean(values) if values else None
        median = statistics.median(values) if values else None
        minimum = min(values) if values else None
        maximum = max(values) if values else None
        variance = statistics.variance(values) if len(values) >= 2 else None
        std = math.sqrt(variance) if variance is not None else None
        cv = abs(std / mean) if std is not None and mean not in (None, 0.0) else None
        ci_low = None
        ci_high = None
        ci_half_width = None
        if std is not None and mean is not None:
            ci_half_width = _t_critical_95(len(values) - 1) * std / math.sqrt(len(values))
            ci_low = mean - ci_half_width
            ci_high = mean + ci_half_width
        relative_std_percent = cv * 100.0 if cv is not None else None
        relative_ci95_percent = (
            abs(ci_half_width / mean) * 100.0
            if ci_half_width is not None and mean not in (None, 0.0)
            else None
        )
        if len(values) < 2:
            stability_class = "insufficient_samples"
        elif relative_std_percent is None:
            stability_class = "undefined_relative_variation"
        elif relative_std_percent <= 5.0:
            stability_class = "stable_le_5pct_rsd"
        elif relative_std_percent <= 10.0:
            stability_class = "moderate_5_to_10pct_rsd"
        else:
            stability_class = "variable_gt_10pct_rsd"

        first["seconds"] = _number(mean)
        first["value"] = _number(mean)
        first["status"] = status
        correctness_values = _ordered_unique(row.get("correctness") for row in group)
        last_correctness = next(
            (str(row.get("correctness")) for row in reversed(group) if row.get("correctness")),
            "",
        )
        first["correctness"] = last_correctness
        first["correctness_variant_count"] = str(len(correctness_values))
        if not correctness_values:
            first["correctness_consistency"] = "not_reported"
        elif len(correctness_values) == 1:
            first["correctness_consistency"] = "identical"
        else:
            first["correctness_consistency"] = "varied_across_samples"
        first["log"] = ";".join(_ordered_unique(row.get("log") for row in group))
        note_parts = _ordered_unique(row.get("notes") for row in group)
        note_parts.append(
            "sample_statuses="
            + ",".join(f"{name}:{count}" for name, count in sorted(statuses.items()))
        )
        first["notes"] = "; ".join(note_parts)
        first["skip_reason"] = ";".join(
            _ordered_unique(row.get("skip_reason") for row in group)
        )
        first["sample_index"] = ""
        first["sample_count"] = str(expected_samples)
        first["aggregation"] = "arithmetic_mean"
        first["n_total"] = str(n_total)
        first["n_valid"] = str(n_valid)
        first["publishable"] = "true" if publishable else "false"
        first["mean_seconds"] = _number(mean)
        first["std_seconds"] = _number(std)
        first["variance_seconds2"] = _number(variance)
        first["cv"] = _number(cv)
        first["median_seconds"] = _number(median)
        first["min_seconds"] = _number(minimum)
        first["max_seconds"] = _number(maximum)
        first["ci95_low_seconds"] = _number(ci_low)
        first["ci95_high_seconds"] = _number(ci_high)
        first["mean_value"] = _number(mean)
        first["std_value"] = _number(std)
        first["variance_value2"] = _number(variance)
        first["median_value"] = _number(median)
        first["min_value"] = _number(minimum)
        first["max_value"] = _number(maximum)
        first["ci95_low_value"] = _number(ci_low)
        first["ci95_high_value"] = _number(ci_high)
        first["ci95_half_width_value"] = _number(ci_half_width)
        first["relative_std_percent"] = _number(relative_std_percent)
        first["relative_ci95_half_width_percent"] = _number(relative_ci95_percent)
        first["stability_class"] = stability_class
        aggregated.append(first)

    return aggregated
