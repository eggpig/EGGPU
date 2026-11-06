"""Render a four-request scene and actual ready-session times, or an honest pending panel.

Input: first_page_trace_spec.json while pending, then the unmodified receipt from
experiments/campaign_20261104/first_pass_20261106/timing/first_page_trace.py.
This program never executes a GPU job or creates example performance numbers.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math


ARMS = [("Full upload + LRU", "persistent_full", "closure_lru"),
        ("EGGPU", "lease_sparse", "closure")]
REQUESTS = [("IN", 0), ("IN", 1), ("UK", 0), ("IN", 1)]
COLORS = {"baseline": "#7F8C99", "eggpu": "#2166AC", "ink": "#262B30",
          "graph_g": "#DDE9EF", "graph_h": "#E6E0EC", "arrow": "#68747C"}


def binding(path):
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            label + " must be numeric")
    require(math.isfinite(value) and (value > 0 if positive else value >= 0),
            label + " must be finite and " + ("positive" if positive else "nonnegative"))
    return value


def interval(value, label):
    require(isinstance(value, list) and len(value) == 2, label + " must have two bounds")
    start, stop = [numeric(x, label) for x in value]
    require(stop >= start, label + " has reversed bounds")
    return start, stop


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6),
            label + " disagrees with its measured intervals")


def read_evidence(path):
    record = json.loads(path.read_text())
    if record.get("schema") == "eggpu.first-page.public-figure-input.v1":
        require(record.get("source_sha256") == "71ea757d372f8c1b969ea074caaa27d3a77d4cea6d48cd9c6ae9c4b02936d7ce", "Public figure source differs")
        require(record.get("gpu_executed") is True and record.get("quality_pass") is True and record.get("gpu_count") == 8 and record.get("pressure_claim_supported") is True, "Incomplete public trace evidence")
        require(record.get("projection_verified_against_original_receipt") is True and len(record["raw_receipt"]["sha256"]) == 64, "Missing original receipt binding")
        evidence = record["evidence"]
        require(evidence["status"] == "PASS_MEASURED_FIRST_PAGE" and evidence["n"] == 1 and len(evidence["arms"]) == 2, "Wrong measured figure scope")
        for expected, arm in zip(ARMS, evidence["arms"]):
            require(tuple(arm[k] for k in ("label", "route", "policy")) == expected, "Wrong figure policy identity")
            start, stop = interval(arm["ready_session_interval_ns"], "primary")
            close(arm["ready_session_seconds"], (stop-start)/1e9, "primary interval")
            close(sum(arm["phases"].values()), arm["total_seconds"], "four-request total")
            parts = arm["ready_session_breakdown"]
            close(sum(parts[k] for k in ("host_preparation_seconds", "device_execution_seconds", "other_seconds")), arm["ready_session_seconds"], "mutually exclusive main phases")
            close(parts["host_preparation_seconds"]/arm["ready_session_seconds"], parts["host_preparation_fraction"], "preparation fraction")
            require([r["stage"] for r in arm["requests"]] == ["G0", "G1", "H", "G1 revisit"], "Wrong request sequence")
        return evidence

    if record.get("schema") == "eggpu.first-page.figure-input.v1":
        require(record.get("status") == "MEASUREMENT_PENDING", "Unknown figure-input status")
        require(record.get("measurement_receipt") is None and not record.get("gpu_executed"),
                "Pending specification must not masquerade as a measurement")
        require(all("total_seconds" not in arm for arm in record.get("arms", [])),
                "Pending input must not contain invented times")
        return {"status": "MEASUREMENT_PENDING", "n": 0, "arms": []}

    require(record.get("schema") == "eggpu.first-page.four-request.v1", "Unsupported measurement schema")
    require(record.get("status") == "PASS_FIRST_PAGE_TRACE", "Trace is not complete and passing")
    require(record.get("gpu_executed") is True and record.get("quality_pass") is True,
            "Actual GPU execution and complete-output checks are required")
    require(record.get("pressure_claim_supported") is True, "Budget-pressure claim is unsupported")
    require(record.get("source_binding") and record.get("inputs"), "Missing source/input identity")
    require(len(set(record.get("gpu_uuids", []))) == 8, "The scene requires eight distinct GPUs")
    require(record.get("devices") == list(range(8)), "Unexpected logical device placement")
    budget = {str(k): v for k, v in record["budget"]["per_device"].items()}
    require(set(budget) == set(map(str, range(8))) and all(v > 0 for v in budget.values()),
            "Need positive measured budgets for all eight devices")
    require(record["budget"].get("selection_uses_timing") is False,
            "Retention budgets must be fixed without selecting timings")
    raw_arms = record.get("arms", [])
    require(len(raw_arms) == 2, "Exactly two measured arms are required")
    evidence = {"status": "PASS_MEASURED_FIRST_PAGE", "n": 1, "arms": [],
                "source_binding": record["source_binding"], "inputs": record["inputs"],
                "gpu_uuids": record["gpu_uuids"], "budget": record["budget"],
                "timing_scope": record.get("timing_scope", ""),
                "formal_performance_claim": record.get("formal_performance_claim", False),
                "main_timing_scope": "Continuous wall time from the weight-update commit start to the final query return, after the initial G0 request has completed. Includes maintenance, intervening bookkeeping, retention and complete owning results. Original four-request totals remain supplementary.",
                "timing_choice": "Ready-session window fixed at the user's ready-input instruction before the second arm result returned; no timing-based selection."}
    paired_sources = []
    for expected, arm in zip(ARMS, raw_arms):
        require(tuple(arm.get(k) for k in ("label", "route", "policy")) == expected,
                "Unexpected policy identity or arm ordering")
        require({str(k): v for k, v in arm["budget"].items()} == budget,
                "Paired arms must use the same per-device budgets")
        checks = arm.get("checks", [])
        require(arm.get("quality_pass") is True and len(checks) == 4 and
                all(c["quality"]["passed"] is True and c.get("output") and
                    c.get("survives_session_close") is True for c in checks),
                "All four complete results must pass and survive session closure")
        events = arm.get("events", [])
        require([(e.get("dataset"), e.get("version")) for e in events] == REQUESTS,
                "The figure requires G0, G1, H, then the G1 revisit")
        require(all(e.get("function") == "sssp" for e in events), "All four requests must be SSSP")
        require(arm.get("H_pressure_observed") is True and events[2].get("pressured_devices_before_retention")
                and numeric(events[2]["actual_released_bytes"], "H actual release", True),
                "The H request must show actual pressure and physical release")
        sources = [e["source"] for e in events]
        require(sources[0] == sources[1] == sources[3], "IN source must remain fixed")
        paired_sources.append(sources)
        begin, end = interval(arm["continuous_interval_ns"], "session")
        total = numeric(arm["total_seconds"], "total_seconds", True)
        close(total, (end - begin) / 1e9, "total_seconds")
        setup = interval(arm["setup_interval_ns"], "session construction")
        commit = interval(arm["commit"]["interval_ns"], "weight commit")
        require(setup[0] == begin and begin <= setup[1] <= end, "Setup escapes continuous session")
        previous = setup[1]
        query_sum = retention_sum = 0.0
        for index, event in enumerate(events):
            start, stop = interval(event["complete_query_interval_ns"], "complete query")
            left, right = interval(event["retention_interval_ns"], "post-result retention")
            require(previous <= start <= left <= right <= stop <= end,
                    "Measured query/retention intervals overlap or escape the session")
            if index == 1:
                require(previous <= commit[0] <= commit[1] <= start,
                        "Weight commit must be between G0 and G1 queries")
            expected_outside = [[start, left], [right, stop]]
            require(event["query_excluding_retention_intervals_ns"] == expected_outside,
                    "Query-excluding-retention bounds are inconsistent")
            close(event["complete_query_seconds"], (stop - start) / 1e9, "complete query")
            close(event["retention_seconds"], (right - left) / 1e9, "retention")
            close(event["query_excluding_retention_seconds"],
                  ((left - start) + (stop - right)) / 1e9, "query excluding retention")
            query_sum += event["query_excluding_retention_seconds"]
            retention_sum += event["retention_seconds"]
            previous = stop
        close(arm["setup_update_seconds"],
              ((setup[1] - setup[0]) + (commit[1] - commit[0])) / 1e9, "setup/update")
        close(arm["query_excluding_retention_seconds"], query_sum, "summed query time")
        close(arm["retention_seconds"], retention_sum, "summed retention time")
        phase_keys = ("setup_update_seconds", "query_excluding_retention_seconds",
                      "retention_seconds", "bookkeeping_seconds")
        phases = {key: numeric(arm[key], key) for key in phase_keys}
        close(sum(phases.values()), total, "phase sum")
        ready_stop = events[-1]["complete_query_interval_ns"][1]
        ready_seconds = numeric((ready_stop - commit[0]) / 1e9, "ready_session_seconds", True)
        ready_breakdown = None
        if all("host_phase_wall_seconds" in e["metrics"] for e in events[1:]):
            host_preparation = (commit[1] - commit[0]) / 1e9
            device_execution = 0.0
            for event in events[1:]:
                metrics = event["metrics"]
                host_phases = metrics["host_phase_wall_seconds"]
                require("host_plan" in host_phases and "cooperative_gang" in host_phases,
                        "Unknown native phase ledger for the ready-session breakdown")
                phase_sum = sum(numeric(v, "native host phase") for v in host_phases.values())
                close(phase_sum, metrics["native_outer_seconds"], "native phase ledger")
                require(phase_sum <= event["query_excluding_retention_seconds"] + 1e-6,
                        "Native phases exceed the enclosing query-excluding-retention wall time")
                host_preparation += host_phases["host_plan"]
                device_execution += host_phases["cooperative_gang"]
            other = numeric(ready_seconds - host_preparation - device_execution,
                            "ready-session remaining wall time")
            ready_breakdown = {"host_preparation_seconds": host_preparation,
                "device_execution_seconds": device_execution, "other_seconds": other,
                "host_preparation_fraction": host_preparation / ready_seconds,
                "definition": "Host preparation = measured public commit + later native host_plan phases. Device execution = later cooperative_gang wall phases, including upload, communication and results, not pure kernels. Other = continuous main window minus these disjoint wall intervals; nested CUDA/per-rank timings are not added.",
                "native_phase_source": record["source_binding"]["files"].get(
                    "cpp_easygraph/runtime/egmgpu_partitioned_sssp.cpp")}
        evidence["arms"].append({"label": expected[0], "route": expected[1], "policy": expected[2],
                                 "ready_session_seconds": ready_seconds,
                                 "ready_session_interval_ns": [commit[0], ready_stop],
                                 "total_seconds": total, "phases": phases,
                                 "ready_session_breakdown": ready_breakdown,
                                 "commit_seconds": (commit[1] - commit[0]) / 1e9,
                                 "requests": [{"stage": e["stage"],
                                    "complete_query_seconds": e["complete_query_seconds"],
                                    "retention_seconds": e["retention_seconds"],
                                    "actual_released_bytes": e["actual_released_bytes"]}
                                    for e in events]})
    require(paired_sources[0] == paired_sources[1], "Paired arms have different SSSP sources")
    return evidence


def request_scene(ax, fontsize):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ys = [.80, .56, .32, .08]
    names = [r"$G^0$  SSSP", r"$G^1$  SSSP", r"$H$  SSSP", r"$G^1$  SSSP"]
    for index, (y, label) in enumerate(zip(ys, names)):
        ax.add_patch(FancyBboxPatch((.06, y), .65, .14,
            boxstyle="round,pad=0.009,rounding_size=0.018",
            facecolor=COLORS["graph_h" if index == 2 else "graph_g"],
            edgecolor=COLORS["ink"], linewidth=.65))
        ax.text(.385, y + .07, label, ha="center", va="center", fontsize=fontsize)
    for i, label in enumerate(["Weight update", "Switch graph", "Revisit"]):
        top, bottom = ys[i] - .012, ys[i + 1] + .151
        ax.add_patch(FancyArrowPatch((.385, top), (.385, bottom),
                     arrowstyle="-|>", mutation_scale=7, linewidth=.7, color=COLORS["arrow"]))
        ax.text(.47, (top + bottom) / 2, label, ha="left", va="center", fontsize=fontsize - .7,
                color=COLORS["ink"])
    # The initial G0 query is outside the ready-session clock. The bracket begins
    # at the weight-update transition and ends after the last complete query.
    ax.plot([.035, .005, .005, .035], [.755, .755, .072, .072],
            color=COLORS["arrow"], linewidth=.7, clip_on=False)
    ax.text(-.038, .414, "Timed", rotation=90, ha="center", va="center",
            fontsize=fontsize - .8, color=COLORS["arrow"], clip_on=False)
    ax.text(.385, -.015, r"$G$: IN; $H$: UK", ha="center", va="top", fontsize=fontsize - .4)


def time_panel(ax, evidence, fontsize):
    import matplotlib.ticker as ticker
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(.6)
    ax.set_xlim(-.6, 1.6)
    ax.set_xticks([0, 1], ["Full upload\n+ LRU", "EGGPU"])
    ax.tick_params(axis="x", length=0, labelsize=fontsize - .4, pad=5)
    ax.tick_params(axis="y", labelsize=fontsize - .3, width=.6, length=2.5)
    ax.set_ylabel("Time (s)", fontsize=fontsize, labelpad=3)
    if evidence["status"] == "MEASUREMENT_PENDING":
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(.5, .53, "Measurement\npending", ha="center", va="center",
                transform=ax.transAxes, fontsize=fontsize, color="#626B73", linespacing=1.45)
        return
    values = [arm["ready_session_seconds"] for arm in evidence["arms"]]
    phases = [("host_preparation_seconds", "Host preparation", "#2166AC", None),
              ("device_execution_seconds", "Device execution", "#E6A64C", "///"),
              ("other_seconds", "Other", "#CCD1D5", "...")]
    for x, arm, value in zip([0, 1], evidence["arms"], values):
        ledger = arm["ready_session_breakdown"]
        require(ledger is not None, "Measured disjoint phase ledger is required")
        bottom = 0.0
        for key, label, color, hatch in phases:
            ax.bar(x, ledger[key], bottom=bottom, width=.56, color=color,
                   edgecolor=COLORS["ink"], linewidth=.45, hatch=hatch,
                   label=label if x == 0 else None, zorder=3)
            bottom += ledger[key]
        close(bottom, value, "Plotted phase sum")
        ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize)
    ax.set_ylim(0, max(values) * 1.20)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, min_n_ticks=3))
    ax.grid(axis="y", color="#DDDFE1", linewidth=.45, zorder=0)
    ax.set_axisbelow(True)


def render(output, width_mm, evidence):
    import matplotlib.pyplot as plt
    wide = width_mm > 100
    height_mm = 123 if not wide else 73
    fontsize = 7.5 if not wide else 9.0
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    if wide:
        left = fig.add_axes([.05, .20, .435, .68])
        right = fig.add_axes([.63, .26, .34, .59])
        fig.text(.03, .955, "(a) Request sequence", ha="left", va="top", fontsize=fontsize)
        fig.text(.565, .955, "(b) Session time", ha="left", va="top", fontsize=fontsize)
    else:
        left = fig.add_axes([.10, .565, .82, .36])
        right = fig.add_axes([.20, .17, .74, .28])
        fig.text(.04, .965, "(a) Request sequence", ha="left", va="top", fontsize=fontsize)
        fig.text(.04, .485, "(b) Session time", ha="left", va="top", fontsize=fontsize)
    request_scene(left, fontsize)
    time_panel(right, evidence, fontsize)
    handles, labels = right.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .015),
                   ncol=3 if wide else 2, frameon=False, fontsize=fontsize - 1,
                   handlelength=1.5, columnspacing=1.2)
    files = []
    for suffix in ("pdf", "svg", "png"):
        path = output / f"first_page_problem_{width_mm}mm.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        files.append(path)
    plt.close(fig)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = read_evidence(args.input)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "dejavuserif", "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"], "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"]})
    args.output.mkdir(parents=True, exist_ok=True)
    files = render(args.output, 85, evidence) + render(args.output, 175, evidence)
    receipt = {"schema": "eggpu.first-page.figure-provenance.v2",
        "status": evidence["status"], "performance_measurement": evidence["n"] > 0,
        "gpu_executed_by_renderer": False, "input": binding(args.input),
        "generator": binding(Path(__file__)), "evidence": evidence,
        "files": {path.name: binding(path) for path in files}}
    (args.output / "figure_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "n": evidence["n"],
                      "files": len(files), "arms": evidence["arms"]}))


if __name__ == "__main__":
    main()
