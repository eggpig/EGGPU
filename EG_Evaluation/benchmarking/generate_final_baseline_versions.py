#!/usr/bin/env python3
"""Generate a concise, path-free baseline version table for the paper bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def latex(value):
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
    )


def python_row(name, role, package):
    return {
        "system": name,
        "role": role,
        "version": str(package.get("version", "unknown")),
        "commit": "",
        "artifact_scope": str(package.get("role", "Python package")),
    }


def gunrock_rows(executables):
    rows = []
    groups = (
        ("Gunrock maintained", "maintained-v2"),
        ("Gunrock legacy LCC", "legacy-v1"),
    )
    for label, generation in groups:
        selected = {
            name: record
            for name, record in executables.items()
            if record.get("application_generation") == generation
            and record.get("status") == "available"
        }
        if not selected:
            continue
        first = next(iter(selected.values()))
        version = str(first.get("upstream_release_lineage", "unknown"))
        if generation == "legacy-v1" and first.get("source_version"):
            # The legacy LCC binary is newer than the v1.2.0 tag.  Preserve
            # the exact git-describe string instead of collapsing it to the
            # nearest historical release label.
            version = str(first["source_version"])
        rows.append(
            {
                "system": label,
                "role": "GPU baseline",
                "version": version,
                "commit": str(first.get("source_commit", "")),
                "artifact_scope": ", ".join(sorted(selected)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-version-json", required=True, type=Path)
    parser.add_argument("--sygraph-result", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source = json.loads(args.main_version_json.read_text(encoding="utf-8"))
    packages = source["python_baselines"]
    snapshot = source.get("paper_repo_source_snapshot", {})
    easygraph = packages["easygraph"]
    rows = [
        {
            "system": "EGGPU",
            "role": "proposed GPU system",
            "version": f"EasyGraph {easygraph.get('version', 'unknown')} + paper snapshot",
            "commit": str(snapshot.get("digest", "")),
            "artifact_scope": "Python/C++/CUDA EGGPU implementation",
        },
        python_row("EasyGraph CPU", "CPU baseline", easygraph),
        python_row("EasyGraph C++", "C++ baseline", easygraph),
        python_row("NetworkX", "CPU baseline", packages["networkx"]),
        python_row("igraph", "CPU/C baseline", packages["igraph"]),
        python_row("nx-cugraph", "GPU backend baseline", packages["nx-cugraph"]),
    ]
    rows.extend(gunrock_rows(source.get("gunrock_executables", {})))

    sygraph_metadata = None
    sygraph_build = None
    if args.sygraph_result:
        metadata_path = args.sygraph_result / "metadata.json"
        build_path = (
            args.sygraph_result.parents[1]
            / "third_party"
            / "sygraph_build"
            / "build_manifest.json"
        )
        if metadata_path.is_file():
            sygraph_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # The default result path is below benchmarking/results, so prefer the
        # project-local build manifest discovered relative to this script.
        project_build = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "sygraph_build"
            / "build_manifest.json"
        )
        if project_build.is_file():
            build_path = project_build
        if build_path.is_file():
            sygraph_build = json.loads(build_path.read_text(encoding="utf-8"))
    if sygraph_metadata:
        rows.append(
            {
                "system": "SYgraph",
                "role": "GPU baseline (validated BFS only)",
                "version": "upstream HEAD qualification",
                "commit": str(sygraph_metadata.get("source_commit", "")),
                "artifact_scope": "native BFS wrapper; exact distance-matrix validation",
            }
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(output / "paper_table_baseline_versions.csv", index=False)
    concise = {
        "systems": rows,
        "python_runtime": packages.get("python", {}),
        "nx_cugraph_runtime_dependencies": source.get("runtime_dependencies", {}),
        "sygraph_build": sygraph_build,
    }
    (output / "paper_baseline_versions.json").write_text(
        json.dumps(concise, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        r"% Requires: booktabs, tabularx",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Versions of the evaluated systems. SYgraph is listed only if its native BFS qualification completed.}",
        r"\label{tab:baseline-versions}",
        r"\small",
        r"\begin{tabularx}{\columnwidth}{l l X}",
        r"\toprule",
        r"System & Version & Evaluated artifact \\",
        r"\midrule",
    ]
    for row in rows:
        commit = str(row["commit"])
        version = str(row["version"])
        if commit:
            version += f" ({commit[:10]})"
        lines.append(
            f"{latex(row['system'])} & {latex(version)} & "
            f"{latex(row['artifact_scope'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""])
    (output / "paper_table_baseline_versions.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(output / "paper_table_baseline_versions.csv")


if __name__ == "__main__":
    main()
