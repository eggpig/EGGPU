#!/usr/bin/env python3
"""Create a standalone review document for every final paper table."""

from __future__ import annotations

import argparse
from pathlib import Path


TABLES = (
    "paper_table_baseline_versions.tex",
    "paper_table_datasets_13.tex",
    "paper_table_main_compact_best_competitor.tex",
    "paper_table_main_compact_pairwise_speedup.tex",
    "paper_table_category_13_summary.tex",
    "paper_table_build_13_by_dataset.tex",
    "paper_table_e2e_13_centrality.tex",
    "paper_table_e2e_13_connectivity.tex",
    "paper_table_e2e_13_paths_and_spanning_trees.tex",
    "paper_table_e2e_13_structural_holes.tex",
    "paper_table_kernel_13_centrality.tex",
    "paper_table_kernel_13_connectivity.tex",
    "paper_table_kernel_13_paths_and_spanning_trees.tex",
    "paper_table_kernel_13_structural_holes.tex",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.asset_dir.resolve()
    missing = [name for name in TABLES if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"missing final table source(s): {missing}")

    body = []
    for name in TABLES:
        body.extend((f"\\input{{{name}}}", r"\clearpage"))
    source = "\n".join(
        (
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.55in]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{tabularx}",
            r"\usepackage{graphicx}",
            r"\usepackage[table]{xcolor}",
            r"\usepackage{longtable}",
            r"\usepackage{array}",
            r"\usepackage{multirow}",
            r"\usepackage{caption}",
            r"\captionsetup{font=small,labelfont=bf}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\renewcommand{\arraystretch}{1.04}",
            r"\makeatletter",
            r"\setlength{\@fptop}{0pt}",
            r"\makeatother",
            r"\begin{document}",
            r"\begin{center}",
            r"{\Large\bfseries EGGPU Final 13-Dataset Table Review}\par",
            r"\vspace{0.5em}",
            r"All values are generated from the audited final evidence bundle.",
            r"\end{center}",
            r"\clearpage",
            *body,
            r"\end{document}",
            "",
        )
    )
    output = root / "EGGPU_FINAL_TABLES_REVIEW_20260717.tex"
    output.write_text(source, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
