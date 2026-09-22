#!/usr/bin/env python3
"""Generate the authoritative Chinese index for the final EGGPU experiments.

All headline values and failure descriptions are read from finalized CSV/JSON
artifacts.  The guide intentionally contains no hand-maintained performance
number so rerunning a qualified cell cannot leave the prose stale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FIGURES = (
    ("整体性能", "category_time_by_baseline_3panel.png"),
    ("首次使用与稳态实际时间", "first_use_vs_steady_actual_time.png"),
    ("累计冷启动工作流", "cumulative_cold_start_workflow.png"),
    ("核心模块消融", "ablation_actual_time_memory_composite.png"),
    ("四个代表函数的规模扩展", "scaling_four_functions.png"),
    ("同图多函数复用案例", "z_case_study_same_graph_workflow.png"),
)


def fmt(value, digits=2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def seconds(value):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if value < 1.0e-3:
        return f"{value * 1.0e6:.2f} us"
    if value < 1.0:
        return f"{value * 1.0e3:.2f} ms"
    return f"{value:.3f} s"


def md_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["无。", ""]
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                value = ""
            values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def grouped_non_success(data: pd.DataFrame) -> list[str]:
    if data.empty:
        return ["所有被要求的实验均成功。", ""]
    lines = []
    keys = ["experiment", "baseline", "function", "measurement", "status", "reason"]
    data = data.fillna("")
    for experiment, experiment_rows in data.groupby("experiment", sort=True):
        lines.extend([f"### {experiment}", ""])
        grouped = experiment_rows.groupby(keys[1:], dropna=False, sort=True)
        for identity, rows in grouped:
            baseline, function, measurement, status, reason = identity
            datasets = sorted({item for item in rows["dataset"].astype(str) if item})
            subject = " / ".join(item for item in (baseline, function, measurement) if item)
            dataset_text = ", ".join(datasets) if datasets else "experiment-level"
            lines.append(
                f"- **{subject or 'experiment-level'}**: `{status}`; "
                f"datasets=[{dataset_text}]. {reason}"
            )
        lines.append("")
    return lines


def stability_summary(ledger: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_column = f"{metric}_raw_mean_seconds"
    std_column = f"{metric}_std_seconds"
    if mean_column not in ledger or std_column not in ledger:
        return pd.DataFrame(), pd.DataFrame()
    data = ledger[ledger["execution_status"].eq("ok")].copy()
    data[mean_column] = pd.to_numeric(data[mean_column], errors="coerce")
    data[std_column] = pd.to_numeric(data[std_column], errors="coerce")
    data = data[data[mean_column].gt(0) & data[std_column].notna()].copy()
    data["cv_percent"] = 100.0 * data[std_column] / data[mean_column]
    grouped = (
        data.groupby("baseline", as_index=False)
        .agg(
            measured_cells=("cv_percent", "size"),
            median_cv_percent=("cv_percent", "median"),
            p90_cv_percent=("cv_percent", lambda values: values.quantile(0.90)),
            max_cv_percent=("cv_percent", "max"),
        )
        .sort_values("baseline")
    )
    high = data[data["cv_percent"].gt(10.0)].copy()
    high = high.sort_values("cv_percent", ascending=False).head(25)
    return grouped, high


def cpu_host_load_anomalies(core_result: Path) -> pd.DataFrame:
    raw = core_result / "cpu_large_matrix" / "raw"
    rows = []
    if not raw.is_dir():
        return pd.DataFrame()
    for path in sorted(raw.glob("*_timing_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("status") != "ok":
            continue
        cpu_count = record.get("logical_cpu_count")
        loads = [
            record.get("host_loadavg_1m_start"),
            record.get("host_loadavg_1m_end"),
        ]
        loads = [float(value) for value in loads if value is not None]
        if not loads or not cpu_count:
            continue
        peak = max(loads)
        if peak <= 0.75 * float(cpu_count):
            continue
        rows.append(
            {
                "dataset": record.get("dataset", ""),
                "baseline": record.get("baseline", ""),
                "function": record.get("function", ""),
                "sample": record.get("timing_process_index", path.stem.rsplit("_", 1)[-1]),
                "E2E (s)": f"{float(record.get('e2e_seconds', 0.0)):.3f}",
                "peak 1-min load": f"{peak:.1f}",
                "logical CPUs": int(cpu_count),
                "source": path.name,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--core-result", required=True, type=Path)
    parser.add_argument("--followup-result", required=True, type=Path)
    args = parser.parse_args()

    assets = args.asset_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    relative_assets = Path(assets.name)

    summary = json.loads((assets / "final_13_numeric_summary.json").read_text())
    outcomes = json.loads((assets / "final_13_cell_outcome_summary.json").read_text())
    non_success_summary = json.loads(
        (assets / "all_experiment_non_success_summary.json").read_text()
    )
    datasets = pd.read_csv(assets / "paper_table_datasets_13.csv")
    baseline_versions = optional_csv(assets / "paper_table_baseline_versions.csv")
    categories = pd.read_csv(assets / "paper_table_category_13_summary.csv")
    compact = pd.read_csv(assets / "paper_table_main_compact_pairwise_speedup.csv")
    compact_best = pd.read_csv(
        assets / "paper_table_main_compact_best_competitor.csv"
    )
    pairwise = pd.read_csv(assets / "final_13_pairwise_sota_details.csv")
    failures = pd.read_csv(assets / "all_experiment_non_success_ledger.csv").fillna("")
    baseline_count = int(outcomes.get("baselines", 0))
    baseline_names = sorted(
        pd.read_csv(assets / "final_13_cell_outcome_ledger.csv", usecols=["baseline"])[
            "baseline"
        ].dropna().astype(str).unique()
    )
    ledger = pd.read_csv(
        assets / "final_13_cell_outcome_ledger.csv", low_memory=False
    ).fillna("")
    first_use = optional_csv(assets / "first_use_steady_actual_times.csv")
    cumulative = optional_csv(assets / "cumulative_workflow_aggregate.csv")
    case_study = optional_csv(assets / "case_study_call_summary.csv")
    ablation = optional_csv(assets / "ablation_actual_values.csv")
    nonpositive = optional_csv(assets / "ablation_nonpositive_mechanisms.csv")
    scaling = optional_csv(assets / "scaling_four_function_points.csv")
    memory = optional_csv(assets / "eggpu_memory_13_details.csv")
    targeted_decisions = optional_csv(
        assets / "TARGETED_EGGPU_SELECTION_DECISIONS.csv"
    )
    closeness_decisions = optional_csv(
        assets / "CURRENT_CLOSENESS_SELECTION_DECISIONS.csv"
    )
    closeness_validation = optional_csv(
        assets / "CURRENT_CLOSENESS_EXTERNAL_VALIDATION.csv"
    )
    sygraph_path = assets / "SYGRAPH_QUALIFICATION.json"
    sygraph = json.loads(sygraph_path.read_text()) if sygraph_path.is_file() else {}
    e2e_stability, e2e_high_variance = stability_summary(ledger, "e2e")
    kernel_stability, kernel_high_variance = stability_summary(ledger, "kernel")
    host_load_anomalies = cpu_host_load_anomalies(args.core_result.resolve())

    lines = [
        "# EGGPU 最终 13 图实验结果与证据索引",
        "",
        "本文档由最终结果自动生成。主表口径为：EGGPU 报告五次正式测量中的最好值，"
        "同时报告同五次测量的样本标准差；其他系统报告五次算术均值和样本标准差。"
        "Timing 与 memory 分轮执行，所有排名仅使用通过语义与正确性验证的结果。",
        "",
        "## 1. 实验闭环",
        "",
        f"1. 13 张图、16 个函数、{baseline_count} 个系统构成完整结果分类账；每个单元不是有效结果，"
        "就是带明确原因的 unsupported、timeout、representation limit、resource limit、"
        "semantic mismatch 或 execution error。",
        "2. Overall performance 同时报告 graph construction、steady-state E2E、kernel 和显存。",
        "3. First-use/steady-state 给出实际秒数；累计工作流比较 EGGPU、隔离调用、igraph 和"
        "严格 nx-cugraph；case study 只比较 EGGPU 自身有无同图复用。",
        "4. Ablation 控制 GraphContext、C++ graph cache、result reconstruction 和 CSR 等变量。",
        "5. Scaling 统一展示 PageRank、WCC、BFS、SSSP，在真实图和 R-MAT S20/S22/S24/S26"
        "上保留所有有效竞争结果。对数坐标不能包含数学上的零点，图中不绘制 slope 或拟合线。",
        f"6. 最终纳入的系统为：{', '.join(baseline_names)}。SYgraph 已成功构建，但"
        "原生 BFS、SSSP 和 BC 未通过严格正确性资格检查，因此不进入任何性能排名或"
        "scaling 曲线；这与编译失败明确区分。",
        "",
        "## 2. 测量与统计口径",
        "",
        "- **主实验范围**：13 datasets x 16 functions = 208 EGGPU workloads；每个"
        "baseline 只在原生函数语义可对齐时进入数值比较。",
        "- **稳态 E2E**：同图状态已经由预热调用建立后，从高层函数入口到返回兼容结果的"
        "完整时间；包含 dispatch、GPU 执行和结果重构，不包含单列的 graph construction。",
        "- **Kernel**：后端记录的算法计算时间。CPU 系统没有独立设备 kernel，表中对应其"
        "原生算法调用时间，不能解释为 CUDA kernel。",
        "- **首次使用**：从首个 GPU 函数调用开始，包含首次建立可复用 host/device 图状态；"
        "它与 steady state 分开报告，不能拿稳态值冒充用户第一次调用。",
        "- **重复与误差**：EGGPU 预热 2 次后正式测量 5 次，论文值取 5 次最好值并同时报告"
        "这 5 次的样本标准差；其他系统不预热，报告 5 次算术均值和样本标准差。",
        "- **显存**：另起 3 个隔离进程做 memory pass，计时轮不采样显存；可见性 marker 在"
        "正式 memory pass 中关闭，不从观察值里做事后猜测性扣除。",
        "- **超时**：单函数调用上限 100 s；graph load 采用独立上限。超时属于性能结果，"
        "不改变函数支持 T/P/F 的定义。",
        "- **正确性**：排名前必须通过图方向、源点、参数、返回形状和数值/分区验证；CPU"
        "fallback、native-cuGraph fallback 和错误语义不会进入 nx-cugraph 数字。",
        "- **坐标语义**：柱状图和累计时间图使用从 0 开始的线性轴；类别总览、逐函数"
        "first-use 细节和 scaling 跨越多个数量级，使用明确标注的对数轴。倍数只作为注释，"
        "不作为坐标轴。",
        "- **当前版本替换**：只有当前实现的五次样本完整、语义对齐且正确性通过时，"
        "才替换对应历史 EGGPU 单元；替换不以新版本是否更快为条件，也不跨版本混合样本。"
        f"主结果共审查 {len(targeted_decisions)} 个指标，其中 "
        f"{int(targeted_decisions.get('selected_source', pd.Series(dtype=str)).eq('current_qualified_retest').sum())} 个由当前合格版本替换。",
        "- **Closeness 补充验证**：4 张规模受控图均使用相同 16 个确定性源点与 EasyGraph CPU"
        f"逐值对齐；通过组数为 {int(closeness_validation.get('validation_status', pd.Series(dtype=str)).eq('pass').sum())}/4，"
        f"由当前合格版本替换的 E2E/kernel 指标为 {int(closeness_decisions.get('selected_source', pd.Series(dtype=str)).eq('current_qualified_retest').sum())}/8。",
        "- **制图制表**：遵循 WritingAIPaper 的 claim--evidence 对齐和自包含图注原则，以及"
        "Paper-Writing-Tips 的可读字号、矢量输出、统一配色和避免冗余信息原则；数据正确性"
        "优先于视觉装饰。参考："
        "[WritingAIPaper](https://github.com/hzwer/WritingAIPaper)，"
        "[Paper-Writing-Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)。",
        "",
        "## 3. 权威目录",
        "",
        f"- 两张规模锚点和全函数补齐：`{args.core_result.resolve()}`",
        f"- R-MAT、累计工作流和 CPU/GPU 对照：`{args.followup_result.resolve()}`",
        f"- 最终论文资产：`{assets}`",
        "- 完整逐单元分类账："
        f"[`final_13_cell_outcome_ledger.csv`]({relative_assets / 'final_13_cell_outcome_ledger.csv'})",
        "- 全实验非成功分类账："
        f"[`all_experiment_non_success_ledger.csv`]({relative_assets / 'all_experiment_non_success_ledger.csv'})",
        "",
        "## 4. 实验系统版本",
        "",
    ]
    lines.extend(md_table(baseline_versions))
    if sygraph:
        sygraph_tests = pd.DataFrame(sygraph.get("tests", []))
        sygraph_view = sygraph_tests[
            [
                column
                for column in (
                    "function",
                    "native_execution_status",
                    "validation_status",
                    "reason",
                )
                if column in sygraph_tests.columns
            ]
        ]
        lines.extend(
            [
                "### SYgraph 资格检查",
                "",
                f"源码 commit `{sygraph.get('source_commit', '')}`，build=`{sygraph.get('build_status', '')}`，"
                f"最终状态=`{sygraph.get('evaluation_status', '')}`。最小适配仅涉及结果访问器和"
                "AdaptiveCpp 兼容性，没有修改算法实现。",
                "",
            ]
        )
        lines.extend(md_table(sygraph_view))
    lines.extend(
        [
        "精简表不包含本机绝对路径；完整可复核信息保存在 "
        "`paper_baseline_versions.json` 和原始 run manifest 中。",
        "",
        "## 5. 数据集",
        "",
        ]
    )
    display_columns = [
        column for column in (
            "dataset", "nodes", "simple_edges", "csr_entries", "avg_degree",
            "max_degree", "density", "directed", "role"
        ) if column in datasets.columns
    ]
    lines.extend(md_table(datasets[display_columns]))

    lines.extend(
        [
            "## 6. Overall Performance",
            "",
            f"- E2E 成功 workload：**{summary['e2e_eggpu_successful_workloads']}/208**。",
            f"- E2E fastest/tied/only：**{summary['e2e_fastest_tied_or_only']}/"
            f"{summary['e2e_eggpu_successful_workloads']}**。",
            f"- E2E competitive wins：**{summary['e2e_competitive_wins']}/"
            f"{summary['e2e_competitive_pairs']}**。",
            f"- E2E 对 pairwise best competitor 的几何平均加速："
            f"**{fmt(summary['e2e_speedup_over_best_competitor'])}x**。",
            f"- E2E 对 pairwise best GPU baseline："
            f"**{fmt(summary['e2e_speedup_over_best_gpu'])}x** / "
            f"{summary['e2e_best_gpu_common_pairs']} common pairs。",
            f"- Kernel 成功 workload：**{summary['kernel_eggpu_successful_workloads']}/208**。",
            f"- Kernel fastest/tied/only：**{summary['kernel_fastest_tied_or_only']}/"
            f"{summary['kernel_eggpu_successful_workloads']}**。",
            f"- Kernel competitive wins：**{summary['kernel_competitive_wins']}/"
            f"{summary['kernel_competitive_pairs']}**。",
            f"- Kernel 对 pairwise best GPU baseline："
            f"**{fmt(summary['kernel_speedup_over_best_gpu'])}x** / "
            f"{summary['kernel_best_gpu_common_pairs']} common pairs。",
            f"- 有显存统计的 EGGPU 单元：**{summary.get('eggpu_memory_cells', 0)}**；"
            f"观测到的最大峰值：**{fmt(summary.get('eggpu_max_gpu_peak_mb'))} MiB**。",
            "",
            "这里的 only-supported 表示在当前函数语义和已评测 artifact 下没有其他通过验证的实现；"
            "它计入覆盖率，但不制造加速倍数。",
            "",
            "### 主矩阵完整性（按系统）",
            "",
        ]
    )
    status_table = (
        ledger.groupby(["baseline", "execution_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    status_table["total"] = status_table.drop(columns=["baseline"]).sum(axis=1)
    lines.extend(md_table(status_table))
    lines.extend(["### 四类函数", ""])
    category_view = categories.copy()
    for column in category_view.columns:
        if "speedup" in column:
            category_view[column] = category_view[column].map(
                lambda value: "N/A" if pd.isna(value) else f"{float(value):.2f}x"
            )
    lines.extend(md_table(category_view))
    lines.extend(["### 正文优先使用的紧凑函数表", ""])
    lines.extend(md_table(compact_best))
    lines.extend(["### 逐 baseline 的代表函数明细", ""])
    lines.extend(md_table(compact))
    lines.extend(["### EGGPU 未达到 fastest/tied 的 workload", ""])
    fastest = pairwise["fastest_tied_or_only"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    non_sota = pairwise[
        pairwise["eggpu_status"].eq("ok")
        & ~fastest
    ].copy()
    if not non_sota.empty:
        non_sota["EGGPU (s)"] = non_sota["eggpu_seconds"].map(
            lambda value: f"{float(value):.6g}"
        )
        non_sota["Best competitor (s)"] = non_sota["best_competitor_seconds"].map(
            lambda value: f"{float(value):.6g}"
        )
        non_sota["competitor/EGGPU"] = non_sota["speedup"].map(
            lambda value: f"{float(value):.3f}x"
        )
        lines.extend(
            md_table(
                non_sota[
                    [
                        "metric", "dataset", "function", "best_competitor",
                        "EGGPU (s)", "Best competitor (s)", "competitor/EGGPU",
                    ]
                ]
            )
        )
    else:
        lines.extend(["无。", ""])

    lines.extend(["### 五次测量的波动性", ""])
    lines.extend(
        [
            "变异系数（CV）定义为样本标准差除以五次测量均值。该表用于披露稳定性，"
            "不改变 EGGPU best-of-five 和其他系统 arithmetic-mean 的主表估计量。",
            "",
            "#### E2E CV",
            "",
        ]
    )
    lines.extend(md_table(e2e_stability.round(3)))
    lines.extend(["#### Kernel CV", ""])
    lines.extend(md_table(kernel_stability.round(3)))
    for title, data in (
        ("E2E CV > 10% 的最高波动单元", e2e_high_variance),
        ("Kernel CV > 10% 的最高波动单元", kernel_high_variance),
    ):
        lines.extend([f"#### {title}", ""])
        if data.empty:
            lines.extend(["无。", ""])
            continue
        view = data[["baseline", "dataset", "function", "cv_percent"]].copy()
        view["cv_percent"] = view["cv_percent"].map(lambda value: f"{value:.2f}%")
        lines.extend(md_table(view))
    lines.extend(["#### CPU 大图计时期间的高主机负载样本", ""])
    if host_load_anomalies.empty:
        lines.extend(["未观察到超过逻辑 CPU 数 75% 的 1 分钟 load average。", ""])
    else:
        lines.extend(
            [
                "这些样本保留原始时间和负载证据；它们会在定向复核后决定是否纳入最终均值，"
                "不会静默删除或用 EGGPU 数值替代。",
                "",
            ]
        )
        lines.extend(md_table(host_load_anomalies))

    lines.extend(["## 7. First Use 与累计冷启动", ""])
    lines.extend(
        [
            "这一组实验回答两个互补问题：first-use/steady-state 量化第一次函数调用的"
            "一次性代价；累计冷启动把 EGGPU 与 igraph、严格 nx-cugraph 放在同一条三函数"
            "工作流中比较用户总等待时间。实验最后的 case study 再只在 EGGPU 内部关闭"
            "同图复用，单独隔离复用机制本身的贡献。",
            "",
        ]
    )
    if not first_use.empty:
        first_view = first_use.copy()
        first_view["first-use"] = first_view["first_use_geomean_seconds"].map(seconds)
        first_view["steady-state"] = first_view["steady_state_geomean_seconds"].map(seconds)
        first_view["first/steady"] = first_view["first_use_over_steady"].map(
            lambda value: f"{float(value):.2f}x"
        )
        lines.extend(["### 7.1 First-use versus steady-state", ""])
        lines.extend(
            md_table(
                first_view[
                    ["family", "metric", "pairs", "first-use", "steady-state", "first/steady"]
                ]
            )
        )
    if not cumulative.empty:
        cumulative_view = cumulative.copy()
        cumulative_view["cumulative time"] = cumulative_view[
            "cumulative_geomean_seconds"
        ].map(seconds)
        lines.extend(["### 7.2 与竞争系统的累计冷启动", ""])
        lines.extend(
            md_table(
                cumulative_view[
                    ["baseline", "call_position", "function", "datasets", "cumulative time", "estimator"]
                ]
            )
        )
    lines.extend(["## 8. 核心模块消融", ""])
    lines.extend(
        [
            "主消融只保留具有稳定正向证据的模块，并以实际时间或内存作为坐标；倍数只做"
            "注释。GraphContext 与 C++ graph cache 在相同 dataset 和 workflow order 上一一配对，"
            "汇总 22 个受控 dataset-order 对；result reconstruction 使用相同函数和相同预热图，"
            "CSR storage/traversal 使用相同图输入。",
            "",
        ]
    )
    if not ablation.empty:
        ablation_view = ablation.copy()
        ablation_view["complete"] = ablation_view.apply(
            lambda row: f"{float(row['complete_value']):.4g} {str(row['axis']).split('(')[-1].rstrip(')')}",
            axis=1,
        )
        ablation_view["ablated/reference"] = ablation_view.apply(
            lambda row: f"{float(row['ablated_or_reference_value']):.4g} {str(row['axis']).split('(')[-1].rstrip(')')}",
            axis=1,
        )
        ablation_view["effect"] = ablation_view["ratio"].map(
            lambda value: f"{float(value):.2f}x"
        )
        lines.extend(
            md_table(
                ablation_view[
                    ["module", "axis", "complete_label", "complete", "reference_label", "ablated/reference", "effect", "cases"]
                ]
            )
        )
    if not nonpositive.empty:
        lines.extend(
            [
                "### 未作为主收益 claim 的机制",
                "",
                "以下机制保留为工程选择或边界观察，但当前聚合数据不足以支持正向性能 claim。",
                "",
            ]
        )
        nonpositive_view = nonpositive.copy()
        nonpositive_view["geomean_ratio"] = nonpositive_view["geomean_ratio"].map(
            lambda value: f"{float(value):.3f}x"
        )
        lines.extend(md_table(nonpositive_view))

    lines.extend(["## 9. Scaling 与资源边界", ""])
    lines.extend(
        [
            "Scaling 只选择 PageRank、WCC、BFS、SSSP 四个代表函数。每个点仍是完整 E2E，"
            "不是仅 kernel；实心/空心分别表示真实图/R-MAT。横纵轴因跨越约五个数量级而使用"
            "对数尺度，零点在数学上无定义，所以这是全文唯一不从零开始的核心图。",
            "",
        ]
    )
    if not scaling.empty:
        scale_rows = []
        for (function, baseline), part in scaling.groupby(
            ["function", "baseline"], sort=True
        ):
            part = part.sort_values("csr_entries")
            largest = part.iloc[-1]
            scale_rows.append(
                {
                    "function": function,
                    "baseline": baseline,
                    "successful graphs": len(part),
                    "largest graph": largest["dataset"],
                    "largest CSR entries": f"{int(largest['csr_entries']):,}",
                    "E2E at largest": seconds(largest["e2e_seconds"]),
                    "estimator": largest["estimator"],
                }
            )
        lines.extend(md_table(pd.DataFrame(scale_rows)))
    scale_eggpu = ledger[
        ledger["dataset"].isin(["com-Orkut", "GAP-twitter"])
        & ledger["baseline"].eq("EGGPU")
    ]
    if not scale_eggpu.empty:
        ok_by_dataset = (
            scale_eggpu[scale_eggpu["execution_status"].eq("ok")]
            .groupby("dataset")["function"]
            .apply(lambda values: ", ".join(values.astype(str)))
        )
        lines.extend(["### 两张超大真实图上的 EGGPU 成功边界", ""])
        for dataset, functions in ok_by_dataset.items():
            lines.append(f"- **{dataset}**: {functions}。")
        lines.append("")

    if not memory.empty:
        memory_view = memory.copy()
        memory_view["gpu peak"] = pd.to_numeric(
            memory_view["gpu_peak_mb_mean"], errors="coerce"
        ).map(lambda value: "N/A" if pd.isna(value) else f"{float(value):.1f} MiB")
        memory_view = memory_view.sort_values("gpu_peak_mb_mean", ascending=False).head(12)
        lines.extend(["### 峰值显存最高的 EGGPU workload", ""])
        lines.extend(md_table(memory_view[["dataset", "function", "gpu peak", "memory_sample_count"]]))

    lines.extend(["## 10. Case Study：同图三函数分析工作流", ""])
    lines.extend(
        [
            "该实验只比较 EGGPU 自身的两种执行方式：三个函数分别在隔离图状态下调用，"
            "以及在同一张未修改图上连续调用。它验证复用主要降低 E2E 中的图提取、表示构建"
            "和传输，而不会把 CUDA kernel 本身伪装成更快。",
            "",
        ]
    )
    if not case_study.empty:
        case_view = case_study.copy()
        case_view["isolated"] = case_view["isolated_geomean_seconds"].map(seconds)
        case_view["same-graph workflow"] = case_view[
            "same_graph_workflow_geomean_seconds"
        ].map(seconds)
        case_view["reuse benefit"] = case_view["isolated_over_workflow"].map(
            lambda value: f"{float(value):.2f}x"
        )
        lines.extend(
            md_table(
                case_view[
                    ["function", "call_position", "metric", "datasets", "isolated", "same-graph workflow", "reuse benefit"]
                ]
            )
        )

    lines.extend(["## 11. 论文图", ""])
    for title, filename in FIGURES:
        if (assets / filename).is_file():
            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"![{title}]({relative_assets / filename})",
                    "",
                ]
            )

    lines.extend(
        [
            "## 12. 完整表格与原始数字",
            "",
            f"- 全部 TeX 表格编译审阅稿：[`EGGPU_FINAL_TABLES_REVIEW_20260717.pdf`]({relative_assets / 'EGGPU_FINAL_TABLES_REVIEW_20260717.pdf'})",
            f"- 数据集表：[`paper_table_datasets_13.tex`]({relative_assets / 'paper_table_datasets_13.tex'})",
            f"- Baseline 版本表：[`paper_table_baseline_versions.tex`]({relative_assets / 'paper_table_baseline_versions.tex'})",
            f"- Build 表：[`paper_table_build_13_by_dataset.tex`]({relative_assets / 'paper_table_build_13_by_dataset.tex'})",
            f"- 类别表：[`paper_table_category_13_summary.tex`]({relative_assets / 'paper_table_category_13_summary.tex'})",
            f"- 正文紧凑表：[`paper_table_main_compact_best_competitor.tex`]({relative_assets / 'paper_table_main_compact_best_competitor.tex'})",
            f"- E2E 全表 CSV：[`paper_table_e2e_13_full.csv`]({relative_assets / 'paper_table_e2e_13_full.csv'})",
            f"- Kernel 全表 CSV：[`paper_table_kernel_13_full.csv`]({relative_assets / 'paper_table_kernel_13_full.csv'})",
            f"- Pairwise SOTA 明细：[`final_13_pairwise_sota_details.csv`]({relative_assets / 'final_13_pairwise_sota_details.csv'})",
            f"- 显存明细：[`eggpu_memory_13_details.csv`]({relative_assets / 'eggpu_memory_13_details.csv'})",
            f"- 最终资产审计：[`FINAL_13_ASSET_AUDIT.md`]({relative_assets / 'FINAL_13_ASSET_AUDIT.md'})",
            "",
            "## 13. 所有非成功结果及原因",
            "",
            f"最终主分类账未分类单元：**{outcomes.get('missing_experiment_cells', 'N/A')}**；"
            f"全部实验非成功记录：**{non_success_summary['non_success_rows']}**。"
            "以下内容按相同原因聚合数据集，逐单元原始记录保存在 CSV 中。",
            "",
        ]
    )
    lines.extend(grouped_non_success(failures))
    lines.extend(
        [
            "## 14. 论文使用边界",
            "",
            "- 不把 timeout、OOM、表示上限或语义不一致写成函数不支持；支持矩阵与性能结果分开。",
            "- 不把无竞争实现的单元写成数值 speedup。",
            "- GAP-twitter 上仅对成功函数声称十亿边级执行能力；不能声称 16 个函数都突破十亿边。",
            "- 类别总览、逐函数 first-use 细节和 Scaling 图采用对数坐标以容纳多个数量级；"
            "数学上的零点不能出现在这些轴上。其余时间、显存和累计工作流面板从零开始。",
            "- First-use 缺失的规模保护 Closeness 样本、消融 timeout 和 baseline 失败必须按本"
            "文档列出的原因披露，不得留空或用其他后端替代。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
