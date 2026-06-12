#!/usr/bin/env python3
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import threading
import time

from pathlib import Path

import pandas as pd

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

_NVML_INITIALIZED = False
TRUE_VALUES = {"1", "TRUE", "ON", "YES"}
FUNCTION_ORDER = (
    "PageRank",
    "MST",
    "LCC",
    "WCC",
    "SCC",
    "BFS",
    "Dijkstra",
    "BellmanFord",
    "SSSP",
    "KCore",
    "BC",
    "Closeness",
    "EffectiveSize",
    "Efficiency",
    "Constraint",
    "Hierarchy",
)
PATH_SOURCE_FUNCTIONS = {"BFS", "Dijkstra", "BellmanFord", "SSSP"}
STRUCTURAL_HOLE_FUNCTIONS = {"EffectiveSize", "Efficiency", "Constraint", "Hierarchy"}
LEGACY_FUNCTION_ALIASES = {"CC": ("WCC", "SCC")}
STRICT_VALIDATION = os.environ.get("EGGPU_STRICT_VALIDATION", "").strip().upper() in TRUE_VALUES
DETAIL_DIR = os.environ.get("EGGPU_VALIDATION_DETAIL_DIR", "").strip()


def _ensure_nvml():
    global _NVML_INITIALIZED
    if pynvml is None:
        return False
    if _NVML_INITIALIZED:
        return True
    try:
        pynvml.nvmlInit()
        _NVML_INITIALIZED = True
        return True
    except Exception:
        return False


def _resolve_monitor_gpu_index():
    env_idx = os.environ.get("EGGPU_MONITOR_GPU_INDEX", "").strip()
    if env_idx.isdigit():
        return int(env_idx)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        tok = cvd.split(",")[0].strip()
        if tok.isdigit():
            return int(tok)
    return 0


def _nvml_compute_processes(handle):
    if pynvml is None:
        return []
    fn_names = (
        "nvmlDeviceGetComputeRunningProcesses_v3",
        "nvmlDeviceGetComputeRunningProcesses_v2",
        "nvmlDeviceGetComputeRunningProcesses",
    )
    for name in fn_names:
        fn = getattr(pynvml, name, None)
        if fn is None:
            continue
        try:
            procs = fn(handle)
            return procs if procs is not None else []
        except Exception:
            continue
    return []


def _safe_used_gpu_memory_bytes(proc_info):
    try:
        used = int(getattr(proc_info, "usedGpuMemory", 0))
    except Exception:
        return 0
    # Some NVML versions may return "not available" as a huge sentinel value.
    if used < 0 or used >= (1 << 62):
        return 0
    return used


def visibility_marker_adjust_mb():
    if os.environ.get("EGGPU_GPU_VISIBILITY_MARKER", "").strip().upper() not in TRUE_VALUES:
        return 0.0
    raw = os.environ.get("EGGPU_GPU_VISIBILITY_MARKER_ADJUST_MB", "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return 0.0
    return max(0.0, value)


def load_graph(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/c":
                continue
            p = s.split()
            if len(p) < 2:
                continue
            try:
                u, v = int(p[0]), int(p[1])
            except ValueError:
                continue
            rows.append((u, v))
    if not rows:
        raise SystemExit(f"empty/invalid graph: {path}")

    raw = pd.DataFrame(rows, columns=["src", "dst"])

    def normalize(pdf, include_self_loop_vertices):
        if not include_self_loop_vertices:
            pdf = pdf[pdf["src"] != pdf["dst"]].copy()
        cats = pd.Categorical(pd.concat([pdf["src"], pdf["dst"]], ignore_index=True))
        uniq = pd.Index(cats.categories)
        remap = {int(x): i for i, x in enumerate(uniq)}
        out = pdf.copy()
        out["src"] = out["src"].map(remap).astype("int32")
        out["dst"] = out["dst"].map(remap).astype("int32")
        out = out[out["src"] != out["dst"]].drop_duplicates().reset_index(drop=True)

        lo = out[["src", "dst"]].min(axis=1)
        hi = out[["src", "dst"]].max(axis=1)
        undir = pd.DataFrame({"src": lo, "dst": hi}).drop_duplicates().reset_index(drop=True)
        n = int(uniq.size)
        undir["weight"] = (1 + (undir["src"].astype("int64") * undir["dst"].astype("int64")) % n).astype("int32")
        return n, out, undir

    # All algorithm edge lists drop self-loops.  "all_vertices" still keeps
    # loop-only vertices in ID space for CC/MST consistency, but does not expose
    # self-loop edges to any baseline.
    n_clean, directed_clean, undirected_clean = normalize(raw, include_self_loop_vertices=False)
    n_all, directed_all, undirected_all = normalize(raw, include_self_loop_vertices=True)
    return {
        "clean": (n_clean, directed_clean, undirected_clean),
        "all_vertices": (n_all, directed_all, undirected_all),
    }


def sync_gpu():
    allow = os.environ.get("EGGPU_ALLOW_CUDA_SYNC", "").strip().upper() in {
        "1",
        "TRUE",
        "ON",
        "YES",
    }
    if not allow and os.environ.get("EASYGRAPH_ENABLE_GPU", "").strip().upper() not in {
        "1",
        "TRUE",
        "ON",
        "YES",
    }:
        return
    try:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def barrier(cooldown):
    sync_gpu()
    gc.collect()
    if cooldown > 0:
        time.sleep(cooldown)
    sync_gpu()


class PeakMemoryMonitor:
    def __init__(self, poll_seconds=0.01):
        self.poll_seconds = max(0.002, float(poll_seconds))
        self._stop = threading.Event()
        self._thread = None
        self._proc = None
        self._gpu_handle = None
        self._gpu_index = None
        self.peak_rss_bytes = 0
        self.peak_gpu_bytes = 0
        self.sum_gpu_bytes = 0
        self.num_gpu_samples = 0
        self.start_gpu_bytes = None
        self.peak_gpu_delta_bytes = 0
        self.sum_gpu_delta_bytes = 0
        self.num_gpu_delta_samples = 0
        self.peak_gpu_proc_bytes = 0
        self.sum_gpu_proc_bytes = 0
        self.num_gpu_proc_samples = 0
        self.start_gpu_proc_bytes = None
        self.peak_gpu_proc_delta_bytes = 0
        self.sum_gpu_proc_delta_bytes = 0
        self.num_gpu_proc_delta_samples = 0
        if psutil is not None:
            try:
                self._proc = psutil.Process(os.getpid())
                self.peak_rss_bytes = int(self._proc.memory_info().rss)
            except Exception:
                self._proc = None
        if _ensure_nvml():
            try:
                self._gpu_index = _resolve_monitor_gpu_index()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            except Exception:
                self._gpu_handle = None
                self._gpu_index = None

    def _process_tree_pids(self):
        pids = {os.getpid()}
        if self._proc is None:
            return pids
        try:
            pids.add(int(self._proc.pid))
        except Exception:
            pass
        try:
            for ch in self._proc.children(recursive=True):
                try:
                    pids.add(int(ch.pid))
                except Exception:
                    pass
        except Exception:
            pass
        return pids

    def _sample_once(self):
        if self._proc is not None:
            rss = 0
            procs = [self._proc]
            try:
                procs.extend(self._proc.children(recursive=True))
            except Exception:
                pass
            for p in procs:
                try:
                    rss += int(p.memory_info().rss)
                except Exception:
                    pass
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
        if self._gpu_handle is not None:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                used = int(info.used)
                if self.start_gpu_bytes is None:
                    self.start_gpu_bytes = used
                if used > self.peak_gpu_bytes:
                    self.peak_gpu_bytes = used
                self.sum_gpu_bytes += used
                self.num_gpu_samples += 1
                delta = used - self.start_gpu_bytes
                if delta < 0:
                    delta = 0
                if delta > self.peak_gpu_delta_bytes:
                    self.peak_gpu_delta_bytes = delta
                self.sum_gpu_delta_bytes += delta
                self.num_gpu_delta_samples += 1
            except Exception:
                pass
            try:
                pids = self._process_tree_pids()
                proc_used = 0
                for pinfo in _nvml_compute_processes(self._gpu_handle):
                    try:
                        pid = int(getattr(pinfo, "pid", -1))
                    except Exception:
                        pid = -1
                    if pid in pids:
                        proc_used += _safe_used_gpu_memory_bytes(pinfo)
                if self.start_gpu_proc_bytes is None:
                    self.start_gpu_proc_bytes = proc_used
                if proc_used > self.peak_gpu_proc_bytes:
                    self.peak_gpu_proc_bytes = proc_used
                self.sum_gpu_proc_bytes += proc_used
                self.num_gpu_proc_samples += 1
                proc_delta = proc_used - self.start_gpu_proc_bytes
                if proc_delta < 0:
                    proc_delta = 0
                if proc_delta > self.peak_gpu_proc_delta_bytes:
                    self.peak_gpu_proc_delta_bytes = proc_delta
                self.sum_gpu_proc_delta_bytes += proc_delta
                self.num_gpu_proc_delta_samples += 1
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.poll_seconds)

    def start(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.05, 2 * self.poll_seconds))
        self._sample_once()
        gpu_peak_mb = None
        gpu_avg_mb = None
        gpu_peak_delta_mb = None
        gpu_avg_delta_mb = None
        gpu_proc_peak_mb = None
        gpu_proc_avg_mb = None
        gpu_proc_peak_delta_mb = None
        gpu_proc_avg_delta_mb = None
        if self.num_gpu_samples > 0:
            gpu_peak_mb = self.peak_gpu_bytes / (1024.0 * 1024.0)
            gpu_avg_mb = (self.sum_gpu_bytes / self.num_gpu_samples) / (1024.0 * 1024.0)
            marker_adjust_mb = visibility_marker_adjust_mb()
            if marker_adjust_mb > 0:
                gpu_peak_mb = max(0.0, gpu_peak_mb - marker_adjust_mb)
                gpu_avg_mb = max(0.0, gpu_avg_mb - marker_adjust_mb)
        if self.num_gpu_delta_samples > 0:
            gpu_peak_delta_mb = self.peak_gpu_delta_bytes / (1024.0 * 1024.0)
            gpu_avg_delta_mb = (self.sum_gpu_delta_bytes / self.num_gpu_delta_samples) / (1024.0 * 1024.0)
        if self.num_gpu_proc_samples > 0:
            gpu_proc_peak_mb = self.peak_gpu_proc_bytes / (1024.0 * 1024.0)
            gpu_proc_avg_mb = (self.sum_gpu_proc_bytes / self.num_gpu_proc_samples) / (1024.0 * 1024.0)
        if self.num_gpu_proc_delta_samples > 0:
            gpu_proc_peak_delta_mb = self.peak_gpu_proc_delta_bytes / (1024.0 * 1024.0)
            gpu_proc_avg_delta_mb = (self.sum_gpu_proc_delta_bytes / self.num_gpu_proc_delta_samples) / (1024.0 * 1024.0)
        return {
            "rss_mb": (self.peak_rss_bytes / (1024.0 * 1024.0)) if self.peak_rss_bytes > 0 else None,
            "gpu_peak_mb": gpu_peak_mb,
            "gpu_avg_mb": gpu_avg_mb,
            "gpu_peak_delta_mb": gpu_peak_delta_mb,
            "gpu_avg_delta_mb": gpu_avg_delta_mb,
            "gpu_proc_peak_mb": gpu_proc_peak_mb,
            "gpu_proc_avg_mb": gpu_proc_avg_mb,
            "gpu_proc_peak_delta_mb": gpu_proc_peak_delta_mb,
            "gpu_proc_avg_delta_mb": gpu_proc_avg_delta_mb,
            "gpu_index": self._gpu_index,
        }


def timed_algorithm(callable_obj, sync_after=True):
    monitor = PeakMemoryMonitor().start()
    t0 = time.perf_counter()
    out = callable_obj()
    if sync_after:
        sync_gpu()
    elapsed = time.perf_counter() - t0
    mem = monitor.stop()
    return out, elapsed, mem


def emit(backend, function, metric, seconds, status="ok", correctness="", notes=""):
    row = {
        "backend": backend,
        "function": function,
        "metric": metric,
        "seconds": None if seconds is None else float(seconds),
        "status": status,
        "correctness": correctness,
        "notes": notes,
    }
    print("RESULT_JSON " + json.dumps(row, sort_keys=True), flush=True)


def emit_metrics(backend, function, build_s, algo_s, kernel_s, status="ok", correctness="", notes="", memory=None):
    emit(backend, function, "build", build_s, status=status, correctness=correctness, notes=notes)
    emit(backend, function, "e2e", algo_s, status=status, correctness=correctness, notes=notes)
    emit(backend, function, "kernel", kernel_s, status=status, correctness=correctness, notes=notes)
    if isinstance(memory, dict):
        rss_mb = memory.get("rss_mb")
        if rss_mb is not None:
            emit(
                backend,
                function,
                "memory_peak_rss_mb",
                rss_mb,
                status=status,
                correctness=correctness,
                notes=(notes + "; peak RSS during algorithm window" if notes else "peak RSS during algorithm window"),
            )
        gpu_peak_mb = memory.get("gpu_peak_mb")
        if gpu_peak_mb is not None:
            emit(
                backend,
                function,
                "memory_peak_gpu_mb",
                gpu_peak_mb,
                status=status,
                correctness=correctness,
                notes=(notes + "; peak GPU memory during algorithm window" if notes else "peak GPU memory during algorithm window"),
            )
        gpu_avg_mb = memory.get("gpu_avg_mb")
        if gpu_avg_mb is not None:
            emit(
                backend,
                function,
                "memory_avg_gpu_mb",
                gpu_avg_mb,
                status=status,
                correctness=correctness,
                notes=(notes + "; average GPU memory during algorithm window" if notes else "average GPU memory during algorithm window"),
            )
        gpu_peak_delta_mb = memory.get("gpu_peak_delta_mb")
        if gpu_peak_delta_mb is not None:
            emit(
                backend,
                function,
                "memory_peak_gpu_delta_mb",
                gpu_peak_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; peak GPU memory delta from run-start baseline"
                    if notes
                    else "peak GPU memory delta from run-start baseline"
                ),
            )
        gpu_avg_delta_mb = memory.get("gpu_avg_delta_mb")
        if gpu_avg_delta_mb is not None:
            emit(
                backend,
                function,
                "memory_avg_gpu_delta_mb",
                gpu_avg_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; average GPU memory delta from run-start baseline"
                    if notes
                    else "average GPU memory delta from run-start baseline"
                ),
            )
        gpu_proc_peak_mb = memory.get("gpu_proc_peak_mb")
        if gpu_proc_peak_mb is not None:
            emit(
                backend,
                function,
                "memory_peak_gpu_proc_mb",
                gpu_proc_peak_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; peak GPU memory of benchmark process tree during algorithm window"
                    if notes
                    else "peak GPU memory of benchmark process tree during algorithm window"
                ),
            )
        gpu_proc_avg_mb = memory.get("gpu_proc_avg_mb")
        if gpu_proc_avg_mb is not None:
            emit(
                backend,
                function,
                "memory_avg_gpu_proc_mb",
                gpu_proc_avg_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; average GPU memory of benchmark process tree during algorithm window"
                    if notes
                    else "average GPU memory of benchmark process tree during algorithm window"
                ),
            )
        gpu_proc_peak_delta_mb = memory.get("gpu_proc_peak_delta_mb")
        if gpu_proc_peak_delta_mb is not None:
            emit(
                backend,
                function,
                "memory_peak_gpu_proc_delta_mb",
                gpu_proc_peak_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; peak GPU memory delta of benchmark process tree from run-start baseline"
                    if notes
                    else "peak GPU memory delta of benchmark process tree from run-start baseline"
                ),
            )
        gpu_proc_avg_delta_mb = memory.get("gpu_proc_avg_delta_mb")
        if gpu_proc_avg_delta_mb is not None:
            emit(
                backend,
                function,
                "memory_avg_gpu_proc_delta_mb",
                gpu_proc_avg_delta_mb,
                status=status,
                correctness=correctness,
                notes=(
                    notes + "; average GPU memory delta of benchmark process tree from run-start baseline"
                    if notes
                    else "average GPU memory delta of benchmark process tree from run-start baseline"
                ),
            )


def emit_skip(backend, function, notes):
    emit_metrics(backend, function, None, None, None, status="skipped", notes=notes)


def emit_exception(backend, function, exc):
    notes = str(exc)
    status = "skipped" if "not implemented by 'cugraph' backend" in notes else "failed"
    emit_metrics(backend, function, None, None, None, status=status, notes=notes)


def is_nx_cugraph_not_implemented(exc):
    return "not implemented by 'cugraph' backend" in str(exc)


def concise_error(exc, limit=220):
    text = str(exc).strip()
    head = text.splitlines()[0] if text else ""
    return head if len(head) <= limit else (head[: limit - 3] + "...")


def warmup_call(callable_obj, warmup):
    for _ in range(max(0, int(warmup))):
        try:
            callable_obj()
        except Exception:
            break


def pick_sources(n, k):
    if n <= 0:
        return []
    k = max(1, int(k))
    if k >= n:
        return list(range(n))
    step = max(1, n // k)
    out = list(range(0, n, step))[:k]
    if len(out) < k:
        tail = n - 1
        while len(out) < k and tail >= 0:
            if tail not in out:
                out.append(tail)
            tail -= 1
    return out


def pick_optional_sources(n, k):
    k = int(k or 0)
    if k <= 0:
        return None
    return pick_sources(n, k)


def call_with_supported_kwargs(func, kwargs):
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return func(**kwargs), kwargs
    filtered = {k: v for k, v in kwargs.items() if k in params}
    if "G" in filtered:
        g = filtered["G"]
        rest = {k: v for k, v in filtered.items() if k != "G"}
        return func(g, **rest), filtered
    return func(**filtered), filtered


def build_igraph(n, edges, directed, weighted=False):
    import igraph as ig

    pairs = list(edges[["src", "dst"]].itertuples(index=False, name=None))
    g = ig.Graph(n=n, edges=[(int(u), int(v)) for u, v in pairs], directed=directed)
    if weighted:
        g.es["weight"] = [int(w) for w in edges["weight"].tolist()]
    return g


def build_networkx(n, edges, directed, weighted=False):
    import networkx as nx

    g = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(range(n))
    if weighted:
        for u, v, w in edges[["src", "dst", "weight"]].itertuples(index=False, name=None):
            g.add_edge(int(u), int(v), weight=int(w))
    else:
        g.add_edges_from((int(u), int(v)) for u, v in edges[["src", "dst"]].itertuples(index=False, name=None))
    return g


def build_easygraph(n, edges, directed, weighted=False):
    import easygraph as eg

    g = eg.DiGraph() if directed else eg.Graph()
    g.add_nodes_from(range(n))
    if weighted:
        for u, v, w in edges[["src", "dst", "weight"]].itertuples(index=False, name=None):
            g.add_edge(int(u), int(v), weight=int(w))
    else:
        g.add_edges_from((int(u), int(v)) for u, v in edges[["src", "dst"]].itertuples(index=False, name=None))
    return g


def component_semantics(func, graph_type, views):
    """Return a normalized component benchmark plan for WCC/SCC."""
    if func not in {"WCC", "SCC"}:
        raise ValueError(f"not a component function: {func}")
    if func == "SCC" and graph_type == "directed":
        n, directed_edges, _ = views["all_vertices"]
        return {
            "n": n,
            "edges": directed_edges,
            "build_directed": True,
            "igraph_mode": "strong",
            "nx_kind": "scc",
            "eg_kind": "scc",
            "cugraph_directed": True,
            "note": "SCC semantics",
            "kernel_key": "scc",
        }

    n, _, undirected_edges = views["all_vertices"]
    note = "WCC semantics; undirected projection" if graph_type == "directed" else "WCC semantics"
    if func == "SCC" and graph_type != "directed":
        note = "undirected graph: SCC equals WCC"
    return {
        "n": n,
        "edges": undirected_edges,
        "build_directed": False,
        "igraph_mode": "weak",
        "nx_kind": "wcc",
        "eg_kind": "wcc",
        "cugraph_directed": False,
        "note": note,
        "kernel_key": "cc",
    }


def easygraph_edges(graph):
    edges = graph.edges
    return edges() if callable(edges) else edges


def deterministic_weighted_edges(n, edges):
    if len(edges) == 0:
        out = edges[["src", "dst"]].copy()
        out["weight"] = pd.Series(dtype="int32")
        return out
    out = edges[["src", "dst"]].copy()
    mod = max(1, int(n))
    out["weight"] = (
        1
        + (
            out["src"].astype("int64") * out["dst"].astype("int64")
        ) % mod
    ).astype("int32")
    return out


def path_benchmark_plan(func, graph_type, views, source_count):
    n, directed_edges, undirected_edges = views["clean"]
    base_edges = directed_edges if graph_type == "directed" else undirected_edges
    weighted = func in {"Dijkstra", "BellmanFord", "SSSP"}
    edges = deterministic_weighted_edges(n, base_edges) if weighted else base_edges
    sources = pick_sources(n, source_count)
    if func == "BFS":
        note = f"unweighted shortest paths; sources={len(sources)}"
        detail_name = "BFS"
        kernel_key = "bfs"
    elif func == "BellmanFord":
        note = f"Bellman-Ford shortest paths on deterministic nonnegative weights; sources={len(sources)}"
        detail_name = "BellmanFord"
        kernel_key = "bellman_ford"
    elif func == "Dijkstra":
        note = f"Dijkstra shortest paths on deterministic nonnegative weights; sources={len(sources)}"
        detail_name = "Dijkstra"
        kernel_key = "dijkstra"
    else:
        note = f"weighted deterministic edges; sources={len(sources)}"
        detail_name = "SSSP"
        kernel_key = "sssp"
    return {
        "n": n,
        "edges": edges,
        "weighted": weighted,
        "directed": graph_type == "directed",
        "sources": sources,
        "note": note,
        "detail_name": detail_name,
        "kernel_key": kernel_key,
    }


def structural_hole_plan(func, graph_type, views):
    n, directed_edges, undirected_edges = views["clean"]
    base_edges = directed_edges if graph_type == "directed" else undirected_edges
    key = {
        "EffectiveSize": "effective_size",
        "Efficiency": "efficiency",
        "Constraint": "constraint",
        "Hierarchy": "hierarchy",
    }[func]
    return {
        "n": n,
        "edges": base_edges,
        "directed": graph_type == "directed",
        "kernel_key": key,
        "note": "Burt structural-hole metric; unweighted graph",
    }


def summarize_numeric_mapping(values):
    if hasattr(values, "tolist") and not isinstance(values, dict):
        values = values.tolist()
    if isinstance(values, dict):
        seq = list(values.values())
    elif isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        seq = []
    clean = []
    for v in seq:
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            clean.append(x)
    total = float(sum(clean)) if clean else 0.0
    mean = total / len(clean) if clean else 0.0
    return len(seq), total, mean


def summarize_sssp_result(result):
    reachable = 0
    checksum = 0.0
    if isinstance(result, dict) and "values" in result:
        result = result.get("values")
    if isinstance(result, dict):
        if result and all(isinstance(v, dict) for v in result.values()):
            for dist_map in result.values():
                for dv in dist_map.values():
                    try:
                        x = float(dv)
                    except Exception:
                        continue
                    if math.isfinite(x) and abs(x) < 1.0e30:
                        reachable += 1
                        checksum += x
            return reachable, checksum
        for dv in result.values():
            try:
                x = float(dv)
            except Exception:
                continue
            if math.isfinite(x) and abs(x) < 1.0e30:
                reachable += 1
                checksum += x
        return reachable, checksum
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, (list, tuple)):
        for row in result:
            if hasattr(row, "tolist"):
                row = row.tolist()
            if isinstance(row, (list, tuple)):
                for dv in row:
                    try:
                        x = float(dv)
                    except Exception:
                        continue
                    if math.isfinite(x) and abs(x) < 1.0e30:
                        reachable += 1
                        checksum += x
            else:
                try:
                    x = float(row)
                except Exception:
                    continue
                if math.isfinite(x) and abs(x) < 1.0e30:
                    reachable += 1
                    checksum += x
    return reachable, checksum


def _detail_path(backend, function, suffix="npz"):
    if not STRICT_VALIDATION or not DETAIL_DIR:
        return None
    safe_backend = str(backend).replace("/", "_")
    safe_function = str(function).replace("/", "_")
    out_dir = Path(DETAIL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe_backend}_{safe_function}.{suffix}"


def _array_digest(arr):
    import numpy as np

    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(tuple(arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()[:16]


def _numeric_vector_from_mapping(values, n, dtype="float64", default=0.0):
    import numpy as np

    arr = np.full(int(n), default, dtype=dtype)
    if hasattr(values, "tolist") and not isinstance(values, dict):
        values = values.tolist()
    if isinstance(values, dict):
        for k, v in values.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if 0 <= idx < n:
                arr[idx] = v
    elif isinstance(values, (list, tuple)):
        upto = min(int(n), len(values))
        if upto:
            arr[:upto] = values[:upto]
    return arr


def write_vector_detail(backend, function, values, n, dtype="float64", default=0.0):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    arr = _numeric_vector_from_mapping(values, n, dtype=dtype, default=default)
    np.savez_compressed(path, kind="vector", values=arr)
    return f", detail={path}, detail_kind=vector, detail_sha={_array_digest(arr)}"


def write_source_vector_detail(backend, function, sources, values, n, dtype="float64"):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    src = np.asarray([int(s) for s in sources], dtype=np.int64)
    vals = np.asarray([float(v) for v in values], dtype=dtype)
    np.savez_compressed(path, kind="source_vector", sources=src, values=vals, graph_n=int(n))
    digest_arr = np.column_stack((src, vals.astype(np.float64, copy=False))) if len(src) else vals
    return f", detail={path}, detail_kind=source_vector, detail_sha={_array_digest(digest_arr)}"


def write_cc_detail(backend, function, components, n):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    labels = np.full(int(n), -1, dtype=np.int64)
    normalized_components = []
    for comp in components:
        vals = []
        try:
            iterator = list(comp)
        except Exception:
            iterator = []
        for node in iterator:
            try:
                idx = int(node)
            except Exception:
                continue
            if 0 <= idx < n:
                vals.append(idx)
        if vals:
            normalized_components.append(sorted(set(vals)))
    normalized_components.sort(key=lambda xs: (xs[0], len(xs)))
    for label, comp in enumerate(normalized_components):
        for idx in comp:
            labels[idx] = label
    next_label = len(normalized_components)
    for idx in range(int(n)):
        if labels[idx] < 0:
            labels[idx] = next_label
            next_label += 1
    np.savez_compressed(path, kind="cc_labels", values=labels)
    return f", detail={path}, detail_kind=cc_labels, detail_sha={_array_digest(labels)}"


def write_sssp_detail(backend, function, result, sources, n):
    import numpy as np

    path = _detail_path(backend, function)
    if path is None:
        return ""
    source_list = [int(s) for s in sources]
    arr = np.full((len(source_list), int(n)), np.inf, dtype=np.float64)
    source_pos = {s: i for i, s in enumerate(source_list)}

    def put(row, node, value):
        try:
            idx = int(node)
            val = float(value)
        except Exception:
            return
        if 0 <= idx < n and math.isfinite(val) and abs(val) < 1.0e30:
            arr[row, idx] = val

    if isinstance(result, dict) and "values" in result:
        result = result.get("values")

    if isinstance(result, dict):
        if result and all(isinstance(v, dict) for v in result.values()):
            for source, dist_map in result.items():
                try:
                    row = source_pos[int(source)]
                except Exception:
                    continue
                for node, value in dist_map.items():
                    put(row, node, value)
        else:
            row = 0
            for node, value in result.items():
                put(row, node, value)
    else:
        rows = result.tolist() if hasattr(result, "tolist") else list(result)
        for row, dist_row in enumerate(rows[: len(source_list)]):
            vals = dist_row.tolist() if hasattr(dist_row, "tolist") else list(dist_row)
            for idx, value in enumerate(vals[: int(n)]):
                put(row, idx, value)
    finite = np.isfinite(arr)
    digest_arr = np.where(finite, arr, -1.0)
    np.savez_compressed(path, kind="sssp", sources=np.asarray(source_list, dtype=np.int64), values=arr)
    return f", detail={path}, detail_kind=sssp, detail_sha={_array_digest(digest_arr)}"


def nx_cugraph_call(func, *args, **kwargs):
    kwargs["backend"] = "cugraph"
    out = func(*args, **kwargs)
    sync_gpu()
    return out


def cugraph_graph_from_edges(edges_df, directed=False, weighted=False):
    import cudf
    import cugraph

    if weighted:
        cdf = cudf.from_pandas(edges_df[["src", "dst", "weight"]].copy())
    else:
        cdf = cudf.from_pandas(edges_df[["src", "dst"]].copy())
    try:
        G = cugraph.DiGraph() if directed and hasattr(cugraph, "DiGraph") else cugraph.Graph(directed=directed)
    except Exception:
        G = cugraph.Graph(directed=directed)
    if weighted:
        G.from_cudf_edgelist(cdf, source="src", destination="dst", edge_attr="weight", renumber=False)
    else:
        G.from_cudf_edgelist(cdf, source="src", destination="dst", renumber=False)
    return G


def cugraph_mst_weight(undirected_edges):
    import cugraph

    G = cugraph_graph_from_edges(undirected_edges, directed=False, weighted=True)
    tree = cugraph.minimum_spanning_tree(G, weight="weight")
    sync_gpu()
    edf = tree.view_edge_list() if hasattr(tree, "view_edge_list") else tree
    pdf = edf.to_pandas()
    wcol = "weight" if "weight" in pdf.columns else ("weights" if "weights" in pdf.columns else "w")
    return int(pdf[wcol].astype("int64").sum())


def cugraph_lcc_values(undirected_edges):
    import cugraph

    G = cugraph_graph_from_edges(undirected_edges, directed=False, weighted=False)
    tri = cugraph.triangle_count(G)
    sync_gpu()
    tri_pdf = tri.to_pandas()
    tri_col = "counts" if "counts" in tri_pdf.columns else ("count" if "count" in tri_pdf.columns else tri_pdf.columns[-1])
    tri_map = dict(zip(tri_pdf["vertex"].astype("int64"), tri_pdf[tri_col].astype("float64")))
    deg = pd.concat([undirected_edges["src"], undirected_edges["dst"]], ignore_index=True).value_counts()
    vals = {}
    for vertex, degree in deg.items():
        comb2 = degree * (degree - 1) / 2.0
        vals[int(vertex)] = float(tri_map.get(int(vertex), 0.0) / comb2) if comb2 > 0 else 0.0
    return vals


def cugraph_cc_components(edges_df, directed):
    import cugraph
    import cudf

    cdf = cudf.from_pandas(edges_df[["src", "dst"]].copy())
    try:
        G = cugraph.DiGraph() if directed and hasattr(cugraph, "DiGraph") else cugraph.Graph(directed=directed)
    except Exception:
        G = cugraph.Graph(directed=directed)
    G.from_cudf_edgelist(cdf, source="src", destination="dst", renumber=bool(directed))
    out = cugraph.strongly_connected_components(G) if directed else cugraph.connected_components(G)
    sync_gpu()
    pdf = out.to_pandas()
    label_col = "labels" if "labels" in pdf.columns else ("label" if "label" in pdf.columns else "component")
    return int(pdf[label_col].nunique())


def cugraph_sssp_multi_source(weighted_edges, sources, directed):
    import cugraph

    G = cugraph_graph_from_edges(weighted_edges, directed=directed, weighted=True)
    out = {}
    for s in sources:
        df = cugraph.sssp(G, source=int(s))
        sync_gpu()
        pdf = df.to_pandas()
        dist_col = "distance" if "distance" in pdf.columns else pdf.columns[-1]
        dist = {}
        for vertex, value in zip(pdf["vertex"], pdf[dist_col]):
            x = float(value)
            if math.isfinite(x) and abs(x) < 1.0e30:
                dist[int(vertex)] = x
        out[int(s)] = dist
    return out


def cugraph_kcore_values(edges_df, directed):
    import cugraph

    G = cugraph_graph_from_edges(edges_df, directed=directed, weighted=False)
    out = cugraph.core_number(G)
    sync_gpu()
    pdf = out.to_pandas()
    col = "core_number" if "core_number" in pdf.columns else ("core" if "core" in pdf.columns else pdf.columns[-1])
    return {int(v): int(c) for v, c in zip(pdf["vertex"], pdf[col])}


def cugraph_bc_values(edges_df, directed, sources, normalized=False, endpoints=False):
    import cugraph

    G = cugraph_graph_from_edges(edges_df, directed=directed, weighted=False)
    out = cugraph.betweenness_centrality(
        G,
        k=[int(s) for s in sources],
        normalized=bool(normalized),
        endpoints=bool(endpoints),
    )
    sync_gpu()
    pdf = out.to_pandas()
    col = (
        "betweenness_centrality"
        if "betweenness_centrality" in pdf.columns
        else ("betweenness" if "betweenness" in pdf.columns else pdf.columns[-1])
    )
    return {int(v): float(c) for v, c in zip(pdf["vertex"], pdf[col])}


def bench_igraph(views, graph_type, skip_cpu, functions, pr_alpha, cooldown, warmup, sssp_sources, bc_sources, closeness_sources):
    backend = "igraph"
    if skip_cpu:
        for func in functions:
            emit_skip(backend, func, "CPU baseline skipped by size threshold")
        return

    try:
        import igraph as ig  # noqa: F401
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"igraph import failed: {e}")
        return

    for func in functions:
        try:
            if func == "PageRank":
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_igraph(n, directed_edges if graph_type == "directed" else undirected_edges, graph_type == "directed")
                t1 = time.perf_counter()
                warmup_call(lambda: g.pagerank(directed=(graph_type == "directed"), damping=pr_alpha), warmup)
                barrier(cooldown)
                ranks, algo_s, mem = timed_algorithm(lambda: g.pagerank(directed=(graph_type == "directed"), damping=pr_alpha), sync_after=False)
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sum={sum(ranks):.9g}" + write_vector_detail(backend, "PageRank", ranks, n),
                    notes=f"alpha={pr_alpha}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "MST":
                n, _, undirected_edges = views["all_vertices"]
                t0 = time.perf_counter()
                g = build_igraph(n, undirected_edges, False, weighted=True)
                t1 = time.perf_counter()
                warmup_call(lambda: g.spanning_tree(weights=g.es["weight"], return_tree=True), warmup)
                barrier(cooldown)
                tree, algo_s, mem = timed_algorithm(lambda: g.spanning_tree(weights=g.es["weight"], return_tree=True), sync_after=False)
                weight = int(sum(tree.es["weight"])) if tree.ecount() else 0
                emit_metrics(
                    backend,
                    "MST",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"weight={weight}",
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "LCC":
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_igraph(n, undirected_edges, False)
                t1 = time.perf_counter()
                warmup_call(lambda: g.transitivity_local_undirected(mode="zero"), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: g.transitivity_local_undirected(mode="zero"), sync_after=False)
                finite = [v for v in vals if v is not None and not math.isnan(float(v))]
                corr = f"vertices={len(vals)}, mean={sum(finite)/len(finite):.9g}" if finite else f"vertices={len(vals)}"
                corr += write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=corr,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "Closeness":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                t0 = time.perf_counter()
                g = build_igraph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                mode = "OUT" if graph_type == "directed" else "ALL"
                source_nodes = pick_optional_sources(n, closeness_sources)

                def run_closeness():
                    vertices = None if source_nodes is None else source_nodes
                    vals = g.closeness(vertices=vertices, mode=mode, weights=None, normalized=True)
                    reachable = g.neighborhood_size(vertices=vertices, order=max(0, n), mode=mode)
                    out = []
                    denom = max(1, n - 1)
                    for idx, v in enumerate(vals):
                        try:
                            x = float(v)
                        except Exception:
                            x = 0.0
                        if not math.isfinite(x):
                            x = 0.0
                        try:
                            wf = max(0, int(reachable[idx]) - 1) / float(denom)
                        except Exception:
                            wf = 0.0
                        x *= wf
                        out.append(x)
                    return out

                warmup_call(run_closeness, warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_closeness, sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                if source_nodes is None:
                    detail = write_vector_detail(backend, "Closeness", vals, n)
                    corr_prefix = f"nodes={count}"
                    note_prefix = "outgoing distance for directed graphs"
                else:
                    detail = write_source_vector_detail(backend, "Closeness", source_nodes, vals, n)
                    corr_prefix = f"sources={len(source_nodes)}, graph_nodes={n}"
                    note_prefix = (
                        "sampled-target exact closeness; deterministic evenly spaced sources; "
                        f"sources={len(source_nodes)}; outgoing distance for directed graphs"
                    )
                emit_metrics(
                    backend,
                    "Closeness",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"{corr_prefix}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note_prefix
                    + "; igraph closeness mode="
                    + mode
                    + "; Wasserman-Faust disconnected correction applied; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in {"WCC", "SCC"}:
                plan = component_semantics(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_igraph(n, plan["edges"], plan["build_directed"])
                t1 = time.perf_counter()
                warmup_call(lambda: g.connected_components(mode=plan["igraph_mode"]), warmup)
                barrier(cooldown)
                comps, algo_s, mem = timed_algorithm(
                    lambda: g.connected_components(mode=plan["igraph_mode"]),
                    sync_after=False,
                )
                sizes = comps.sizes()
                cc_detail = write_cc_detail(backend, func, comps, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"components={len(sizes)}" + cc_detail,
                    notes=plan["note"] + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_igraph(n, plan["edges"], plan["directed"], weighted=plan["weighted"])
                t1 = time.perf_counter()
                mode = "OUT" if graph_type == "directed" else "ALL"
                weights = "weight" if plan["weighted"] else None
                algorithm = "bellman_ford" if func == "BellmanFord" else ("dijkstra" if plan["weighted"] else "auto")
                warmup_call(lambda: g.distances(source=source_nodes, weights=weights, mode=mode, algorithm=algorithm), warmup)
                barrier(cooldown)
                dists, algo_s, mem = timed_algorithm(
                    lambda: g.distances(source=source_nodes, weights=weights, mode=mode, algorithm=algorithm),
                    sync_after=False,
                )
                reachable, checksum = summarize_sssp_result(dists)
                sssp_detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + sssp_detail,
                    notes=plan["note"] + f"; igraph algorithm={algorithm}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                if func != "Constraint":
                    emit_skip(backend, func, "igraph has no native Burt effective-size/efficiency/hierarchy API aligned with EasyGraph")
                    continue
                plan = structural_hole_plan(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_igraph(n, plan["edges"], plan["directed"])
                t1 = time.perf_counter()
                warmup_call(lambda: g.constraint(weights=None), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: g.constraint(weights=None), sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                detail = write_vector_detail(backend, func, vals, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={count}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=plan["note"] + "; igraph Graph.constraint; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "KCore":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = build_igraph(n, base_edges, False)
                t1 = time.perf_counter()
                mode = "all"
                warmup_call(lambda: g.coreness(mode=mode), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: g.coreness(mode=mode), sync_after=False)
                s = float(sum(vals)) if vals else 0.0
                mx = int(max(vals)) if vals else 0
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64")
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={s:.9g}, max={mx}" + detail,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "BC":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                source_nodes = pick_sources(n, bc_sources)
                if not source_nodes:
                    emit_skip(backend, "BC", "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_igraph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                all_nodes = list(range(n))
                directed_flag = graph_type == "directed"
                warmup_call(
                    lambda: g.betweenness(
                        vertices=None,
                        directed=directed_flag,
                        weights=None,
                        sources=source_nodes,
                        targets=all_nodes,
                    ),
                    warmup,
                )
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    lambda: g.betweenness(
                        vertices=None,
                        directed=directed_flag,
                        weights=None,
                        sources=source_nodes,
                        targets=all_nodes,
                    ),
                    sync_after=False,
                )
                total = float(sum(vals)) if vals else 0.0
                detail = write_vector_detail(backend, "BC", vals, n)
                emit_metrics(
                    backend,
                    "BC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={total:.9g}" + detail,
                    notes=f"source-sampled exact mode; sources={len(source_nodes)}; normalized=False; cpu backend kernel=algorithm",
                    memory=mem,
                )
        except Exception as e:
            emit_exception(backend, func, e)


def bench_networkx(views, graph_type, skip_cpu, functions, pr_alpha, pr_max_iter, pr_tol, cooldown, warmup, sssp_sources, bc_sources, closeness_sources):
    backend = "networkx"
    if skip_cpu:
        for func in functions:
            emit_skip(backend, func, "CPU baseline skipped by explicit --skip-cpu")
        return

    try:
        import networkx as nx
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"networkx import failed: {e}")
        return

    for func in functions:
        try:
            if func == "PageRank":
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_networkx(n, directed_edges if graph_type == "directed" else undirected_edges, graph_type == "directed")
                t1 = time.perf_counter()
                warmup_call(lambda: nx.pagerank(g, alpha=pr_alpha, max_iter=pr_max_iter, tol=pr_tol), warmup)
                barrier(cooldown)
                ranks, algo_s, mem = timed_algorithm(
                    lambda: nx.pagerank(g, alpha=pr_alpha, max_iter=pr_max_iter, tol=pr_tol),
                    sync_after=False,
                )
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sum={sum(ranks.values()):.9g}" + write_vector_detail(backend, "PageRank", ranks, n),
                    notes=f"alpha={pr_alpha}, max_iter={pr_max_iter}, tol={pr_tol}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "MST":
                n, _, undirected_edges = views["all_vertices"]
                t0 = time.perf_counter()
                g = build_networkx(n, undirected_edges, False, weighted=True)
                t1 = time.perf_counter()
                warmup_call(lambda: nx.minimum_spanning_tree(g, weight="weight"), warmup)
                barrier(cooldown)
                tree, algo_s, mem = timed_algorithm(lambda: nx.minimum_spanning_tree(g, weight="weight"), sync_after=False)
                weight = int(sum(data.get("weight", 1) for _, _, data in tree.edges(data=True)))
                emit_metrics(
                    backend,
                    "MST",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"weight={weight}",
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "LCC":
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_networkx(n, undirected_edges, False)
                t1 = time.perf_counter()
                warmup_call(lambda: nx.clustering(g), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: nx.clustering(g), sync_after=False)
                mean = sum(vals.values()) / len(vals) if vals else 0.0
                detail = write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"vertices={len(vals)}, mean={mean:.9g}" + detail,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "Closeness":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, graph_type == "directed")
                close_graph = g.reverse(copy=False) if graph_type == "directed" else g
                t1 = time.perf_counter()
                source_nodes = pick_optional_sources(n, closeness_sources)
                if source_nodes is None:
                    run_closeness = lambda: nx.closeness_centrality(
                        close_graph,
                        distance=None,
                        wf_improved=True,
                    )
                else:
                    run_closeness = lambda: [
                        nx.closeness_centrality(
                            close_graph,
                            u=int(source),
                            distance=None,
                            wf_improved=True,
                        )
                        for source in source_nodes
                    ]
                warmup_call(run_closeness, warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_closeness, sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                if source_nodes is None:
                    detail = write_vector_detail(backend, "Closeness", vals, n)
                    corr_prefix = f"nodes={count}"
                    note = "outgoing distance for directed graphs"
                else:
                    detail = write_source_vector_detail(backend, "Closeness", source_nodes, vals, n)
                    corr_prefix = f"sources={len(source_nodes)}, graph_nodes={n}"
                    note = (
                        "sampled-target exact closeness; deterministic evenly spaced sources; "
                        f"sources={len(source_nodes)}; outgoing distance for directed graphs"
                    )
                if graph_type == "directed":
                    note += "; networkx uses reverse graph view because its directed closeness is inward by default"
                emit_metrics(
                    backend,
                    "Closeness",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"{corr_prefix}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in {"WCC", "SCC"}:
                plan = component_semantics(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["build_directed"])
                t1 = time.perf_counter()
                if plan["nx_kind"] == "scc":
                    run_components = lambda: list(nx.strongly_connected_components(g))
                else:
                    run_components = lambda: list(nx.connected_components(g))
                warmup_call(run_components, warmup)
                barrier(cooldown)
                comps, algo_s, mem = timed_algorithm(run_components, sync_after=False)
                detail = write_cc_detail(backend, func, comps, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"components={len(comps)}" + detail,
                    notes=plan["note"] + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["directed"], weighted=plan["weighted"])
                t1 = time.perf_counter()
                if func == "BFS":
                    nx_path_fn = lambda s: nx.single_source_shortest_path_length(g, s)
                    algo_name = "single_source_shortest_path_length"
                elif func == "BellmanFord":
                    nx_path_fn = lambda s: nx.single_source_bellman_ford_path_length(g, s, weight="weight")
                    algo_name = "single_source_bellman_ford_path_length"
                else:
                    nx_path_fn = lambda s: nx.single_source_dijkstra_path_length(g, s, weight="weight")
                    algo_name = "single_source_dijkstra_path_length"
                warmup_call(
                    lambda: {
                        s: nx_path_fn(s)
                        for s in source_nodes
                    },
                    warmup,
                )
                barrier(cooldown)
                dists, algo_s, mem = timed_algorithm(
                    lambda: {
                        s: nx_path_fn(s)
                        for s in source_nodes
                    },
                    sync_after=False,
                )
                reachable, checksum = summarize_sssp_result(dists)
                detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + detail,
                    notes=plan["note"] + f"; networkx algorithm={algo_name}; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                if func == "Hierarchy":
                    emit_skip(backend, func, "networkx has no native Burt hierarchy metric API")
                    continue
                plan = structural_hole_plan(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["directed"])
                t1 = time.perf_counter()

                if func == "EffectiveSize":
                    run_structural = lambda: nx.effective_size(g, weight=None)
                    note = plan["note"] + "; networkx structuralholes.effective_size"
                elif func == "Constraint":
                    run_structural = lambda: nx.constraint(g, weight=None)
                    note = plan["note"] + "; networkx structuralholes.constraint"
                else:
                    def run_efficiency():
                        e_size = nx.effective_size(g, weight=None)
                        degree = dict(g.degree())
                        return {
                            node: (float("nan") if degree.get(node, 0) == 0 else float(val) / float(degree[node]))
                            for node, val in e_size.items()
                        }

                    run_structural = run_efficiency
                    note = plan["note"] + "; derived as networkx effective_size / degree"

                warmup_call(run_structural, warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_structural, sync_after=False)
                count, total, mean = summarize_numeric_mapping(vals)
                detail = write_vector_detail(backend, func, vals, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={count}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note + "; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "KCore":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, False)
                t1 = time.perf_counter()
                warmup_call(lambda: nx.core_number(g), warmup)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: nx.core_number(g), sync_after=False)
                total = float(sum(vals.values())) if vals else 0.0
                max_core = int(max(vals.values())) if vals else 0
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64")
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={total:.9g}, max={max_core}" + detail,
                    notes="undirected projection; cpu backend kernel=algorithm",
                    memory=mem,
                )

            elif func == "BC":
                from networkx.algorithms.centrality import betweenness_centrality_subset

                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                source_nodes = pick_sources(n, bc_sources)
                if not source_nodes:
                    emit_skip(backend, "BC", "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                targets = list(g.nodes())
                warmup_call(
                    lambda: betweenness_centrality_subset(
                        g,
                        sources=source_nodes,
                        targets=targets,
                        normalized=False,
                        weight=None,
                    ),
                    warmup,
                )
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    lambda: betweenness_centrality_subset(
                        g,
                        sources=source_nodes,
                        targets=targets,
                        normalized=False,
                        weight=None,
                    ),
                    sync_after=False,
                )
                total = float(sum(vals.values())) if vals else 0.0
                detail = write_vector_detail(backend, "BC", vals, n)
                emit_metrics(
                    backend,
                    "BC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={len(vals)}, sum={total:.9g}" + detail,
                    notes=f"source-sampled subset mode; sources={len(source_nodes)}; normalized=False; cpu backend kernel=algorithm",
                    memory=mem,
                )
        except Exception as e:
            emit_exception(backend, func, e)


def bench_easygraph_mode(
    views,
    graph_type,
    skip_cpu,
    functions,
    pr_alpha,
    pr_max_iter,
    pr_tol,
    warmup,
    easygraph_warmup,
    easygraph_gpu_backend,
    sssp_sources,
    bc_sources,
    closeness_sources,
    cooldown,
    mode,
):
    if mode == "gpu":
        backend = "EGGPU"
        os.environ["EASYGRAPH_ENABLE_GPU"] = "TRUE"
        os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
        os.environ["EASYGRAPH_GPU_BACKEND"] = "mine"
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "TRUE"
        os.environ["EASYGRAPH_GPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_GPU_PR_EPS"] = str(pr_tol)
        os.environ["EASYGRAPH_CPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_CPU_PR_TOL"] = str(pr_tol)
        # Keep timed runs unbiased by backend result cache hits.
        os.environ["EASYGRAPH_GPU_RESULT_CACHE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_RESULT_CACHE_RETURN_COPY"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
    elif mode == "cpu":
        backend = "easygraph-cpu"
        os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        os.environ.pop("EASYGRAPH_GPU_BACKEND", None)
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
        os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_CPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_CPU_PR_TOL"] = str(pr_tol)
    elif mode == "cpp":
        backend = "easygraph-cpp"
        os.environ["EASYGRAPH_ENABLE_GPU"] = "FALSE"
        os.environ.pop("EASYGRAPH_GPU_BACKEND", None)
        os.environ["EASYGRAPH_GPU_STRICT_ERRORS"] = "FALSE"
        os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SCC_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_KCORE_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_GPU_SSSP_HOST_ENABLE"] = "FALSE"
        os.environ["EASYGRAPH_CPU_PR_MAX_ITER"] = str(pr_max_iter)
        os.environ["EASYGRAPH_CPU_PR_TOL"] = str(pr_tol)
    else:
        raise ValueError(f"unknown easygraph mode: {mode}")

    if skip_cpu and mode in ("cpu", "cpp"):
        for func in functions:
            emit_skip(backend, func, "CPU baseline skipped by explicit --skip-cpu")
        return

    print(
        f"[easygraph-mode] backend={backend} mode={mode} "
        f"EASYGRAPH_ENABLE_GPU={os.environ.get('EASYGRAPH_ENABLE_GPU', '')} "
        f"EASYGRAPH_GPU_BACKEND={os.environ.get('EASYGRAPH_GPU_BACKEND', '')}",
        flush=True,
    )

    try:
        import easygraph as eg
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"easygraph import failed: {e}")
        return

    eg_mine_backend = None
    if mode == "gpu":
        try:
            from easygraph.utils import gpu_mine_backend as eg_mine_backend_mod

            eg_mine_backend = eg_mine_backend_mod
        except Exception:
            eg_mine_backend = None

    def to_mode_graph(n, edges, directed, weighted=False):
        g = build_easygraph(n, edges, directed, weighted=weighted)
        if mode == "cpp":
            return g.cpp()
        if mode == "gpu" and eg_mine_backend is not None:
            try:
                eg_mine_backend._graph_context(g, prewarm_cpp=True)
            except Exception as exc:
                if os.environ.get("EASYGRAPH_GPU_STRICT_ERRORS", "").strip().upper() in {
                    "1",
                    "TRUE",
                    "ON",
                    "YES",
                }:
                    raise RuntimeError("EGGPU graph-context prewarm failed") from exc
                pass
        return g

    if mode == "gpu":
        effective_warmup = max(0, int(max(int(warmup), int(easygraph_warmup))))
    else:
        effective_warmup = max(0, int(warmup))
    # EGGPU kernels synchronize/copy required outputs before returning. An
    # extra Python-side CUDA sync is not part of the user-facing call path and
    # over-penalizes small-graph e2e latency.
    sync_after_eg_call = False

    def easygraph_warmup(callable_obj):
        for _ in range(effective_warmup):
            try:
                callable_obj()
            except Exception:
                if mode == "gpu":
                    raise
                break

    def kernel_or_algo(kernel_key, algo_seconds):
        if mode != "gpu" or eg_mine_backend is None:
            return algo_seconds
        try:
            k = eg_mine_backend.get_last_kernel_time(kernel_key)
        except Exception:
            k = None
        return algo_seconds if k is None else float(k)

    def reset_kernel_key(kernel_key):
        if mode != "gpu" or eg_mine_backend is None:
            return
        try:
            eg_mine_backend.set_last_kernel_time(kernel_key, None)
        except Exception:
            pass

    for func in functions:
        try:
            if func == "PageRank":
                reset_kernel_key("pagerank")
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, directed_edges if graph_type == "directed" else undirected_edges, graph_type == "directed")
                t1 = time.perf_counter()
                easygraph_warmup(
                    lambda: call_with_supported_kwargs(
                        eg.pagerank,
                        {"G": g, "alpha": pr_alpha, "max_iter": pr_max_iter, "tol": pr_tol},
                    )[0]
                )
                barrier(cooldown)
                def run_pagerank_attempts():
                    kwargs = {"G": g, "alpha": pr_alpha, "max_iter": pr_max_iter, "tol": pr_tol}
                    return call_with_supported_kwargs(eg.pagerank, kwargs)

                (ranks, used), algo_s, mem = timed_algorithm(run_pagerank_attempts, sync_after=sync_after_eg_call)
                if isinstance(ranks, dict):
                    rank_sum = float(sum(ranks.values()))
                elif isinstance(ranks, (list, tuple)):
                    rank_sum = float(sum(ranks))
                else:
                    raise TypeError(f"unsupported pagerank result type: {type(ranks).__name__}")
                alpha_used = used.get("alpha") if used else None
                if alpha_used is None or abs(float(alpha_used) - float(pr_alpha)) > 1.0e-12:
                    raise RuntimeError(
                        f"easygraph pagerank did not use requested alpha={pr_alpha}; used={alpha_used}"
                    )
                note = f"alpha={alpha_used}, max_iter={pr_max_iter}, tol={pr_tol}"
                if isinstance(ranks, (list, tuple)):
                    note += "; pagerank returned dense vector"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, "PageRank", ranks, n)
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("pagerank", algo_s),
                    correctness=f"sum={rank_sum:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "MST":
                reset_kernel_key("mst")
                n, _, undirected_edges = views["all_vertices"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, undirected_edges, False, weighted=True)
                t1 = time.perf_counter()
                easygraph_warmup(lambda: eg.minimum_spanning_tree(g, weight="weight"))
                barrier(cooldown)
                tree, algo_s, mem = timed_algorithm(
                    lambda: eg.minimum_spanning_tree(g, weight="weight"),
                    sync_after=sync_after_eg_call,
                )
                weight = int(sum(data.get("weight", 1) for _, _, data in easygraph_edges(tree)))
                note = "undirected projection"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                emit_metrics(
                    backend,
                    "MST",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("mst", algo_s),
                    correctness=f"weight={weight}",
                    notes=note,
                    memory=mem,
                )

            elif func == "LCC":
                reset_kernel_key("lcc")
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, undirected_edges, False)
                t1 = time.perf_counter()
                easygraph_warmup(lambda: eg.clustering(g))
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(lambda: eg.clustering(g), sync_after=sync_after_eg_call)
                mean = sum(vals.values()) / len(vals) if vals else 0.0
                note = "undirected projection"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("lcc", algo_s),
                    correctness=f"vertices={len(vals)}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "Closeness":
                reset_kernel_key("closeness")
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                t0 = time.perf_counter()
                g = to_mode_graph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                source_nodes = pick_optional_sources(n, closeness_sources)
                run_closeness = lambda: eg.closeness_centrality(
                    g,
                    weight=None,
                    sources=source_nodes,
                )
                easygraph_warmup(run_closeness)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_closeness, sync_after=sync_after_eg_call)
                count, total, mean = summarize_numeric_mapping(vals)
                if source_nodes is None:
                    detail = write_vector_detail(backend, "Closeness", vals, n)
                    corr_prefix = f"nodes={count}"
                    note = "outgoing distance for directed graphs"
                else:
                    detail = write_source_vector_detail(backend, "Closeness", source_nodes, vals, n)
                    corr_prefix = f"sources={len(source_nodes)}, graph_nodes={n}"
                    note = (
                        "sampled-target exact closeness; deterministic evenly spaced sources; "
                        f"sources={len(source_nodes)}; outgoing distance for directed graphs"
                    )
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                else:
                    note += "; unweighted GPU path uses BFS kernel"
                emit_metrics(
                    backend,
                    "Closeness",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("closeness", algo_s),
                    correctness=f"{corr_prefix}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func in {"WCC", "SCC"}:
                plan = component_semantics(func, graph_type, views)
                reset_kernel_key(plan["kernel_key"])
                n = plan["n"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, plan["edges"], plan["build_directed"])
                t1 = time.perf_counter()
                if plan["eg_kind"] == "scc":
                    run_components = lambda: list(eg.strongly_connected_components(g))
                else:
                    run_components = lambda: list(eg.connected_components(g))
                easygraph_warmup(run_components)
                barrier(cooldown)
                comps, algo_s, mem = timed_algorithm(run_components, sync_after=sync_after_eg_call)
                note = plan["note"]
                kernel_key = plan["kernel_key"]
                if mode != "gpu":
                    note = (note + "; " if note else "") + "cpu backend kernel=algorithm"
                detail = write_cc_detail(backend, func, comps, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_or_algo(kernel_key, algo_s),
                    correctness=f"components={len(comps)}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                reset_kernel_key(plan["kernel_key"])
                if func in {"Dijkstra", "BFS"}:
                    reset_kernel_key("sssp")
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                if mode == "cpp" and func == "BellmanFord":
                    emit_skip(backend, func, "easygraph-cpp has no Bellman-Ford binding; CPU/GPU EasyGraph paths are benchmarked")
                    continue
                t0 = time.perf_counter()
                g = to_mode_graph(n, plan["edges"], plan["directed"], weighted=plan["weighted"])
                t1 = time.perf_counter()

                if func == "BFS":
                    if mode == "cpp":
                        run_paths = lambda: eg.multi_source_dijkstra(g, source_nodes, weight=None, target=None)
                    else:
                        run_paths = lambda: eg.multi_source_bfs(g, source_nodes, target=None)
                    note = plan["note"]
                    if mode == "gpu":
                        note += "; unweighted GPU path uses BFS kernel"
                elif func == "BellmanFord":
                    run_paths = lambda: eg.multi_source_bellman_ford(g, source_nodes, weight="weight", target=None)
                    note = plan["note"]
                else:
                    run_paths = lambda: eg.multi_source_dijkstra(g, source_nodes, weight="weight", target=None)
                    note = plan["note"]

                easygraph_warmup(run_paths)
                barrier(cooldown)
                dists, algo_s, mem = timed_algorithm(
                    run_paths,
                    sync_after=sync_after_eg_call,
                )
                reachable, checksum = summarize_sssp_result(dists)
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_or_algo(plan["kernel_key"], algo_s),
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                if mode == "cpp":
                    emit_skip(
                        backend,
                        func,
                        "GPU-enabled cpp_easygraph structural-hole bindings route to CUDA at compile time; skipped to keep CPU C++ baseline isolated",
                    )
                    continue
                plan = structural_hole_plan(func, graph_type, views)
                reset_kernel_key(plan["kernel_key"])
                n = plan["n"]
                t0 = time.perf_counter()
                g = to_mode_graph(n, plan["edges"], plan["directed"])
                t1 = time.perf_counter()
                fn = {
                    "EffectiveSize": eg.effective_size,
                    "Efficiency": eg.efficiency,
                    "Constraint": eg.constraint,
                    "Hierarchy": eg.hierarchy,
                }[func]
                run_structural = lambda: fn(g, weight=None)
                easygraph_warmup(run_structural)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(run_structural, sync_after=sync_after_eg_call)
                count, total, mean = summarize_numeric_mapping(vals)
                note = plan["note"]
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, func, vals, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    kernel_or_algo(plan["kernel_key"], algo_s),
                    correctness=f"nodes={count}, sum={total:.9g}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "KCore":
                reset_kernel_key("kcore")
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = to_mode_graph(n, base_edges, False)
                t1 = time.perf_counter()
                kcore_note_suffix = ""

                def run_kcore_once():
                    return eg.k_core(g)

                easygraph_warmup(run_kcore_once)
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    run_kcore_once,
                    sync_after=sync_after_eg_call,
                )
                if hasattr(vals, "tolist"):
                    vals = vals.tolist()
                if isinstance(vals, (list, tuple)) and len(vals) == n + 1:
                    vals = list(vals)[1:]
                if isinstance(vals, dict):
                    seq = list(vals.values())
                elif isinstance(vals, (list, tuple)):
                    seq = list(vals)
                else:
                    seq = []
                total = float(sum(seq)) if seq else 0.0
                max_core = int(max(seq)) if seq else 0
                note = "undirected projection"
                if mode != "gpu":
                    note = (note + "; " if note else "") + "cpu backend kernel=algorithm"
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64")
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("kcore", algo_s),
                    correctness=f"nodes={len(seq)}, sum={total:.9g}, max={max_core}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "BC":
                reset_kernel_key("bc")
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                source_nodes = pick_sources(n, bc_sources)
                if not source_nodes:
                    emit_skip(backend, "BC", "empty source set")
                    continue
                t0 = time.perf_counter()
                g = to_mode_graph(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()
                easygraph_warmup(
                    lambda: eg.betweenness_centrality(
                        g,
                        weight=None,
                        sources=source_nodes,
                        normalized=False,
                        endpoints=False,
                    )
                )
                barrier(cooldown)
                vals, algo_s, mem = timed_algorithm(
                    lambda: eg.betweenness_centrality(
                        g,
                        weight=None,
                        sources=source_nodes,
                        normalized=False,
                        endpoints=False,
                    ),
                    sync_after=sync_after_eg_call,
                )
                if hasattr(vals, "tolist"):
                    vals = vals.tolist()
                if isinstance(vals, dict):
                    seq = list(vals.values())
                elif isinstance(vals, (list, tuple)):
                    seq = list(vals)
                else:
                    seq = []
                total = float(sum(seq)) if seq else 0.0
                note = f"source-sampled; sources={len(source_nodes)}; normalized=False"
                if mode != "gpu":
                    note += "; cpu backend kernel=algorithm"
                else:
                    note += "; unweighted GPU path uses BFS-Brandes kernel"
                detail = write_vector_detail(backend, "BC", vals, n)
                emit_metrics(
                    backend,
                    "BC",
                    t1 - t0,
                    algo_s,
                    kernel_or_algo("bc", algo_s),
                    correctness=f"nodes={len(seq)}, sum={total:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )
        except Exception as e:
            emit_exception(backend, func, e)


def bench_nx_cugraph(views, graph_type, functions, pr_alpha, pr_max_iter, pr_tol, cooldown, warmup, sssp_sources, bc_sources, closeness_sources):
    backend = "nx-cugraph"
    os.environ["EGGPU_ALLOW_CUDA_SYNC"] = "TRUE"
    if all(func in STRUCTURAL_HOLE_FUNCTIONS for func in functions):
        for func in functions:
            emit_skip(backend, func, "nx-cugraph has no native Burt structural-hole metric API")
        return
    try:
        import nx_cugraph  # noqa: F401
        import networkx as nx
    except Exception as e:
        for func in functions:
            emit_skip(backend, func, f"nx-cugraph import failed: {e}")
        return

    for func in functions:
        try:
            if func == "PageRank":
                n, directed_edges, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_networkx(n, directed_edges if graph_type == "directed" else undirected_edges, graph_type == "directed")
                t1 = time.perf_counter()
                warmup_call(
                    lambda: nx_cugraph_call(nx.pagerank, g, alpha=pr_alpha, max_iter=pr_max_iter, tol=pr_tol),
                    warmup,
                )
                barrier(cooldown)
                ranks, algo_s, mem = timed_algorithm(
                    lambda: nx_cugraph_call(nx.pagerank, g, alpha=pr_alpha, max_iter=pr_max_iter, tol=pr_tol),
                    sync_after=False,
                )
                emit_metrics(
                    backend,
                    "PageRank",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sum={sum(ranks.values()):.9g}" + write_vector_detail(backend, "PageRank", ranks, n),
                    notes=f"alpha={pr_alpha}, max_iter={pr_max_iter}, tol={pr_tol}; kernel uses algorithm timer (backend does not expose kernel)",
                    memory=mem,
                )

            elif func == "MST":
                n, _, undirected_edges = views["all_vertices"]
                t0 = time.perf_counter()
                g = build_networkx(n, undirected_edges, False, weighted=True)
                t1 = time.perf_counter()
                warmup_call(lambda: nx_cugraph_call(nx.minimum_spanning_tree, g, weight="weight"), warmup)
                barrier(cooldown)
                def run_mst():
                    try:
                        tree = nx_cugraph_call(nx.minimum_spanning_tree, g, weight="weight")
                        w = int(sum(data.get("weight", 1) for _, _, data in tree.edges(data=True)))
                        n = "undirected projection; kernel uses algorithm timer (backend does not expose kernel)"
                    except Exception as e:
                        w = cugraph_mst_weight(undirected_edges)
                        n = f"undirected projection; native cuGraph fallback (nx-cugraph path failed: {concise_error(e)}); kernel uses algorithm timer"
                    return w, n

                (weight, note), algo_s, mem = timed_algorithm(run_mst, sync_after=False)
                emit_metrics(
                    backend,
                    "MST",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"weight={weight}",
                    notes=note,
                    memory=mem,
                )

            elif func == "LCC":
                n, _, undirected_edges = views["clean"]
                t0 = time.perf_counter()
                g = build_networkx(n, undirected_edges, False)
                t1 = time.perf_counter()
                warmup_call(lambda: nx_cugraph_call(nx.clustering, g), warmup)
                barrier(cooldown)
                def run_lcc():
                    try:
                        v = nx_cugraph_call(nx.clustering, g)
                        n = "undirected projection; kernel uses algorithm timer (backend does not expose kernel)"
                    except Exception as e:
                        v = cugraph_lcc_values(undirected_edges)
                        n = f"undirected projection; native cuGraph fallback (nx-cugraph path failed: {concise_error(e)}); kernel uses algorithm timer"
                    return v, n

                (vals, note), algo_s, mem = timed_algorithm(run_lcc, sync_after=False)
                mean = sum(vals.values()) / len(vals) if vals else 0.0
                detail = write_vector_detail(backend, "LCC", vals, n)
                emit_metrics(
                    backend,
                    "LCC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"vertices={len(vals)}, mean={mean:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "Closeness":
                emit_skip(
                    backend,
                    "Closeness",
                    "nx-cugraph supported-algorithm list does not include closeness_centrality; "
                    "skipped to avoid measuring an unsupported backend or CPU fallback",
                )

            elif func in {"WCC", "SCC"}:
                plan = component_semantics(func, graph_type, views)
                n = plan["n"]
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["build_directed"])
                t1 = time.perf_counter()
                if plan["nx_kind"] == "scc":
                    nx_component_call = lambda: list(nx_cugraph_call(nx.strongly_connected_components, g))
                else:
                    nx_component_call = lambda: list(nx_cugraph_call(nx.connected_components, g))
                warmup_call(nx_component_call, warmup)
                barrier(cooldown)

                def run_components():
                    try:
                        comps = nx_component_call()
                        c = len(comps)
                        n_note = plan["note"] + "; kernel uses algorithm timer (backend does not expose kernel)"
                    except Exception as e:
                        c = cugraph_cc_components(plan["edges"], directed=plan["cugraph_directed"])
                        n_note = (
                            plan["note"]
                            + f"; native cuGraph fallback (nx-cugraph path failed: {concise_error(e)}); "
                            + "kernel uses algorithm timer"
                        )
                    return c, n_note

                (comp_count, note), algo_s, mem = timed_algorithm(run_components, sync_after=False)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"components={comp_count}",
                    notes=note,
                    memory=mem,
                )

            elif func in PATH_SOURCE_FUNCTIONS:
                plan = path_benchmark_plan(func, graph_type, views, sssp_sources)
                n = plan["n"]
                source_nodes = plan["sources"]
                if not source_nodes:
                    emit_skip(backend, func, "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_networkx(n, plan["edges"], plan["directed"], weighted=plan["weighted"])
                t1 = time.perf_counter()

                def run_paths():
                    if func == "BFS":
                        out = {
                            s: nx_cugraph_call(
                                nx.single_source_shortest_path_length,
                                g,
                                s,
                            )
                            for s in source_nodes
                        }
                        note = (
                            plan["note"]
                            + "; nx-cugraph single_source_shortest_path_length; kernel uses algorithm timer"
                        )
                        return out, note
                    if func == "BellmanFord":
                        out = {
                            s: nx_cugraph_call(
                                nx.single_source_bellman_ford_path_length,
                                g,
                                s,
                                weight="weight",
                            )
                            for s in source_nodes
                        }
                        note = (
                            plan["note"]
                            + "; nx-cugraph single_source_bellman_ford_path_length; kernel uses algorithm timer"
                        )
                        return out, note
                    try:
                        out = cugraph_sssp_multi_source(
                            plan["edges"],
                            source_nodes,
                            directed=(graph_type == "directed"),
                        )
                        note = plan["note"] + "; native cuGraph SSSP path; kernel uses algorithm timer"
                    except Exception as e:
                        out = {
                            s: nx_cugraph_call(
                                nx.single_source_dijkstra_path_length,
                                g,
                                source=s,
                                weight="weight",
                            )
                            for s in source_nodes
                        }
                        note = (
                            plan["note"]
                            + "; nx-cugraph backend fallback "
                            f"(native cuGraph path failed: {concise_error(e)}); kernel uses algorithm timer"
                        )
                    return out, note

                warmup_call(lambda: run_paths()[0], warmup)
                barrier(cooldown)
                (dists, note), algo_s, mem = timed_algorithm(run_paths, sync_after=False)
                reachable, checksum = summarize_sssp_result(dists)
                detail = write_sssp_detail(backend, func, dists, source_nodes, n)
                emit_metrics(
                    backend,
                    func,
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"sources={len(source_nodes)}, reachable={reachable}, checksum={checksum:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func in STRUCTURAL_HOLE_FUNCTIONS:
                emit_skip(backend, func, "nx-cugraph has no native Burt structural-hole metric API")

            elif func == "KCore":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = undirected_edges
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, False)
                t1 = time.perf_counter()

                def run_kcore():
                    try:
                        vals = cugraph_kcore_values(base_edges, directed=False)
                        note = "undirected projection; native cuGraph path; kernel uses algorithm timer"
                    except Exception as e:
                        vals = nx_cugraph_call(nx.core_number, g)
                        note = (
                            "undirected projection; nx-cugraph backend fallback "
                            f"(native cuGraph path failed: {concise_error(e)}); kernel uses algorithm timer"
                        )
                    return vals, note

                warmup_call(lambda: run_kcore()[0], warmup)
                barrier(cooldown)
                (vals, note), algo_s, mem = timed_algorithm(run_kcore, sync_after=False)
                total = float(sum(vals.values())) if isinstance(vals, dict) and vals else 0.0
                max_core = int(max(vals.values())) if isinstance(vals, dict) and vals else 0
                count = len(vals) if isinstance(vals, dict) else 0
                detail = write_vector_detail(backend, "KCore", vals, n, dtype="int64") if isinstance(vals, dict) else ""
                emit_metrics(
                    backend,
                    "KCore",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={count}, sum={total:.9g}, max={max_core}" + detail,
                    notes=note,
                    memory=mem,
                )

            elif func == "BC":
                n, directed_edges, undirected_edges = views["clean"]
                base_edges = directed_edges if graph_type == "directed" else undirected_edges
                source_nodes = pick_sources(n, bc_sources)
                if not source_nodes:
                    emit_skip(backend, "BC", "empty source set")
                    continue
                t0 = time.perf_counter()
                g = build_networkx(n, base_edges, graph_type == "directed")
                t1 = time.perf_counter()

                def run_bc():
                    try:
                        vals = cugraph_bc_values(
                            base_edges,
                            directed=(graph_type == "directed"),
                            sources=source_nodes,
                            normalized=False,
                            endpoints=False,
                        )
                        note = (
                            "source-sampled exact-source native cuGraph path "
                            f"(sources={len(source_nodes)}); kernel uses algorithm timer"
                        )
                    except Exception as e:
                        vals = nx_cugraph_call(
                            nx.betweenness_centrality,
                            g,
                            k=len(source_nodes),
                            normalized=False,
                            weight=None,
                            endpoints=False,
                            seed=0,
                        )
                        note = (
                            f"source-sampled k={len(source_nodes)} with fixed seed; "
                            "nx-cugraph backend fallback "
                            f"(native cuGraph path failed: {concise_error(e)}); kernel uses algorithm timer"
                        )
                    return vals, note

                warmup_call(lambda: run_bc()[0], warmup)
                barrier(cooldown)
                (vals, note), algo_s, mem = timed_algorithm(run_bc, sync_after=False)
                total = float(sum(vals.values())) if isinstance(vals, dict) and vals else 0.0
                count = len(vals) if isinstance(vals, dict) else 0
                detail = write_vector_detail(backend, "BC", vals, n) if isinstance(vals, dict) else ""
                emit_metrics(
                    backend,
                    "BC",
                    t1 - t0,
                    algo_s,
                    algo_s,
                    correctness=f"nodes={count}, sum={total:.9g}" + detail,
                    notes=note,
                    memory=mem,
                )
        except Exception as e:
            if is_nx_cugraph_not_implemented(e):
                emit_skip(backend, func, f"not implemented by nx-cugraph: {e}")
            else:
                emit_exception(backend, func, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edge_path")
    ap.add_argument("graph_type", choices=["directed", "undirected"])
    ap.add_argument(
        "--backend",
        choices=["all", "igraph", "networkx", "EGGPU", "easygraph-cpu", "easygraph-cpp", "nx-cugraph"],
        default="all",
    )
    ap.add_argument("--function", choices=["all", *FUNCTION_ORDER, *LEGACY_FUNCTION_ALIASES], default="all")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--pr-alpha", type=float, default=0.75)
    ap.add_argument("--pr-tol", type=float, default=1e-6)
    ap.add_argument("--pr-max-iter", type=int, default=200)
    ap.add_argument("--easygraph-gpu-backend", default=os.environ.get("EASYGRAPH_GPU_BACKEND", "mine"))
    ap.add_argument("--warmup", type=int, default=1, help="Warmup calls before each timed run (easygraph series only).")
    ap.add_argument("--easygraph-warmup", type=int, default=1)
    ap.add_argument("--sssp-sources", type=int, default=8, help="Number of deterministic sources for SSSP.")
    ap.add_argument("--bc-sources", type=int, default=16, help="Number of deterministic sources for BC source-sampled mode.")
    ap.add_argument(
        "--closeness-sources",
        type=int,
        default=0,
        help=(
            "If >0, run Closeness in sampled-target exact mode on this many deterministic nodes. "
            "Default 0 keeps exact all-node Closeness semantics."
        ),
    )
    ap.add_argument("--cooldown", type=float, default=0.2, help="Sleep seconds between build/algorithm phases for run isolation.")
    args = ap.parse_args()

    if args.function == "all":
        functions = list(FUNCTION_ORDER)
    elif args.function in LEGACY_FUNCTION_ALIASES:
        functions = list(LEGACY_FUNCTION_ALIASES[args.function])
    else:
        functions = [args.function]

    views = load_graph(args.edge_path)
    n_clean, directed_clean, undirected_clean = views["clean"]
    n_all, directed_all, undirected_all = views["all_vertices"]
    print(
        f"[data] clean_nodes={n_clean} all_nodes={n_all} directed_edges={len(directed_clean)} undirected_unique={len(undirected_clean)} graph_type={args.graph_type}",
        flush=True,
    )
    print(
        "[data] self-loop rows are removed during benchmark graph construction; raw files are left unchanged.",
        flush=True,
    )
    print(
        f"[params] pagerank alpha={args.pr_alpha} tol={args.pr_tol} max_iter={args.pr_max_iter}",
        flush=True,
    )
    print(
        f"[params] easygraph_series_warmup={args.warmup} easygraph_warmup={args.easygraph_warmup} "
        f"sssp_sources={args.sssp_sources} bc_sources={args.bc_sources} closeness_sources={args.closeness_sources}",
        flush=True,
    )
    print(
        "[note] timings exclude raw edge-list parsing and import time; metrics include build, e2e, kernel plus memory metrics (if available).",
        flush=True,
    )

    if args.backend in ("all", "igraph"):
        bench_igraph(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.cooldown,
            0,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
        )
    if args.backend in ("all", "networkx"):
        bench_networkx(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.cooldown,
            0,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
        )
    if args.backend in ("all", "EGGPU"):
        bench_easygraph_mode(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.warmup,
            args.easygraph_warmup,
            args.easygraph_gpu_backend,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
            args.cooldown,
            mode="gpu",
        )
    if args.backend in ("all", "easygraph-cpu"):
        bench_easygraph_mode(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.warmup,
            args.easygraph_warmup,
            args.easygraph_gpu_backend,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
            args.cooldown,
            mode="cpu",
        )
    if args.backend in ("all", "easygraph-cpp"):
        bench_easygraph_mode(
            views,
            args.graph_type,
            args.skip_cpu,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.warmup,
            args.easygraph_warmup,
            args.easygraph_gpu_backend,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
            args.cooldown,
            mode="cpp",
        )
    if args.backend in ("all", "nx-cugraph"):
        bench_nx_cugraph(
            views,
            args.graph_type,
            functions,
            args.pr_alpha,
            args.pr_max_iter,
            args.pr_tol,
            args.cooldown,
            0,
            args.sssp_sources,
            args.bc_sources,
            args.closeness_sources,
        )


if __name__ == "__main__":
    main()
