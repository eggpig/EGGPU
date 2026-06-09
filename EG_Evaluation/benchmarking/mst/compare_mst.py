#!/usr/bin/env python3
import argparse, os, tempfile, subprocess, time, sys, re
from pathlib import Path
import pandas as pd

# Optional RAPIDS memory pool (can help stability)
try:
    os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")
    import rmm
    rmm.reinitialize(pool_allocator=True, initial_pool_size="2GB")
    import cupy as cp
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
except Exception:
    pass

"""
HOW TO RUN:

 python ./benchmarking/mst/compare_mst.py \
  ./basic_functions/mst_v2 \
  ./datasets/as-skitter.txt \
  --mode e2e
"""


# -------- load & normalize (0..N-1), undirected canonical, deterministic weights --------

def load_and_normalize_edgelist(path, force_undirected=True):
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/c":
                continue
            p = s.split()
            if len(p) >= 2:
                try:
                    rows.append((int(p[0]), int(p[1])))
                except:
                    pass
    if not rows:
        raise SystemExit("Empty/invalid graph")

    df = pd.DataFrame(rows, columns=["src", "dst"])

    # compress IDs to 0..N-1
    codes, uniques = pd.factorize(pd.concat([df["src"], df["dst"]], ignore_index=True), sort=True)
    codes = codes.astype("int32", copy=False)
    N = int(uniques.size)
    df["src"] = codes[:len(df)]
    df["dst"] = codes[len(df):]

    # drop self-loops
    df = df[df.src != df.dst]

    if force_undirected:
        u = df[["src","dst"]].min(axis=1)
        v = df[["src","dst"]].max(axis=1)
        df = pd.DataFrame({"src": u, "dst": v}).drop_duplicates().reset_index(drop=True)

    # deterministic weights on *these* labels
    w = (1 + (df.src.astype("int64") * df.dst.astype("int64")) % N).astype("int32")
    df["w"] = w
    df = df.astype({"src":"int32","dst":"int32"})
    return df, N, len(df), uniques

def write_txt(df, path, with_weight=False):
    cols = ["src","dst","w"] if with_weight else ["src","dst"]
    df[cols].to_csv(path, sep=" ", header=False, index=False)
    return path

# ---------------- B runner ----------------

def parse_float(s):
    m = re.findall(r"([0-9]+\.[0-9]+)", s)
    return float(m[-1]) if m else None

def run_binary_B(binary, edgelist_txt, no_write=False):
    base = Path(edgelist_txt).name
    default_out = f"result_{base}"
    try: Path(default_out).unlink()
    except FileNotFoundError: pass

    args = [binary, edgelist_txt, "--skip-cpu"]
    if no_write: args.append("--no-write")
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout); print(out.stderr, file=sys.stderr)
        raise SystemExit("Binary B failed")

    kernel_s = None; e2e_s = None; result_path = None
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("Device GPU runtime:") and "end-to-end" not in s:
            kernel_s = parse_float(s)
        elif "Device GPU end-to-end" in s or "E2E" in s:
            e2e_s = parse_float(s)
        elif "MST has been written to" in s:
            result_path = s.split()[-1]
    if result_path is None and not no_write and Path(default_out).exists():
        result_path = default_out

    mst_edges = []
    if result_path and Path(result_path).exists():
        with open(result_path, "r") as f:
            for line in f:
                s = line.strip().strip("()")
                if not s: continue
                p = s.split(",")
                if len(p) >= 2:
                    u = int(p[0]); v = int(p[1])
                    if u > v: u, v = v, u
                    mst_edges.append((u, v))
    return kernel_s, e2e_s, mst_edges, result_path

def weight_formula_on_edges(edges, N):
    total = 0
    for (u, v) in edges:
        total += 1 + (int(u)*int(v)) % int(N)
    return int(total)

def make_weight_map(df):
    weights = {}
    for u, v, w in df[["src", "dst", "w"]].itertuples(index=False, name=None):
        u = int(u); v = int(v)
        if u > v:
            u, v = v, u
        prev = weights.get((u, v))
        if prev is None or int(w) < prev:
            weights[(u, v)] = int(w)
    return weights

def weight_from_map(edges, weights):
    missing = []
    total = 0
    for u, v in edges:
        u = int(u); v = int(v)
        if u > v:
            u, v = v, u
        w = weights.get((u, v))
        if w is None:
            missing.append((u, v))
        else:
            total += w
    return total, missing

# ---------------- cuGraph runners ----------------

def run_cugraph_kernel_only(df):
    import cudf, cugraph, time
    cdf = cudf.from_pandas(df[["src","dst","w"]].rename(columns={"w":"weight"}))
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(cdf, source="src", destination="dst", edge_attr="weight", renumber=False)
    _ = cugraph.minimum_spanning_tree(G)
    t0 = time.perf_counter()
    res = cugraph.minimum_spanning_tree(G)
    try:
        import cupy as cp; cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    t1 = time.perf_counter()
    edf = res.view_edge_list() if hasattr(res, "view_edge_list") else res
    pdf = edf.to_pandas()
    wcol = "weight" if "weight" in pdf.columns else ("weights" if "weights" in pdf.columns else "w")
    wsum = int(pdf[wcol].astype("int64").sum())
    edges = list(zip(pdf["src"].tolist(), pdf["dst"].tolist()))
    edges = [(min(u,v), max(u,v)) for u,v in edges]
    return (t1 - t0), wsum, edges

def run_cugraph_e2e(df):
    import cudf, cugraph, time
    t0 = time.perf_counter()
    cdf = cudf.from_pandas(df[["src","dst","w"]].rename(columns={"w":"weight"}))
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(cdf, source="src", destination="dst", edge_attr="weight", renumber=False)
    res = cugraph.minimum_spanning_tree(G)
    try:
        import cupy as cp; cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    edf = res.view_edge_list() if hasattr(res, "view_edge_list") else res
    pdf = edf.to_pandas()
    t1 = time.perf_counter()
    wcol = "weight" if "weight" in pdf.columns else ("weights" if "weights" in pdf.columns else "w")
    wsum = int(pdf[wcol].astype("int64").sum())
    edges = list(zip(pdf["src"].tolist(), pdf["dst"].tolist()))
    edges = [(min(u,v), max(u,v)) for u,v in edges]
    return (t1 - t0), wsum, edges

def run_cugraph_full_from_csv(csv_path):
    import cudf, cugraph, time
    t0 = time.perf_counter()
    # read weighted canonical file
    cdf = cudf.read_csv(csv_path, sep=" ", names=["src","dst","weight"], header=None,
                        dtype=["int32","int32","int32"])
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(cdf, source="src", destination="dst", edge_attr="weight", renumber=False)
    res = cugraph.minimum_spanning_tree(G)
    try:
        import cupy as cp; cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    edf = res.view_edge_list() if hasattr(res, "view_edge_list") else res
    pdf = edf.to_pandas()
    t1 = time.perf_counter()
    wsum = int(pdf["weight"].astype("int64").sum()) if "weight" in pdf.columns else int(pdf["weights"].astype("int64").sum())
    edges = list(zip(pdf["src"].tolist(), pdf["dst"].tolist()))
    edges = [(min(u,v), max(u,v)) for u,v in edges]
    # derive N from file
    N = int(max(int(cdf["src"].max()), int(cdf["dst"].max()))) + 1
    return (t1 - t0), wsum, edges, N

# ---------------- optional CPU ref ----------------

def maybe_run_cpu(df, which="igraph"):
    if which == "igraph":
        try:
            import igraph as ig
        except Exception:
            return None
        g = ig.Graph(n=int(df[["src","dst"]].values.max())+1, directed=False)
        g.add_edges(list(map(tuple, df[["src","dst"]].to_numpy())))
        g.es["weight"] = df["w"].tolist()
        t0 = time.perf_counter(); T = g.spanning_tree(weights="weight", return_tree=True); t1 = time.perf_counter()
        return (t1 - t0), int(sum(T.es["weight"]))
    elif which == "networkx":
        try:
            import networkx as nx
        except Exception:
            return None
        G = nx.Graph()
        G.add_weighted_edges_from(df[["src","dst","w"]].itertuples(index=False, name=None))
        t0 = time.perf_counter(); T = nx.minimum_spanning_tree(G, algorithm="kruskal"); t1 = time.perf_counter()
        return (t1 - t0), int(sum(d["weight"] for *_ , d in T.edges(data=True)))
    return None

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary_b", help="path to mst binary (your B)")
    ap.add_argument("graph", help="path to edgelist (two ints per line)")
    ap.add_argument("--mode", choices=["kernel","e2e","full"], default="e2e",
                    help="kernel: kernels only; e2e: GPU alloc+H2D+kernels+D2H; full: both sides read their own WEIGHTED files")
    ap.add_argument("--cpu-baseline", choices=["none","igraph","networkx"], default="none")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    df, N, E, uniques = load_and_normalize_edgelist(args.graph, force_undirected=True)
    print(f"[DBG] Normalized graph: N={N:,}  E={E:,}")

    with tempfile.TemporaryDirectory() as td:
        # B: symmetric WEIGHTED file
        df_sym = pd.concat(
            [df[["src","dst","w"]],
             df.rename(columns={"src":"dst","dst":"src"})[["src","dst","w"]]],
            ignore_index=True,
        )
        path_b = write_txt(df_sym, os.path.join(td, "b_sym_w.txt"), with_weight=True)

        # cuGraph FULL: canonical WEIGHTED file
        path_cu = write_txt(df[["src","dst","w"]], os.path.join(td, "cu_canon_w.txt"), with_weight=True)

        print(f"[DBG] Wrote B symmetric (weighted): {path_b}")
        print(f"[DBG] Wrote cuGraph canonical (weighted): {path_cu}")

        # ----- run B -----
        no_write = args.skip_verify or (args.mode != "full")
        b_kernel, b_e2e, b_edges_raw, b_out = run_binary_B(args.binary_b, path_b, no_write=no_write)

        # B prints *original IDs from the file*; here those are already 0..N-1
        b_edges = [(min(u,v), max(u,v)) for (u,v) in b_edges_raw]
        if not no_write:
            print(f"[DBG] B result file: {b_out}")
            print(f"[DBG] B edges parsed: {len(b_edges)} (expect ~ N-1)")

        # small sanity: compare a few edge weights against file
        if not no_write and b_edges:
            sample = b_edges[:5]
            samp_df = pd.DataFrame(sample, columns=["src","dst"])
            # check either direction in symmetric file
            m1 = samp_df.merge(df_sym, on=["src","dst"], how="left")
            m2 = samp_df.rename(columns={"src":"dst","dst":"src"}).merge(df_sym, on=["src","dst"], how="left")
            w_from_file = (m1["w"].fillna(m2["w"])).astype("Int64")
            weight_map = make_weight_map(df)
            w_from_formula = [weight_map.get((min(u,v), max(u,v))) for (u,v) in sample]
            print("[DBG] B weight sanity (first 5):")
            for i, (e, wf, wff) in enumerate(zip(sample, w_from_file.tolist(), w_from_formula)):
                print(f"       {i}: edge {e}, file_w={wf}, map_w={wff}")

        # compute B time label
        if args.mode == "kernel":
            b_time = b_kernel if b_kernel is not None else float("nan")
        else:
            b_time = b_e2e if b_e2e is not None else (b_kernel if b_kernel is not None else float("nan"))

        # if verifying, compute B weight via the same formula on these labels
        b_weight = None
        b_missing = []
        if not no_write and b_edges:
            b_weight, b_missing = weight_from_map(b_edges, make_weight_map(df))
            if b_missing:
                print(f"[DBG] B edges missing from weight map: {len(b_missing)}  sample: {b_missing[:10]}")

        # ----- run cuGraph -----
        if args.mode == "kernel":
            cu_t, cu_w, cu_edges = run_cugraph_kernel_only(df)
        elif args.mode == "e2e":
            cu_t, cu_w, cu_edges = run_cugraph_e2e(df)
        else:
            cu_t, cu_w, cu_edges, N_full = run_cugraph_full_from_csv(path_cu)
            print(f"[DBG] cuGraph FULL mode N from file: {N_full} (should equal {N})")

    # optional CPU
    cpu = None
    if args.cpu_baseline != "none":
        cpu = maybe_run_cpu(df, which=args.cpu_baseline)

    # extra debug: cuGraph weight using the same host-side weight map
    cu_formula_w, cu_missing = weight_from_map(cu_edges, make_weight_map(df))
    if cu_missing:
        print(f"[DBG] cuGraph edges missing from weight map: {len(cu_missing)}  sample: {cu_missing[:10]}")
    print(f"[DBG] cuGraph MST edges: {len(cu_edges)}  map-weight: {cu_formula_w}  api-weight: {cu_w}")

    # compare sets if both available
    eq = None
    if not args.skip_verify and (b_weight is not None):
        if b_weight != cu_w:
            b_set, cu_set = set(b_edges), set(cu_edges)
            if b_set != cu_set:
                diff = list(b_set ^ cu_set)
                print(f"[DBG] Edge-set symmetric-diff size: {len(diff)}  sample: {diff[:10]}")
            else:
                print("[DBG] Edge sets equal; weight mismatch would indicate weight calc problem.")
        eq = (b_weight == cu_w)

    # summary
    label = {"kernel":"kernel-only", "e2e":"GPU end-to-end", "full":"FULL (from weighted files)"}[args.mode]
    print("\n=== Summary (MST) ===")
    print(f"Nodes: {N:,}  Edges (undirected canonical): {E:,}")
    print(f"B ({label}): {b_time:.6f} s   Throughput: {E / max(b_time,1e-9)/1e6:.2f} MEdges/s")
    if b_weight is not None:
        print(f"B MST total weight: {b_weight}")
    print(f"cuGraph ({label}): {cu_t:.6f} s   Throughput: {E / max(cu_t,1e-9)/1e6:.2f} MEdges/s   MST weight: {cu_w}")
    if cpu is not None:
        print(f"{args.cpu_baseline}:    {cpu[0]:.6f} s   MST weight: {cpu[1]}")
    if eq is not None:
        print(f"Weights equal (B vs cuGraph)? {eq}")
    print()
    
if __name__ == "__main__":
    main()
