#!/usr/bin/env python3
import argparse, os, subprocess, tempfile, time, math
import pandas as pd


import os
os.environ["RAPIDS_NO_INITIALIZE"] = "1"  # allow manual RMM init

import rmm
# choose a pool size that comfortably fits your GPU (example: 2 GB)
rmm.reinitialize(pool_allocator=True, initial_pool_size="2GB")


"""
HOW TO RUN:

python ./benchmarking/scc/compare_vs_cugraph.py \
  ./basic_functions/gpu_graph_scc_v2 \
  ./datasets/as-skitter.txt \
  undirected \
  --warmup 1 --repeat 3
"""

# Optional: make CuPy use RMM too
try:
    import cupy as cp
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
except Exception:
    pass


try:
    import cudf, cugraph
except Exception as e:
    raise SystemExit(f"Install RAPIDS first: {e}")

def read_edgelist_to_cudf(path, directed):
    rows = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#%/c":
                continue
            parts = s.split()
            if len(parts) < 2: 
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
                rows.append((u, v))
            except:
                pass
    if not rows:
        raise SystemExit("Empty/invalid graph")

    import pandas as pd
    pdf = pd.DataFrame(rows, columns=["src","dst"])

    # 0..N-1 remap (as before)
    cats = pd.Categorical(pd.concat([pdf["src"], pdf["dst"]], ignore_index=True))
    uniq = pd.Index(cats.categories)
    remap = {int(x): i for i, x in enumerate(uniq)}
    pdf["src"] = pdf["src"].map(remap)
    pdf["dst"] = pdf["dst"].map(remap)

    # Match the standalone binary preprocessing:
    # - drop self-loops
    # - for undirected CC, canonicalize each edge to one (min, max) row before
    #   de-duplicating; the binary later builds a symmetric CSR internally.
    pdf = pdf[pdf["src"] != pdf["dst"]]
    if directed:
        pdf = pdf.drop_duplicates().reset_index(drop=True)
    else:
        lo = pdf[["src", "dst"]].min(axis=1)
        hi = pdf[["src", "dst"]].max(axis=1)
        pdf = pd.DataFrame({"src": lo, "dst": hi}).drop_duplicates().reset_index(drop=True)

    # **NEW**: int32 to save memory
    pdf = pdf.astype({"src": "int32", "dst": "int32"})

    import cudf
    return cudf.from_pandas(pdf), len(uniq), len(pdf)


def run_cugraph(df, directed, N):
    import cugraph

    # Version-compatible graph construction
    try:
        G = cugraph.DiGraph() if directed and hasattr(cugraph, "DiGraph") else cugraph.Graph(directed=directed)
    except Exception:
        G = cugraph.Graph(directed=directed)

    # No store_transposed here (it doubles memory).
    G.from_cudf_edgelist(df, source="src", destination="dst", renumber=True)

    # Warmup
    for _ in range(1):
        _ = (cugraph.strongly_connected_components(G) if directed
             else cugraph.connected_components(G))

    import time
    t0 = time.perf_counter()
    res = (cugraph.strongly_connected_components(G) if directed
           else cugraph.connected_components(G))
    try:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass
    t1 = time.perf_counter()

    # Unrenumber result back to your 0..N-1 ids (handle old/new RAPIDS)
    try:
        res = G.unrenumber(res, "vertex")
    except Exception:
        try:
            from cugraph.utilities import unrenumber
            res = unrenumber(res, "vertex", G.renumber_map)
        except Exception:
            pass

    label_col = "labels" if "labels" in res.columns else ("label" if "label" in res.columns else "component")
    res = res.to_pandas()
    if len(res) < N:
        # cuGraph builds the vertex set from the edge list. The standalone
        # binary keeps vertices that appear only in self-loops, then drops those
        # loops from traversal, so add any missing vertices back as singleton
        # components before comparing label-invariant component sizes.
        present = set(int(v) for v in res["vertex"].tolist())
        missing = [v for v in range(N) if v not in present]
        max_label = int(res[label_col].max()) if len(res) else -1
        extra = pd.DataFrame({
            "vertex": missing,
            label_col: list(range(max_label + 1, max_label + 1 + len(missing))),
        })
        res = pd.concat([res, extra], ignore_index=True)
    res = res.sort_values("vertex")
    return res[label_col].to_numpy(), (t1 - t0)




def multiset_counts(labels):
    from collections import Counter
    cnt = Counter(labels)
    return tuple(sorted(cnt.values()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", help="path to your compiled cc/scc binary")
    ap.add_argument("graph", help="edge list file (two ints per line)")
    ap.add_argument("type", choices=["directed","undirected"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    directed = (args.type=="directed")
    df, N, E = read_edgelist_to_cudf(args.graph, directed)

    # --- run your binary (GPU only) and dump labels ---
    with tempfile.TemporaryDirectory() as tmpd:
        dump = os.path.join(tmpd, "labels.tsv")
        cmd = [
            args.binary,
            args.graph,
            args.type,
            "--skip-cpu",
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            f"--dump-gpu={dump}",
            "--quiet",
        ]
        print(">>", " ".join(cmd))
        r0 = time.perf_counter()
        out = subprocess.run(cmd, capture_output=True, text=True)
        r1 = time.perf_counter()
        if out.returncode != 0:
            print(out.stdout)
            print(out.stderr)
            raise SystemExit("Your binary failed")

        # parse your labels
        gpu_lbl = pd.read_csv(dump, sep="\t", header=None, names=["v","lab"]).sort_values("v")["lab"].to_numpy()
        # sanity
        assert len(gpu_lbl)==N, "label length mismatch"

    # --- run cuGraph ---
    cu_lbl, cu_t = run_cugraph(df, directed, N)

    # --- compare label-invariant correctness via size multisets ---
    from collections import Counter
    def k_and_sizes(lbl):
        c = Counter(lbl); return len(c), tuple(sorted(c.values()))
    k_you, ms_you = k_and_sizes(gpu_lbl)
    k_cu,  ms_cu  = k_and_sizes(cu_lbl)
    ok = (k_you==k_cu and ms_you==ms_cu)

    # --- timings & throughput ---
    # Your binary printed times; we compute overall wall time as a fallback and show cuGraph time here.
    t_you = None
    for line in out.stdout.splitlines():
        if line.startswith("GPU Time:"):
            t_you = float(line.split()[2])
            break
    if t_you is None:
        t_you = (r1 - r0)

    edge_work = E if directed else 2 * E
    eps_you = edge_work / t_you / 1e6
    eps_cu  = edge_work / cu_t   / 1e6

    print("\n=== Summary ===")
    print(f"Nodes: {N:,}  Edges: {edge_work:,}  Directed: {directed}")
    print(f"Your GPU:  {t_you:.4f} s  ({eps_you:.2f} MEdges/s)")
    print(f"cuGraph:   {cu_t:.4f} s  ({eps_cu:.2f} MEdges/s)")
    print(f"Components: yours={k_you}, cuGraph={k_cu}")
    print(f"Size multisets equal? {ok}")
    if not ok:
        print("NOTE: Different component structure. Check preprocessing parity (self-loops, duplicate edges, symmetry for CC).")

if __name__ == "__main__":
    main()
