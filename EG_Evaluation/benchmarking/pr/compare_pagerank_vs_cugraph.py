#!/usr/bin/env python3
import argparse, os, re, sys, time, tempfile, subprocess
from pathlib import Path

"""
HOW TO RUN:

python ./benchmarking/pr/compare_pagerank_vs_cugraph.py   ./datasets/as-skitter.txt   --my-bin ./basic_functions/pagerank_v3   --alpha 0.85 --eps 0 --max-iter 50   --undirected --warmup 1 --renumber
"""

# ---------- optional RAPIDS memory pool (helps stability/speed) ----------
os.environ.setdefault("RAPIDS_NO_INITIALIZE", "1")
try:
    import rmm
    rmm.reinitialize(pool_allocator=True, initial_pool_size="2GB")
    import cupy as cp
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)
except Exception:
    pass

try:
    import cudf
    import cugraph
except Exception as e:
    print("ERROR: RAPIDS (cuDF/cuGraph) not available:", e, file=sys.stderr)
    sys.exit(2)

# ---------- I/O & cleaning (shared for both systems) ----------
def load_and_clean(path, undirected: bool):
    """
    Reads 'src dst' edgelist (text), ignores lines starting with # % / c.
    Remaps ids to 0..N-1, drops self-loops, and de-duplicates. For undirected
    view it canonicalizes (min,max); for directed view it preserves direction.
    Returns: pandas DataFrame (src,dst int32), N, E_canon
    """
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
                u = int(p[0]); v = int(p[1])
                if u != v:
                    rows.append((u, v))
            except:
                pass
    if not rows:
        raise SystemExit("Empty/invalid graph")

    import pandas as pd
    pdf = pd.DataFrame(rows, columns=["src","dst"])

    # compress IDs to 0..N-1 (deterministic)
    cats = pd.Categorical(pd.concat([pdf["src"], pdf["dst"]], ignore_index=True))
    uniq = pd.Index(cats.categories)
    remap = {int(x): i for i, x in enumerate(uniq)}
    pdf["src"] = pdf["src"].map(remap)
    pdf["dst"] = pdf["dst"].map(remap)

    if undirected:
        u = pdf[["src","dst"]].min(axis=1)
        v = pdf[["src","dst"]].max(axis=1)
        canon = pd.DataFrame({"src": u, "dst": v}).drop_duplicates().reset_index(drop=True)
    else:
        canon = pdf.drop_duplicates().reset_index(drop=True)
    canon = canon.astype({"src":"int32","dst":"int32"})
    N = int(uniq.size)
    E_canon = int(len(canon))
    return canon, N, E_canon

def write_clean_txt(pdf, path):
    pdf[["src","dst"]].to_csv(path, sep=" ", header=False, index=False)
    return path

# ---------- cuGraph runner ----------
def run_cugraph(canon_pdf, alpha, eps, max_iter, undirected, renumber=True):
    """
    Builds a Graph on the *directed* edge set used for PR:
      - if undirected=True: symmetrize (add reverse edges)
      - else: use canon_pdf as-is (already deduped/self-loop-free)
    Returns: (build_time_s, pr_time_s, E_dir, pagerank_pandas_or_None)
    In fixed-iterations mode (eps==0), we catch FailedToConvergeError after doing the work.
    """
    import pandas as pd
    from cugraph.exceptions import FailedToConvergeError

    if undirected:
        rev = canon_pdf.rename(columns={"src":"dst","dst":"src"})
        din = cudf.from_pandas(pd.concat([canon_pdf, rev], ignore_index=True))
        E_dir = int(len(din))
    else:
        din = cudf.from_pandas(canon_pdf)
        E_dir = int(len(din))

    # Build (store_transposed=True is best for PR)
    t0 = time.perf_counter()
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(din, source="src", destination="dst",
                         renumber=renumber, store_transposed=True)
    t1 = time.perf_counter()

    # Warmup: use a small, nonzero tol to avoid exceptions here
    try:
        cugraph.pagerank(G, alpha=alpha, tol=max(eps, 1e-3) if eps == 0 else eps, max_iter=min(max_iter, 10))
    except Exception:
        pass  # warmup is best-effort

    # Timed call
    t2 = time.perf_counter()
    result = None
    try:
        result = cugraph.pagerank(G, alpha=alpha, tol=eps, max_iter=max_iter)
        try:
            import cupy as cp; cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass
        t3 = time.perf_counter()
        pr_time = t3 - t2
    except FailedToConvergeError:
        # treat as fixed-iter run
        t3 = time.perf_counter()
        pr_time = t3 - t2
    if result is not None:
        try:
            result = G.unrenumber(result, "vertex")
        except Exception:
            pass
        result = result.to_pandas()
    return (t1 - t0), pr_time, E_dir, result

# ---------- your binary runner ----------
def parse_float_after(label, text):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(label):
            m = re.findall(r"([0-9]+(?:\.[0-9]+)?)", s)
            if m: return float(m[-1])
    return None

def parse_iter_time_seconds(text: str):
    m = re.search(r"Iter time:\s*([0-9.eE+-]+)\s*s\b", text)
    return float(m.group(1)) if m else None

def parse_throughput_edges_per_s(text: str):
    m = re.search(r"Throughput:\s*([0-9.eE+-]+)\s*edges/s", text)
    return float(m.group(1)) if m else None

def parse_iters_pair(text: str):
    m = re.search(r"PR iterations:\s*(\d+)\s*/\s*(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def run_yours(bin_path, cleaned_txt, alpha, eps, max_iter, undirected, warmup, dump_path=None):
    cmd = [
        bin_path, cleaned_txt,
        *(["--undirected"] if undirected else []),
        "--alpha", str(alpha),
        "--eps", str(eps),
        "--max-iter", str(max_iter),
        "--warmup", str(warmup),
    ]
    if dump_path:
        cmd.extend(["--dump", dump_path])
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr, file=sys.stderr)
        raise SystemExit("pagerank_v3 failed")

    stdout = out.stdout
    iter_time = parse_iter_time_seconds(stdout)
    throughput = parse_throughput_edges_per_s(stdout)
    iters, max_iters = parse_iters_pair(stdout)
    build_time = parse_float_after("Build time (CSR host):", stdout)
    e2e = (build_time + iter_time) if (build_time and iter_time) else None

    return {
        "stdout": stdout,
        "iter_time": iter_time,
        "throughput": throughput,
        "iters": iters,
        "build_time": build_time,
        "e2e": e2e,
        "wall": (t1 - t0),
    }

def read_mine_pagerank(path):
    import pandas as pd
    return pd.read_csv(path, sep="\t", header=None, names=["vertex", "pagerank_mine"])

def pagerank_error(mine_pdf, cugraph_pdf):
    import pandas as pd
    pr_col = "pagerank" if "pagerank" in cugraph_pdf.columns else (
        "PageRank" if "PageRank" in cugraph_pdf.columns else None
    )
    if pr_col is None:
        candidates = [c for c in cugraph_pdf.columns if c != "vertex"]
        if not candidates:
            return None
        pr_col = candidates[0]
    rhs = cugraph_pdf[["vertex", pr_col]].rename(columns={pr_col: "pagerank_cugraph"})
    merged = mine_pdf.merge(rhs, on="vertex", how="inner")
    if len(merged) == 0:
        return None
    diff = (merged["pagerank_mine"] - merged["pagerank_cugraph"]).abs()
    denom = merged["pagerank_cugraph"].abs().clip(lower=1e-30)
    return {
        "matched": int(len(merged)),
        "mine_rows": int(len(mine_pdf)),
        "cugraph_rows": int(len(cugraph_pdf)),
        "mae": float(diff.mean()),
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "sum_mine": float(mine_pdf["pagerank_mine"].sum()),
        "sum_cugraph": float(rhs["pagerank_cugraph"].sum()),
    }

# ---------- env info ----------
def env_info():
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return props["name"].decode()
    except Exception:
        return "unknown"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Compare pagerank_v3 vs cuGraph PageRank (kernel & e2e only)")
    ap.add_argument("edge_path", help="edge list text: 'src dst' per line")
    ap.add_argument("--my-bin", default="./basic_functions/pagerank_v3", help="path to your pagerank_v3 binary")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--undirected", action="store_true", help="treat graph as undirected: add reverse edges for both systems")
    ap.add_argument("--warmup", type=int, default=1, help="warmup for your binary")
    ap.add_argument("--renumber", action="store_true", help="cuGraph renumber=True (usually fastest)")
    args = ap.parse_args()

    print(f"[env ] GPU: {env_info()}")
    print(f"[env ] cuDF={getattr(cudf,'__version__','?')}  cuGraph={getattr(cugraph,'__version__','?')}")

    # Clean once → same canonical graph for both
    t_read0 = time.perf_counter()
    canon_pdf, N, E_canon = load_and_clean(args.edge_path, undirected=args.undirected)
    t_read1 = time.perf_counter()
    print(f"[data] cleaned: N={N:,}  E={E_canon:,}  undirected={args.undirected}")
    print(f"[ingest] host read+clean: {t_read1 - t_read0:.6f} s")

    with tempfile.TemporaryDirectory() as td:
        clean_txt = write_clean_txt(canon_pdf, os.path.join(td, "clean.txt"))
        print(f"[data] temp file for your binary: {clean_txt}")

        # cuGraph
        cg_build, cg_kernel, E_dir, cg_pr = run_cugraph(canon_pdf, args.alpha, args.eps,
                                                        args.max_iter, args.undirected,
                                                        renumber=args.renumber)
        cg_e2e = (t_read1 - t_read0) + cg_build + cg_kernel
        print(f"[cuGraph] kernel={cg_kernel:.6f}s  e2e={cg_e2e:.6f}s   (renumber={args.renumber})")

        # your binary
        mine_dump = os.path.join(td, "mine_pagerank.tsv")
        mine = run_yours(args.my_bin, clean_txt, args.alpha, args.eps,
                         args.max_iter, args.undirected, args.warmup,
                         dump_path=mine_dump)
        mine_pr = read_mine_pagerank(mine_dump)
        pr_err = pagerank_error(mine_pr, cg_pr) if cg_pr is not None else None
        # show only kernel + e2e
        print(f"[mine  ] kernel={mine['iter_time'] if mine['iter_time'] is not None else float('nan'):.6f}s  "
              f"e2e~={mine['e2e'] if mine['e2e'] is not None else float('nan'):.6f}s")

    # Summarize
    print("\n=== Summary (PageRank) ===")
    print(f"Nodes: {N:,}  Edges (directed used): {E_dir:,}  alpha={args.alpha}  eps={args.eps}  max_iter={args.max_iter}")

    k_mine = mine["iter_time"] if mine["iter_time"] else float("nan")
    k_cg   = cg_kernel
    spd_k  = (k_cg / k_mine) if (k_mine and k_mine > 0) else float("nan")
    print(f"[kernel] mine={k_mine:.6f}s   cuGraph={k_cg:.6f}s   cuGraph/mine={spd_k:.2f}×")

    e2e_m  = mine["e2e"] if mine["e2e"] else float("nan")
    e2e_c  = cg_e2e
    spd_e  = (e2e_c / e2e_m) if (e2e_m and e2e_m > 0) else float("nan")
    print(f"[e2e   ] mine={e2e_m:.6f}s   cuGraph={e2e_c:.6f}s   cuGraph/mine={spd_e:.2f}×")

    # Throughput (keep in the summary only)
    th_m   = mine["throughput"] if mine["throughput"] else (E_dir * (mine["iters"] or 0) / k_mine if (k_mine and k_mine>0 and mine["iters"]) else float("nan"))
    th_c   = (E_dir * (mine["iters"] or 0) / k_cg) if (k_cg>0 and mine["iters"]) else (E_dir / k_cg if k_cg>0 else float("nan"))
    print(f"[thru  ] edges={E_dir:,}  mine={th_m:.2e} e/s   cuGraph={th_c:.2e} e/s")
    if pr_err is not None:
        print(f"[check ] matched={pr_err['matched']:,}/{pr_err['mine_rows']:,} rows  "
              f"MAE={pr_err['mae']:.3e}  max_abs={pr_err['max_abs']:.3e}  "
              f"max_rel={pr_err['max_rel']:.3e}  "
              f"sum(mine)={pr_err['sum_mine']:.9f}  sum(cuGraph)={pr_err['sum_cugraph']:.9f}")
    else:
        print("[check ] skipped: cuGraph did not return a PageRank vector (likely fixed-iteration eps=0 mode)")

if __name__ == "__main__":
    main()
