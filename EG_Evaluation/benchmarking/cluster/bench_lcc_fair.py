#!/usr/bin/env python3
import argparse, os, re, time, math, tempfile, subprocess
import cupy as cp
import cudf
import cugraph
"""
HOW TO RUN:
python ./benchmarking/cluster/bench_lcc_fair.py \
  ./datasets/as-skitter.txt \
  --my-bin ./basic_functions/cc_gpu \
  --dump ./datasets/mine_cc.tsv \
  --warmup 2 --repeat 3

"""
# ---------- utilities ----------

def env_info():
    try:
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        gpu = props["name"].decode()
        sm = props["major"] * 10 + props["minor"]
    except Exception:
        gpu, sm = "unknown", -1
    try:
        out = subprocess.run(
            ["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader","-i","0"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        driver = out.splitlines()[0] if out else "unknown"
    except Exception:
        driver = "unknown"
    rt = cp.cuda.runtime.runtimeGetVersion()
    cuda_rt = f"{rt//1000}.{(rt%1000)//10}"
    return gpu, sm, driver, cuda_rt

def read_and_clean(edge_path, undirected=True):
    # Ingest (I/O + parse) on GPU
    t0 = time.time()
    df = cudf.read_csv(
        edge_path, delimiter=" ",
        names=["src","dst"], dtype=["int64","int64"],
        comment="#", skip_blank_lines=True
    )
    t_read = time.time() - t0

    # Clean (self-loop drop, (min,max) canonicalization, dedup)
    t1 = time.time()
    df = df[df["src"] != df["dst"]]
    if undirected:
        lo = cudf.Series(cp.minimum(df["src"].values, df["dst"].values), dtype="int64")
        hi = cudf.Series(cp.maximum(df["src"].values, df["dst"].values), dtype="int64")
        df = cudf.DataFrame({"src": lo, "dst": hi}).drop_duplicates(ignore_index=True)
    else:
        df = df.drop_duplicates(ignore_index=True)
    t_clean = time.time() - t1

    n = int(cudf.concat([df["src"], df["dst"]], ignore_index=True).unique().size)
    m = int(len(df))
    return df, t_read, t_clean, n, m

def write_clean_tmp(df, path_hint):
    tmp = os.path.join(tempfile.gettempdir(), f"clean_{os.path.basename(path_hint)}")
    df.to_csv(tmp, sep=" ", header=False, index=False, columns=["src","dst"])
    return tmp

def degree_from_df(df):
    deg = cudf.concat([df["src"], df["dst"]], ignore_index=True).astype("int64").to_frame(name="vertex")
    deg["one"] = 1
    deg = deg.groupby("vertex").agg({"one":"sum"}).reset_index().rename(columns={"one":"degree"})
    return deg

# ---------- cuGraph path (with ingest timing) ----------

def cugraph_lcc_with_breakdown(df_clean, renumber):
    # Build
    G = cugraph.Graph(directed=False)
    tb0 = time.time()
    G.from_cudf_edgelist(df_clean, source="src", destination="dst", renumber=renumber)
    t_build = time.time() - tb0

    # Kernel (triangle_count) — CUDA event timing (GPU time)
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    cp.cuda.Device().synchronize()
    ev0.record()
    tri = cugraph.triangle_count(G)  # ['vertex','counts']
    ev1.record(); ev1.synchronize()
    t_kernel = cp.cuda.get_elapsed_time(ev0, ev1) / 1000.0

    # Degree & LCC (for correctness diff)
    deg = degree_from_df(df_clean)

    if renumber:
        # map back to original ids if needed (robust-ish across versions)
        try:
            rm = G.renumber_map
            # find internal id column (not 'vertex')
            int_cols = [c for c in rm.columns if c != "vertex"]
            if int_cols:
                tri = tri.merge(rm.rename(columns={int_cols[0]:"vertex"}), on="vertex", how="left")
        except Exception:
            # fall back to rebuilding without renumber for correctness only
            G2 = cugraph.Graph(directed=False)
            G2.from_cudf_edgelist(df_clean, "src", "dst", renumber=False)
            tri = cugraph.triangle_count(G2)

    tri = tri.rename(columns={"counts":"triangles"})
    both = deg.merge(tri, on="vertex", how="left")
    both["triangles"] = both["triangles"].fillna(0)
    both["comb2"] = (both["degree"] * (both["degree"] - 1)) / 2
    both["lcc"] = 0.0
    mask = both["comb2"] > 0
    both.loc[mask, "lcc"] = both.loc[mask, "triangles"] / both["comb2"]
    return t_build, t_kernel, both[["vertex","lcc"]]

# ---------- your binary path (build already includes ingest) ----------

def run_mine(bin_path, edge_file_for_mine, dump_path, warmup=3, repeat=5):
    # Warmups (no dump)
    for _ in range(warmup):
        subprocess.run([bin_path, edge_file_for_mine, "undirected"],
                       check=True, capture_output=True, text=True)

    times = []
    last_stdout = ""
    for i in range(repeat):
        args = [bin_path, edge_file_for_mine, "undirected"]
        if i == repeat - 1:
            args += ["--dump", dump_path]
        out = subprocess.run(args, check=True, capture_output=True, text=True)
        last_stdout = out.stdout
        m = re.search(r"GPU compute time:\s*([0-9.]+)\s*s", out.stdout)
        if m: times.append(float(m.group(1)))

    if not times:
        raise RuntimeError("Could not parse GPU compute time from your binary output.")
    times.sort()
    t_kernel_med = times[len(times)//2]

    m2 = re.search(r"Build time \(CSR\):\s*([0-9.]+)\s*s", last_stdout)
    t_build = float(m2.group(1)) if m2 else float("nan")
    t_e2e = (t_build + t_kernel_med) if not math.isnan(t_build) else float("nan")
    return t_build, t_kernel_med, t_e2e

def compare(lcc_cg, mine_dump_path):
    mine = cudf.read_csv(mine_dump_path, sep="\t", names=["vertex","lcc_mine"],
                         dtype=["int64","float64"], header=None)
    cmp = lcc_cg.merge(mine, on="vertex", how="inner").fillna(0)
    cmp["abs_err"] = (cmp["lcc"] - cmp["lcc_mine"]).abs()
    den = (cmp["lcc"].abs() + cmp["lcc_mine"].abs()).clip(lower=1e-15)
    cmp["rel_err"] = (2.0 * cmp["abs_err"]) / den
    N   = int(len(cmp))
    mae = float(cmp["abs_err"].mean())
    med = float(cmp["abs_err"].quantile(0.5))
    p95 = float(cmp["abs_err"].quantile(0.95))
    mx  = float(cmp["abs_err"].max())
    mre = float(cmp["rel_err"].mean())
    mxr = float(cmp["rel_err"].max())
    return N, mae, med, p95, mx, mre, mxr

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Fair E2E LCC benchmark (your CUDA vs cuGraph, undirected)")
    ap.add_argument("edge_path", help="Space-delimited 'src dst' edgelist")
    ap.add_argument("--my-bin", default="../../basic_functions/cc_gpu")
    ap.add_argument("--dump",   default="../../datasets/mine_cc.tsv")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()

    gpu, sm, driver, cuda_rt = env_info()
    print(f"[env ] GPU={gpu}  SM={sm}  Driver={driver}  CUDA_RT={cuda_rt}")
    print(f"[env ] cuDF={getattr(cudf,'__version__','?')}  cuGraph={getattr(cugraph,'__version__','?')}")

    # Ingest + clean (timed) once → same df_clean for both systems
    df_clean, t_read, t_clean, n, m = read_and_clean(args.edge_path, undirected=True)
    print(f"[data] cleaned: nodes={n}  edges={m} (undirected unique)")
    print(f"[cuGraph ingest] read_csv={t_read:.6f}s  clean={t_clean:.6f}s  total={t_read+t_clean:.6f}s")

    # Persist cleaned file for your binary (so input matches)
    clean_file = write_clean_tmp(df_clean, args.edge_path)
    print(f"[data] temp file for your binary: {clean_file}")

    # cuGraph renumber=False (alignment for correctness)
    cg_build_nr, cg_kernel_nr, lcc_nr = cugraph_lcc_with_breakdown(df_clean, renumber=False)
    cg_e2e_nr = t_read + t_clean + cg_build_nr + cg_kernel_nr
    print(f"[cuGraph no-renum] build={cg_build_nr:.6f}s  kernel={cg_kernel_nr:.6f}s  e2e={cg_e2e_nr:.6f}s")

    # cuGraph renumber=True (best-case performance)
    cg_build_r, cg_kernel_r, _ = cugraph_lcc_with_breakdown(df_clean, renumber=True)
    cg_e2e_r = t_read + t_clean + cg_build_r + cg_kernel_r
    print(f"[cuGraph  renum ] build={cg_build_r:.6f}s  kernel={cg_kernel_r:.6f}s  e2e={cg_e2e_r:.6f}s")

    # Your binary (build already includes ingest)
    my_build, my_kernel_med, my_e2e = run_mine(args.my_bin, clean_file, args.dump, args.warmup, args.repeat)
    print(f"[mine           ] build(CSR+ingest)={my_build:.6f}s  kernel={my_kernel_med:.6f}s  e2e={my_e2e:.6f}s")

    # Correctness (vs cuGraph no-renum)
    N, mae, med, p95, mx, mre, mxr = compare(lcc_nr, args.dump)
    print(f"[match] nodes={N}  MAE={mae:.3e}  median|err|={med:.3e}  p95|err|={p95:.3e}  max|err|={mx:.3e}")
    print(f"[match] mean rel err={mre:.3e}  max rel err={mxr:.3e}")

    # Throughput & speedups (kernel + e2e)
    print(f"[thru ] edges={m}  mine={(m/my_kernel_med):.2e} e/s  cuGraph(nr)={(m/cg_kernel_nr):.2e} e/s  cuGraph(r)={(m/cg_kernel_r):.2e} e/s")
    print(f"[perf ] kernel speedup (cuGraph/mine): no-renum={cg_kernel_nr/my_kernel_med:.2f}×  renum={cg_kernel_r/my_kernel_med:.2f}×")
    print(f"[perf ] e2e speedup (cuGraph/mine): no-renum={cg_e2e_nr/my_e2e:.2f}×  renum={cg_e2e_r/my_e2e:.2f}×")

if __name__ == "__main__":
    main()
