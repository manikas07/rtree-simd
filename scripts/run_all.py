#!/usr/bin/env python3
"""Compile every variant and run the experiment grid (fanouts x datasets x query files).

Replaces the per-variant run scripts used during the thesis. Writes one CSV row per
run and, unless --no-plots is given, draws the figures with plot_results.py.

    python scripts/run_all.py --data roads.csv --queries "data/queries/tiger/*.qry"
    python scripts/run_all.py --data roads.csv,edges.csv --fanouts 64 --variants D0,D2_512_plus

Variants whose intrinsics the CPU does not support (e.g. AVX-512) are skipped.
"""
import argparse
import csv
import glob
import os
import platform
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXE = ".exe" if platform.system() == "Windows" else ""

VARIANTS = {
    "D0": "src/D0/D0.cpp",
    "D1_256": "src/D1/D1_256.cpp",
    "D1_256_plus": "src/D1/D1_256_plus.cpp",
    "D1_512": "src/D1/D1_512.cpp",
    "D1_512_plus": "src/D1/D1_512_plus.cpp",
    "D2_256": "src/D2/D2_256.cpp",
    "D2_256_plus": "src/D2/D2_256_plus.cpp",
    "D2_512": "src/D2/D2_512.cpp",
    "D2_512_plus": "src/D2/D2_512_plus.cpp",
}


def label(variant):
    """D2_512_plus -> D2-AVX512+ (the names used in the thesis figures)."""
    m = re.match(r"(D\d)(?:_(256|512))?(_plus)?$", variant)
    layout, width, plus = m.groups()
    if not width:
        return layout
    return f"{layout}-{'AVX2' if width == '256' else 'AVX512'}{'+' if plus else ''}"


def rel(path):
    """Path relative to the repo root. Everything is run with cwd=ROOT and relative
    paths, because MinGW's linker and the programs' ifstream cannot handle non-ASCII
    (e.g. Greek) folder names in absolute Windows paths."""
    try:
        return os.path.relpath(os.path.abspath(path), ROOT)
    except ValueError:  # different drive on Windows
        return path


def compile_variant(name, cxx, build_dir):
    exe = os.path.join(build_dir, name + EXE)
    cmd = [cxx, "-O3", "-march=native", "-o", rel(exe), VARIANTS[name]]
    if platform.system() == "Windows":
        # GCC on Windows cannot align AVX values on the stack (GCC bug 54412) and the
        # programs may crash; this assembler flag makes every vector move unaligned-safe.
        cmd.insert(1, "-Wa,-muse-unaligned-vector-move")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        err = (r.stderr or r.stdout).strip().splitlines()
        avx512 = any("avx512" in l.lower() or "target specific option" in l for l in err)
        reason = "CPU has no AVX-512" if avx512 else "compile error"
        print(f"  skip {name}: {reason}", file=sys.stderr)
        if not avx512:
            print("    " + "\n    ".join(err[:8]), file=sys.stderr)
        return None
    return exe


def run(exe, fanout, data, query):
    r = subprocess.run([exe, str(fanout), rel(data), os.devnull, rel(query)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return None  # e.g. illegal instruction on a CPU without AVX-512
    t = re.search(r"Total Elapsed Time: ([\d.eE+-]+) ms", r.stdout)
    m = re.search(r"Total Matched Points: (\d+)", r.stdout)
    return (float(t.group(1)), int(m.group(1))) if t and m else None


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="points file(s), comma-separated")
    ap.add_argument("--queries", default=os.path.join(ROOT, "data", "queries", "tiger", "*.qry"),
                    help="query file(s): glob pattern or comma-separated list")
    ap.add_argument("--fanouts", default="16,32,64,128,256,512,1024")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "local"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    datasets = a.data.split(",")
    queries = sorted(q for p in a.queries.split(",") for q in (glob.glob(p) or [p]))
    fanouts = [int(f) for f in a.fanouts.split(",")]
    build_dir = os.path.join(ROOT, "build")
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)

    if shutil.which(a.cxx) is None:
        sys.exit(f"compiler '{a.cxx}' not found in PATH (install MinGW-w64 / g++ or pass --cxx)")
    print("compiling...", file=sys.stderr)
    exes = {v: compile_variant(v, a.cxx, build_dir) for v in a.variants.split(",")}
    exes = {v: e for v, e in exes.items() if e}

    out_csv = os.path.join(a.out, "throughput.csv")
    expected = {}  # (dataset, query) -> total matched points; all variants must agree
    ok = True
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "selectivity_pct", "variant", "fanout", "num_queries",
                    "total_ms", "throughput_qps", "total_matched"])
        for data in datasets:
            ds = os.path.splitext(os.path.basename(data))[0]
            for query in queries:
                nq = count_lines(query)
                sel = re.search(r"c([\d.]+)%", os.path.basename(query))
                sel = float(sel.group(1)) if sel else ""
                for fanout in fanouts:
                    for v, exe in exes.items():
                        res = run(exe, fanout, data, query)
                        if res is None:
                            print(f"  {v}: failed to run, skipped", file=sys.stderr)
                            continue
                        ms, matched = res
                        key = (ds, query)
                        if expected.setdefault(key, matched) != matched:
                            print(f"  MISMATCH {v} fanout={fanout} {os.path.basename(query)}: "
                                  f"{matched} != {expected[key]}", file=sys.stderr)
                            ok = False
                        qps = nq / (ms / 1000.0)
                        w.writerow([ds, sel, label(v), fanout, nq, round(ms, 3), round(qps, 1), matched])
                        f.flush()
                        print(f"{ds:>12} {os.path.basename(query):>24} fanout={fanout:<5} "
                              f"{label(v):<11} {ms:10.2f} ms {qps:12.0f} q/s", file=sys.stderr)

    print(f"results: {out_csv}", file=sys.stderr)
    if not a.no_plots:
        subprocess.run([sys.executable, os.path.join(HERE, "plot_results.py"), out_csv,
                        "--out", os.path.join(a.out, "figures")])
    if not ok:
        sys.exit("ERROR: variants returned different results")


if __name__ == "__main__":
    main()
