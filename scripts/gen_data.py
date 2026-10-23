#!/usr/bin/env python3
"""Generate a synthetic point dataset and range-query files.

Lets anyone try the benchmark without downloading the TIGER datasets.
Points are a mix of Gaussian clusters and uniform noise (roughly city-like);
query windows are squares centred on random data points, sized so that each
query returns about the requested fraction of the data on average.

Example:
    python scripts/gen_data.py --points 2000000 --out data/synthetic
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=10_000)
    ap.add_argument("--selectivities", default="0.01,0.05,0.1,0.5,1", help="percent of the data per query")
    ap.add_argument("--clusters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    # 80% clustered, 20% uniform in [0, 1000]^2
    n_cl = int(a.points * 0.8)
    centers = rng.uniform(0, 1000, size=(a.clusters, 2))
    sizes = rng.uniform(2, 25, size=a.clusters)
    which = rng.integers(0, a.clusters, size=n_cl)
    clustered = centers[which] + rng.normal(size=(n_cl, 2)) * sizes[which, None]
    uniform = rng.uniform(0, 1000, size=(a.points - n_cl, 2))
    pts = np.vstack([clustered, uniform]).astype(np.float32)
    rng.shuffle(pts)

    pts_path = os.path.join(a.out, "points.csv")
    np.savetxt(pts_path, pts, fmt="%.4f", delimiter=",")
    print(f"wrote {pts_path} ({len(pts):,} points)")

    # Calibrate the window side on a sample so the average result size matches.
    sample = pts[rng.choice(len(pts), size=min(len(pts), 50_000), replace=False)]
    for sel in [float(s) for s in a.selectivities.split(",")]:
        target = sel / 100.0
        centres = pts[rng.integers(0, len(pts), size=a.queries)]
        lo, hi = 1e-4, 1000.0
        probe = centres[:200]
        for _ in range(30):  # binary search on the half-side length
            mid = (lo + hi) / 2
            frac = np.mean([np.mean((np.abs(sample[:, 0] - c[0]) <= mid) & (np.abs(sample[:, 1] - c[1]) <= mid))
                            for c in probe])
            lo, hi = (mid, hi) if frac < target else (lo, mid)
        h = (lo + hi) / 2
        path = os.path.join(a.out, f"queries_c{sel:g}%_n{a.queries}.qry")
        with open(path, "w") as f:
            for cx, cy in centres:
                f.write(f"{cx - h:.4f} {cy - h:.4f},{cx + h:.4f} {cy + h:.4f}\n")
        print(f"wrote {path} (half-side {h:.3f})")


if __name__ == "__main__":
    main()
