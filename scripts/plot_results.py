#!/usr/bin/env python3
"""Plot benchmark results.

Works on the thesis results (results/thesis/throughput.csv) and on CSV output of
`rtree_bench --csv` / scripts/run_benchmarks.py.

    python scripts/plot_results.py results/thesis/throughput.csv --out results/figures
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# Categorical slots (validated for colour-vision deficiency), baseline in neutral grey.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BASELINE = "#8a8983"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 13, "axes.axisbelow": True, "axes.titleweight": "bold", "legend.frameon": False,
})


def load(path):
    df = pd.read_csv(path)
    if "selectivity_pct" not in df:
        df["selectivity_pct"] = df["queries"].map(
            lambda q: float(re.search(r"c([\d.]+)%", q).group(1)) if re.search(r"c([\d.]+)%", q) else float("nan"))
    df["dataset"] = df["dataset"].map(lambda d: os.path.splitext(str(d))[0])
    return df


def fanout_plot(df, out, sel):
    lines = [("D0", BASELINE), ("D1-AVX2+", SERIES[0]), ("D2-AVX2+", SERIES[1]),
             ("D1-AVX512+", SERIES[2]), ("D2-AVX512+", SERIES[3])]
    sets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 4.3), squeeze=False)
    for ax, ds in zip(axes[0], sets):
        d = df[(df.dataset == ds) & (df.selectivity_pct == sel)]
        for name, color in lines:
            s = d[d.variant == name].sort_values("fanout")
            if s.empty:
                continue
            ax.plot(s.fanout, s.throughput_qps / 1000, color=color, lw=2, marker="o", ms=5,
                    label=name, ls="--" if name == "D0" else "-")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(d.fanout.unique()), [str(x) for x in sorted(d.fanout.unique())])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(f"{ds} · selectivity {sel:g}%", loc="left")
        ax.set_xlabel("fanout (entries per node)")
        ax.set_ylabel("throughput (k queries / s)")
    axes[0][-1].legend(loc="lower center", ncol=1, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fanout.png"), dpi=150)
    plt.close(fig)


def speedup_plot(df, out, fanout):
    bars = [("D1-AVX512", SERIES[0]), ("D2-AVX512", SERIES[1]),
            ("D1-AVX512+", SERIES[2]), ("D2-AVX512+", SERIES[3])]
    sets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 4.3), squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], sets):
        d = df[(df.dataset == ds) & (df.fanout == fanout)]
        base = d[d.variant == "D0"].set_index("selectivity_pct").throughput_qps
        sels = sorted(base.index)
        w = 0.8 / len(bars)
        for i, (name, color) in enumerate(bars):
            s = d[d.variant == name].set_index("selectivity_pct").throughput_qps.reindex(sels) / base.reindex(sels)
            xs = [k + (i - (len(bars) - 1) / 2) * w for k in range(len(sels))]
            ax.bar(xs, s.values, width=w - 0.03, color=color, label=name)
            if name == "D2-AVX512+":
                for x, v in zip(xs, s.values):
                    ax.text(x, v + 0.03, f"{v:.2f}×", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.axhline(1.0, color=BASELINE, lw=1.2, ls="--", label="D0 (scalar baseline)")
        ax.set_xticks(range(len(sels)), [f"{s:g}%" for s in sels])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{ds} · fanout {fanout}", loc="left")
        ax.set_xlabel("query selectivity")
    axes[0][0].set_ylabel("speedup over scalar D0")
    axes[0][-1].legend(loc="upper right", fontsize=9.5, ncol=2)
    for ax in axes[0]:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "speedup.png"), dpi=150)
    plt.close(fig)


def leaves_plot(df, out, fanout, sel):
    pairs = ["D1-AVX2", "D2-AVX2", "D1-AVX512", "D2-AVX512"]
    sets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 4.3), squeeze=False)
    for ax, ds in zip(axes[0], sets):
        d = df[(df.dataset == ds) & (df.fanout == fanout) & (df.selectivity_pct == sel)].set_index("variant")
        if d.empty:
            continue
        inner = [d.throughput_qps.get(p, float("nan")) / 1000 for p in pairs]
        full = [d.throughput_qps.get(p + "+", float("nan")) / 1000 for p in pairs]
        x = range(len(pairs))
        ax.bar([k - 0.2 for k in x], inner, width=0.37, color=SERIES[0], label="SIMD in internal nodes only")
        ax.bar([k + 0.2 for k in x], full, width=0.37, color=SERIES[1], label="+ SIMD leaf scan")
        for k, (a, b) in enumerate(zip(inner, full)):
            ax.text(k + 0.2, b, f"+{(b / a - 1) * 100:.0f}%", ha="center", va="bottom", fontsize=9, color=INK2)
        if "D0" in d.index:
            ax.axhline(d.throughput_qps["D0"] / 1000, color=BASELINE, lw=1.2, ls="--", label="D0 (scalar baseline)")
        ax.set_xticks(list(x), pairs)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(full) * 1.3)
        ax.set_title(f"{ds} · fanout {fanout} · selectivity {sel:g}%", loc="left")
        ax.set_ylabel("throughput (k queries / s)")
    axes[0][0].legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "simd_leaves.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--fanout", type=int, default=64)
    ap.add_argument("--selectivity", type=float, default=0.1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.csv)
    if df.fanout.nunique() > 1:
        fanout_plot(df, a.out, a.selectivity)
    speedup_plot(df, a.out, a.fanout)
    leaves_plot(df, a.out, a.fanout, a.selectivity)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
