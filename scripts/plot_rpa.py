"""Plot RPA results against majority logic, soft ML and the union bound.

Reads results/results.csv (majority, ML) and results/results_rpa.csv (RPA) and
writes results/bler_rpa_r2.png and results/rm28_rpa.png.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from rmcodes import union_bound_bler

# Fixed categorical order (colour follows the decoder); markers and line styles
# carry identity too, so the plot reads without colour.
STYLE = {
    "majority":   dict(color="#2a78d6", marker="o", ls=":", label="Majority logic (hard)"),
    "ml":         dict(color="#eb6834", marker="s", ls="-", label="Soft ML"),
    "rpa":        dict(color="#1baf7a", marker="^", ls="-", label="RPA, exact box-plus"),
    "rpa_minsum": dict(color="#eda100", marker="v", ls="--", label="RPA, min-sum"),
    "rpa_q78":    dict(color="#e87ba4", marker="D", ls="-.", label="RPA, min-sum, Q7.8"),
}
BOUND = dict(color="#6b6a63", lw=1.2)


def load(*paths):
    data = defaultdict(lambda: defaultdict(list))
    for path in paths:
        if not Path(path).exists():
            continue
        for row in csv.DictReader(open(path)):
            if int(row["block_errors"]) == 0:
                continue
            key = (int(row["r"]), int(row["m"]), row["decoder"])
            for col in ("ebn0_db", "ber", "bler"):
                data[key][col].append(float(row[col]))
    return data


def finish(ax, ylabel, title, xmax, ymin):
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, 1)
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_series(ax, d, dec, col):
    st = STYLE[dec]
    ax.plot(d["ebn0_db"], d[col], color=st["color"], marker=st["marker"], ls=st["ls"],
            lw=2, ms=5, label=st["label"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    out = Path(args.outdir)
    data = load(out / "results.csv", out / "results_rpa.csv")

    # 1. Second-order codes: BLER of every decoder vs the union bound.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, m, xmax in zip(axes, (4, 5, 6), (9, 8, 8)):
        x = np.linspace(0, xmax, 200)
        ax.plot(x, union_bound_bler(2, m, x), ls=(0, (4, 3)), label="Union bound (ML)", **BOUND)
        for dec in STYLE:
            if (2, m, dec) in data:
                plot_series(ax, data[(2, m, dec)], dec, "bler")
        n, k = 2 ** m, 1 + m + m * (m - 1) // 2
        finish(ax, "Block error rate" if m == 4 else "", f"RM(2,{m})  [{n}, {k}, {2 ** (m - 2)}]", xmax, 1e-6)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Second-order RM codes over BPSK/AWGN: RPA vs ML vs majority logic", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "bler_rpa_r2.png", dpi=150)

    # 2. RM(2,8), the code in the CPA hardware study: BER and BLER.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.linspace(0, 4, 200)
    a1.plot(x, 0.5 * erfc(np.sqrt(10 ** (x / 10))), color="#1f1f1d", ls="--", lw=1.2, label="Uncoded BPSK")
    a2.plot(x, union_bound_bler(2, 8, x), ls=(0, (4, 3)), label="Union bound (ML)", **BOUND)
    a2.plot(x, union_bound_bler(2, 8, x, truncated=True), ls=":",
            label="Minimum-weight term only, $A_{64}=43180$", **BOUND)
    for dec in ("rpa", "rpa_minsum", "rpa_q78"):
        if (2, 8, dec) in data:
            plot_series(a1, data[(2, 8, dec)], dec, "ber")
            plot_series(a2, data[(2, 8, dec)], dec, "bler")
    finish(a1, "Bit error rate (information bits)", "RM(2,8) [256, 37, 64]: bit error rate", 4, 1e-6)
    finish(a2, "Block error rate", "RM(2,8) [256, 37, 64]: block error rate", 4, 1e-6)
    a1.legend(fontsize=8, loc="lower left")
    a2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "rm28_rpa.png", dpi=150)
    print("wrote", out / "bler_rpa_r2.png", "and", out / "rm28_rpa.png")


if __name__ == "__main__":
    main()
