"""Plot BER/BLER curves from results/results.csv into results/*.png."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

COLORS = {4: "#1f77b4", 5: "#d62728", 6: "#2ca02c"}
MARKERS = {1: "o", 2: "s"}


def load(path):
    data = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader(open(path)):
        if int(row["bit_errors"]) == 0:
            continue  # no errors observed: not a measured point
        key = (int(row["r"]), int(row["m"]), int(row["n"]), int(row["k"]), row["decoder"])
        for col in ("ebn0_db", "ber", "bler"):
            data[key][col].append(float(row[col]))
    return data


def uncoded(ebn0_db):
    return 0.5 * erfc(np.sqrt(10 ** (ebn0_db / 10)))


def style_axes(ax, ylabel, title):
    ax.set_yscale("log")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(0, 10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def label(r, m, n, k):
    return f"RM({r},{m}) [{n},{k}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/results.csv")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()
    data = load(args.csv)
    out = Path(args.outdir)
    x = np.linspace(0, 10, 200)

    # 1. Majority-logic (hard decision) BER for every code.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, uncoded(x), "k--", label="Uncoded BPSK")
    for (r, m, n, k, dec), d in sorted(data.items()):
        if dec == "majority":
            ax.plot(d["ebn0_db"], d["ber"], marker=MARKERS[r], color=COLORS[m],
                    ls="-" if r == 1 else ":", label=label(r, m, n, k))
    style_axes(ax, "Bit error rate", "Reed majority-logic decoding (hard decision), BPSK/AWGN")
    fig.tight_layout()
    fig.savefig(out / "ber_majority.png", dpi=150)

    # 2. Hard (majority) vs soft (ML) decoding, one panel per order r.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, r in zip(axes, (1, 2)):
        ax.plot(x, uncoded(x), "k--", label="Uncoded BPSK")
        for (rr, m, n, k, dec), d in sorted(data.items()):
            if rr != r or not any(key[:4] == (rr, m, n, k) and key[4] == "ml" for key in data):
                continue
            ax.plot(d["ebn0_db"], d["ber"], color=COLORS[m], marker=MARKERS[r],
                    ls="-" if dec == "ml" else ":",
                    label=f"{label(rr, m, n, k)} {'soft ML' if dec == 'ml' else 'majority'}")
        style_axes(ax, "Bit error rate", f"Order r = {r}: soft ML vs hard majority-logic")
    fig.tight_layout()
    fig.savefig(out / "ber_soft_vs_hard.png", dpi=150)

    # 3. Block error rate, all decoders.
    fig, ax = plt.subplots(figsize=(7, 5))
    for (r, m, n, k, dec), d in sorted(data.items()):
        ax.plot(d["ebn0_db"], d["bler"], color=COLORS[m], marker=MARKERS[r],
                ls="-" if dec == "ml" else ":",
                label=f"{label(r, m, n, k)} {'ML' if dec == 'ml' else 'maj.'}")
    style_axes(ax, "Block error rate", "Block error rate, BPSK/AWGN")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "bler_all.png", dpi=150)
    print("wrote", ", ".join(str(out / f) for f in ("ber_majority.png", "ber_soft_vs_hard.png", "bler_all.png")))


if __name__ == "__main__":
    main()
