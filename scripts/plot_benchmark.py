"""Plots and tables for the C++ short-packet and fixed-point results.

Reads results/cpp/bench_*.csv, results/cpp/fixed_sweep_rm27.csv,
results/fbl_curves.csv and results/fbl_thresholds.csv. Writes
  results/benchmark_bler.png     BLER vs Eb/N0 for each (n, k) with bounds
  results/fixed_point_sweep.png  BLER vs clip level for 3-8 bit storage
  results/benchmark_table.csv    Eb/N0 for BLER 1e-3 and 1e-4, gap to the normal approximation
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
CODES = [(128, 29, "RM(2,7)"), (256, 37, "RM(2,8)"), (128, 64, "RM(3,7)")]

# Fixed categorical order: colour follows the decoder; markers and dashes carry identity too.
STYLE = {
    "polar":  dict(color="#2a78d6", marker="o", ls="-",  label="Polar + CRC-6, SCL-32"),
    "exact":  dict(color="#1baf7a", marker="^", ls="-",  label="RM, RPA exact (float)"),
    "minsum": dict(color="#eda100", marker="v", ls="--", label="RM, RPA min-sum (float)"),
    "fixed5": dict(color="#e87ba4", marker="D", ls="-.", label="RM, RPA min-sum, 5-bit integer"),
    "fixed4": dict(color="#008300", marker="s", ls=":",  label="RM, RPA min-sum, 4-bit integer"),
}
BOUNDS = {
    "converse": dict(color="#3d3c38", ls=":",  lw=1.4, label="Meta-converse (any code)"),
    "normal":   dict(color="#3d3c38", ls="--", lw=1.2, label="Normal approximation"),
    "rcus":     dict(color="#8a8981", ls="-",  lw=1.2, label="RCU-s bound (random codes)"),
}


def family(dec):
    if dec.startswith("scl"):
        return "polar"
    if dec == "rpa_exact":
        return "exact"
    if dec == "rpa_minsum":
        return "minsum"
    if dec.startswith("rpa_fixed5"):
        return "fixed5"
    if dec.startswith("rpa_fixed4"):
        return "fixed4"
    return None


def load_sims():
    data = defaultdict(list)
    rows = [row for path in sorted((R / "cpp").glob("bench_*.csv")) for row in csv.DictReader(open(path))]
    for row in rows:
        fam = family(row["decoder"])
        if fam is None or int(row["block_errors"]) == 0:
            continue
        data[(int(row["n"]), int(row["k"]), fam)].append(
            (float(row["ebn0_db"]), float(row["bler"]), int(row["block_errors"]), int(row["blocks"])))
    return data


def crossing(pts, target, min_errors=10):
    pts = [p for p in pts if p[2] >= min_errors]
    for a, b in zip(pts, pts[1:]):
        if a[1] >= target > b[1]:
            f = (np.log10(target) - np.log10(a[1])) / (np.log10(b[1]) - np.log10(a[1]))
            return a[0] + f * (b[0] - a[0]), (a[2], b[2])
    return None, None


def style_axes(ax, title, show_y):
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(0, 4.5)
    ax.set_xticks(np.arange(0, 4.01, 0.5))  # no 4.5 label, so adjacent panels do not collide
    ax.set_xlabel("$E_b/N_0$ (dB)")
    if show_y:
        ax.set_ylabel("Block error rate")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    sims = load_sims()
    bounds = defaultdict(lambda: defaultdict(list))
    if (R / "fbl_curves.csv").exists():
        for row in csv.DictReader(open(R / "fbl_curves.csv")):
            key = (int(row["n"]), int(row["k"]))
            bounds[key]["eb"].append(float(row["ebn0_db"]))
            for b in BOUNDS:
                bounds[key][b].append(float(row[b]))

    # 1. BLER curves with bounds
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, (n, k, name) in zip(axes, CODES):
        bd = bounds.get((n, k))
        if bd:
            for b, st in BOUNDS.items():
                ax.plot(bd["eb"], bd[b], **st)
        for fam, st in STYLE.items():
            pts = sims.get((n, k, fam))
            if pts:
                x, y = zip(*[(p[0], p[1]) for p in pts])
                ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=2, ms=5,
                        label=st["label"])
        style_axes(ax, f"(n, k) = ({n}, {k}):  {name} vs polar", ax is axes[0])
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Short packets over BPSK/AWGN: RM + RPA vs CRC-aided polar vs finite-blocklength bounds",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(R / "benchmark_bler.png", dpi=150)

    # 2. Fixed-point sweep (RM(2,7) at 2.75 dB)
    sweep = defaultdict(list)
    ref = {}
    for row in csv.DictReader(open(R / "cpp" / "fixed_sweep_rm27.csv"),
                              fieldnames="scheme,code,n,k,decoder,ebn0_db,blocks,block_errors,bler,seconds".split(",")):
        d = row["decoder"]
        if d.startswith("rpa_fixed"):
            bits = int(d[len("rpa_fixed"):].split("_")[0])
            clip = float(d.split("_clip")[1])
            sweep[bits].append((clip, float(row["bler"])))
        else:
            ref[d] = float(row["bler"])
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (bits, pts), col in zip(sorted(sweep.items()), ramp):
        x, y = zip(*sorted(pts))
        ax.plot(x, y, color=col, marker="o", lw=2, ms=5, label=f"{bits}-bit integer")
    ax.axhline(ref["rpa_minsum"], color="#3d3c38", ls="--", lw=1.2, label="Float min-sum")
    ax.axhline(ref["rpa_exact"], color="#3d3c38", ls=":", lw=1.4, label="Float exact box-plus")
    ax.set_yscale("log")
    ax.set_xlabel("Quantiser clip level (multiples of the BPSK amplitude)")
    ax.set_ylabel("Block error rate")
    ax.set_title("RM(2,7) at $E_b/N_0$ = 2.75 dB: storage width and clipping", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "fixed_point_sweep.png", dpi=150)

    # 3. Table: Eb/N0 at BLER 1e-3 and gap to the normal approximation
    thr = {}
    if (R / "fbl_thresholds.csv").exists():
        for row in csv.DictReader(open(R / "fbl_thresholds.csv")):
            thr[(int(row["n"]), int(row["k"]), float(row["eps"]))] = row
    rows = []
    for n, k, name in CODES:
        t3, t4 = thr.get((n, k, 1e-3), {}), thr.get((n, k, 1e-4), {})
        na = float(t3["normal"]) if t3 else None
        for fam, st in STYLE.items():
            pts = sims.get((n, k, fam), [])
            x3, e3 = crossing(pts, 1e-3)
            x4, e4 = crossing(pts, 1e-4)
            rows.append({"n": n, "k": k, "code": name, "scheme": st["label"],
                         "ebn0_at_1e-3": None if x3 is None else round(float(x3), 2),
                         "gap_to_normal_approx_db": None if x3 is None or na is None else round(float(x3 - na), 2),
                         "errors_bracketing_1e-3": e3,
                         "ebn0_at_1e-4": None if x4 is None else round(float(x4), 2),
                         "errors_bracketing_1e-4": e4})
        for b in BOUNDS:
            if t3:
                rows.append({"n": n, "k": k, "code": name, "scheme": BOUNDS[b]["label"],
                             "ebn0_at_1e-3": float(t3[b]),
                             "gap_to_normal_approx_db": round(float(t3[b]) - na, 2),
                             "errors_bracketing_1e-3": "",
                             "ebn0_at_1e-4": float(t4[b]) if t4 else None,
                             "errors_bracketing_1e-4": ""})
    with open(R / "benchmark_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
