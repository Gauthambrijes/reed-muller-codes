"""Finite-blocklength bounds for the short-packet benchmark (BI-AWGN, uniform inputs).

Writes
  results/fbl_thresholds.csv : Eb/N0 needed by each bound for BLER 1e-3 and 1e-4
  results/fbl_curves.csv     : bound values on an Eb/N0 grid, for plotting
"""

import csv
from pathlib import Path

import numpy as np

from rmcodes.fbl import (metaconverse_bound, normal_approx_eps, rcus_bound, required_ebn0,
                         sigma_from_ebn0, block_laws)

CODES = [(128, 29), (256, 37), (128, 64)]
OUT = Path("results")


def thresholds():
    rows = []
    for n, k in CODES:
        for eps in (1e-3, 1e-4):
            row = {"n": n, "k": k, "eps": eps}
            for which in ("converse", "normal", "rcus"):
                row[which] = round(required_ebn0(n, k, eps, which, lo=0.0, hi=6.0, tol=0.005), 3)
            rows.append(row)
            print(row, flush=True)
    with open(OUT / "fbl_thresholds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def curves():
    with open(OUT / "fbl_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "k", "ebn0_db", "converse", "normal", "rcus"])
        for n, k in CODES:
            for eb in np.round(np.arange(0.0, 4.51, 0.25), 2):
                sig = float(sigma_from_ebn0(eb, k / n))
                laws = block_laws(n, sig)
                vals = [metaconverse_bound(n, k, sig, laws=laws), normal_approx_eps(n, k, sig),
                        rcus_bound(n, k, sig, iters=8)]
                w.writerow([n, k, eb, *(f"{v:.6e}" for v in vals)])
                f.flush()
            print("curves done for", n, k, flush=True)


if __name__ == "__main__":
    thresholds()
    curves()
