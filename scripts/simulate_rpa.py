"""Monte Carlo simulation of RPA decoding for second-order RM codes, BPSK/AWGN.

Decoders:
  rpa          exact box-plus projection, floating point
  rpa_minsum   min-sum projection, floating point
  rpa_q78      min-sum projection, every stored LLR rounded to Q7.8
               (16-bit, step 1/256, symmetric saturation at +-32767/256)

Same channel and CSV columns as simulate.py, written to results/results_rpa.csv.
Each point runs until --target-errors block errors or the code's block cap.

Usage:
  PYTHONPATH=. python scripts/simulate_rpa.py
  PYTHONPATH=. python scripts/simulate_rpa.py --quick
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from rmcodes import ReedMullerCode, RPADecoder

# (r, m, Eb/N0 grid, block cap)
PLAN = [
    (2, 4, np.arange(0.0, 8.01, 0.5), 400_000),
    (2, 5, np.arange(0.0, 6.51, 0.5), 300_000),
    (2, 6, np.arange(0.0, 5.01, 0.5), 100_000),
    (2, 8, np.arange(0.0, 2.51, 0.5), 10_000),
]
DECODERS = {
    "rpa": dict(min_sum=False),
    "rpa_minsum": dict(min_sum=True),
    "rpa_q78": dict(min_sum=True, llr_format=(7, 8)),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-errors", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results/results_rpa.csv")
    p.add_argument("--m", type=int, nargs="*", help="only simulate RM(2, m) for these m")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    plan = [row for row in PLAN if not args.m or row[1] in args.m]
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "r", "m", "n", "k", "decoder", "ebn0_db",
                    "blocks", "bit_errors", "block_errors", "ber", "bler"])
        for r, m, grid, cap in plan:
            code = ReedMullerCode(r, m)
            if args.quick:
                cap = min(cap, 2_000)
            batch = 2_000 if m <= 6 else 500
            for name, kw in DECODERS.items():
                dec = RPADecoder(code, **kw)
                t0 = time.time()
                for ebn0 in np.round(grid, 2):
                    sigma = np.sqrt(1 / (2 * code.rate * 10 ** (ebn0 / 10)))
                    blocks = be = ble = 0
                    while ble < args.target_errors and blocks < cap:
                        b = min(batch, cap - blocks)
                        u = rng.integers(0, 2, (b, code.k), dtype=np.uint8)
                        y = 1 - 2.0 * code.encode(u) + sigma * rng.standard_normal((b, code.n))
                        errs = dec.decode(2 * y / sigma ** 2) != u
                        be += int(errs.sum()); ble += int(errs.any(axis=1).sum()); blocks += b
                    w.writerow([f"RM({r},{m})", r, m, code.n, code.k, name, ebn0,
                                blocks, be, ble, be / (blocks * code.k), ble / blocks])
                    f.flush()
                    if ble == 0:
                        break
                print(f"{code!r:28s} {name:11s} {time.time() - t0:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
