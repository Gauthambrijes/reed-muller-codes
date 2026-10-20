"""Monte Carlo BER/BLER simulation of Reed-Muller codes over BPSK/AWGN.

For each code and Eb/N0 point, random messages are encoded, BPSK-modulated
(0 -> +1, 1 -> -1) and sent through AWGN with noise variance
sigma^2 = 1 / (2 * R * Eb/N0). Each point runs until `--target-errors` block
errors are seen or `--max-blocks` blocks have been sent.

Decoders:
  majority  hard decisions (sign of y) + Reed majority-logic decoding
  ml        soft-decision maximum likelihood (FHT for RM(1,m), brute force
            for small RM(2,m))

Usage:
  python scripts/simulate.py                 # full run, writes results/results.csv
  python scripts/simulate.py --quick         # fast smoke run
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from rmcodes import BruteForceMLDecoder, MajorityLogicDecoder, ReedMullerCode, ml_decode_rm1

CODES = [(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6)]


def make_decoders(code):
    maj = MajorityLogicDecoder(code)
    decoders = {"majority": lambda y, rng: maj.decode((y < 0).astype(np.uint8), rng)}
    if code.r == 1:
        decoders["ml"] = lambda y, rng: ml_decode_rm1(code, y)
    elif code.k <= BruteForceMLDecoder.MAX_K:
        bf = BruteForceMLDecoder(code)
        decoders["ml"] = lambda y, rng: bf.decode(y)
    return decoders


def simulate_point(code, decode, ebn0_db, rng, target_errors, max_blocks, batch):
    sigma = np.sqrt(1.0 / (2 * code.rate * 10 ** (ebn0_db / 10)))
    blocks = bit_err = block_err = 0
    while block_err < target_errors and blocks < max_blocks:
        b = min(batch, max_blocks - blocks)
        u = rng.integers(0, 2, (b, code.k), dtype=np.uint8)
        y = 1.0 - 2.0 * code.encode(u) + sigma * rng.standard_normal((b, code.n))
        errs = decode(y, rng) != u
        bit_err += int(errs.sum())
        block_err += int(errs.any(axis=1).sum())
        blocks += b
    return blocks, bit_err, block_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ebn0", type=float, nargs=3, default=[0.0, 10.0, 0.5],
                   metavar=("START", "STOP", "STEP"))
    p.add_argument("--target-errors", type=int, default=300)
    p.add_argument("--max-blocks", type=int, default=2_000_000)
    p.add_argument("--max-blocks-bruteforce", type=int, default=200_000)
    p.add_argument("--min-ber", type=float, default=1e-6,
                   help="stop a curve once BER falls below this")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.target_errors, args.max_blocks, args.max_blocks_bruteforce = 50, 20_000, 2_000

    start, stop, step = args.ebn0
    grid = np.round(np.arange(start, stop + step / 2, step), 2)
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "r", "m", "n", "k", "decoder", "ebn0_db",
                    "blocks", "bit_errors", "block_errors", "ber", "bler"])
        for r, m in CODES:
            code = ReedMullerCode(r, m)
            for name, decode in make_decoders(code).items():
                max_blocks = (args.max_blocks_bruteforce
                              if name == "ml" and code.r > 1 else args.max_blocks)
                batch = 512 if name == "ml" and code.r > 1 else 20_000
                t0 = time.time()
                for ebn0 in grid:
                    blocks, be, ble = simulate_point(code, decode, ebn0, rng,
                                                     args.target_errors, max_blocks, batch)
                    ber, bler = be / (blocks * code.k), ble / blocks
                    w.writerow([f"RM({r},{m})", r, m, code.n, code.k, name, ebn0,
                                blocks, be, ble, ber, bler])
                    f.flush()
                    if ber < args.min_ber or be == 0:
                        break
                print(f"{code!r:28s} {name:9s} done in {time.time() - t0:6.1f}s, "
                      f"last point {ebn0} dB BER={ber:.2e}")


if __name__ == "__main__":
    main()
