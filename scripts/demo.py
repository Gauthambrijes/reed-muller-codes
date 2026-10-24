"""Live demo: send one packet through a noisy channel and decode it.

Usage:  PYTHONPATH=. python scripts/demo.py [--r 2 --m 7 --ebn0 2.5 --seed 7]

Shows the message, how many bits the channel flipped, and what each decoder recovers:
hard-decision Reed majority logic, and soft RPA in floating point and with min-sum.
"""

import argparse

import numpy as np

from rmcodes import MajorityLogicDecoder, ReedMullerCode, RPADecoder


def bits(v, width=64):
    s = "".join(str(int(x)) for x in v)
    return "\n      ".join(s[i:i + width] for i in range(0, len(s), width))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=int, default=2)
    p.add_argument("--m", type=int, default=7)
    p.add_argument("--ebn0", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    code = ReedMullerCode(a.r, a.m)
    rng = np.random.default_rng(a.seed)
    sigma = np.sqrt(1 / (2 * code.rate * 10 ** (a.ebn0 / 10)))

    print(f"\n{code}   rate {code.rate:.3f}   Eb/N0 = {a.ebn0} dB   noise sigma = {sigma:.2f}")
    print("-" * 72)
    u = rng.integers(0, 2, code.k, dtype=np.uint8)
    c = code.encode(u)
    y = 1 - 2.0 * c + sigma * rng.standard_normal(code.n)
    hard = (y < 0).astype(np.uint8)
    flips = int((hard != c).sum())
    print(f"message ({code.k} bits):\n      {bits(u)}")
    print(f"codeword ({code.n} bits) sent as BPSK (+1 / -1)")
    print(f"channel flipped {flips} of {code.n} bits "
          f"(Reed's decoder is only guaranteed for up to {code.t})")
    print("-" * 72)

    llr = 2 * y / sigma ** 2
    results = [
        ("Reed majority logic (hard)", MajorityLogicDecoder(code).decode(hard[None])[0]),
        ("RPA, exact box-plus (soft)", RPADecoder(code).decode(llr[None])[0]),
        ("RPA, min-sum (soft)", RPADecoder(code, min_sum=True).decode(llr[None])[0]),
    ]
    for name, uh in results:
        wrong = int((uh != u).sum())
        verdict = "CORRECT" if wrong == 0 else f"{wrong} message bits wrong"
        print(f"{name:30s} -> {verdict}")
    print()


if __name__ == "__main__":
    main()
