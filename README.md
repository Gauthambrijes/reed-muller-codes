# Reed-Muller Codes: Encoding, Decoding and BER Simulation

A NumPy implementation of binary Reed-Muller codes RM(r, m), with Reed's majority-logic decoder, soft-decision maximum-likelihood decoders, and Monte Carlo BER / block-error-rate simulations over the BPSK/AWGN channel.

## Results

All curves are BPSK over AWGN, with BER measured on the information bits. Each point runs until 300 block errors are seen or the block budget is used up (1,000,000 blocks for majority-logic and FHT decoding, 200,000 for brute-force ML).

![Soft ML vs hard majority-logic decoding](results/ber_soft_vs_hard.png)

Required Eb/N0 for BER = 10⁻⁴ (uncoded BPSK needs 8.40 dB):

| Code | [n, k, d] | Rate | Majority-logic (hard) | Soft ML | Soft-decision advantage |
| --- | --- | --- | --- | --- | --- |
| RM(1,4) | [16, 5, 8] | 0.31 | 8.16 dB (gain 0.24 dB) | 5.74 dB (gain 2.66 dB) | 2.42 dB |
| RM(1,5) | [32, 6, 16] | 0.19 | 8.04 dB (gain 0.36 dB) | 5.23 dB (gain 3.16 dB) | 2.81 dB |
| RM(1,6) | [64, 7, 32] | 0.11 | 8.28 dB (gain 0.12 dB) | 4.82 dB (gain 3.58 dB) | 3.46 dB |
| RM(2,4) | [16, 11, 4] | 0.69 | 8.11 dB (gain 0.29 dB) | 6.04 dB (gain 2.36 dB) | 2.07 dB |
| RM(2,5) | [32, 16, 8] | 0.50 | 7.27 dB (gain 1.13 dB) | 4.82 dB (gain 3.57 dB) | 2.45 dB |
| RM(2,6) | [64, 22, 16] | 0.34 | 7.09 dB (gain 1.30 dB) | — | — |

Values are interpolated between simulated points. Soft ML for RM(2,6) is not shown because brute-force decoding over 2²² codewords is impractical.

**Observations**

- **Soft decisions matter most for low-rate codes.** First-order codes gain almost nothing with hard majority-logic decoding at BER 10⁻⁴ (their low rate costs nearly as much as the error correction recovers), but 2.7–3.6 dB with soft ML.
- **Under ML decoding, longer first-order codes keep improving** (RM(1,4) → RM(1,6) gains about 0.9 dB). Under hard majority-logic decoding they don't, because the rate loss grows as fast as the correction capability.
- **Second-order codes are the better trade-off for hard decoding.** RM(2,5) and RM(2,6) reach about 1.1–1.3 dB of gain with majority logic, thanks to their higher rate.
- **RM(2,5) with soft ML matches RM(1,6)** at BER 10⁻⁴ (4.82 dB) while carrying more than twice as many information bits per block (16 vs 7).

![Majority-logic BER for all codes](results/ber_majority.png)

![Block error rate](results/bler_all.png)

The raw numbers are in [`results/results.csv`](results/results.csv).

## Background

RM(r, m) is the set of evaluation vectors of all Boolean polynomials in m variables of degree at most r, evaluated at the 2^m points of GF(2)^m.

- Length n = 2^m, dimension k = Σ_{i=0}^{r} C(m, i), minimum distance d = 2^(m−r)
- RM(1, m) is the first-order (biorthogonal) code; RM(m−2, m) is the extended Hamming code
- The generator matrix has one row per monomial of degree ≤ r (`rmcodes/code.py`)

## Decoders

**Reed majority-logic decoding (hard decision)** — `MajorityLogicDecoder`

Decodes coefficients from the highest degree down. For a degree-d monomial, summing the received word over any coset of the subspace spanned by that monomial's variables gives the monomial's coefficient: every other term of degree ≤ d sums to zero over the coset, and higher-degree terms have already been subtracted. The 2^(m−d) cosets give 2^(m−d) independent votes, and the coefficient is their majority. This corrects every error pattern of weight up to 2^(m−r−1) − 1, which the tests verify.

**Soft ML for first-order codes** — `ml_decode_rm1`

The BPSK images of RM(1, m) codewords are ±(−1)^(a·x), so correlating the received vector with all 2^(m+1) codewords is just a Walsh–Hadamard transform. The ML codeword maximises |FHT(y)[a]|, in O(n log n).

**Soft ML by exhaustive search** — `BruteForceMLDecoder`

Correlates with all 2^k codewords and picks the best. Used for RM(2,4) and RM(2,5) (k ≤ 16).

## Repository layout

```
rmcodes/
├── code.py        # ReedMullerCode: parameters, generator matrix, encoder
└── decoders.py    # majority-logic, FHT-based ML, brute-force ML
scripts/
├── simulate.py    # Monte Carlo BER/BLER simulation -> results/results.csv
└── plot.py        # results/*.png
tests/
└── test_rm.py     # parameters, minimum distance, error-correction guarantee, ML checks
results/           # CSV and plots from the run above
```

## Running it

```bash
pip install -r requirements.txt

python -m pytest tests                       # 27 tests, under a second
PYTHONPATH=. python scripts/simulate.py      # full simulation, a few minutes
PYTHONPATH=. python scripts/simulate.py --quick
PYTHONPATH=. python scripts/plot.py
```

`simulate.py` options include `--ebn0 START STOP STEP`, `--target-errors`, `--max-blocks` and `--seed`. Results are reproducible with the default seed.

## Possible extensions

- Recursive (Dumer / Plotkin-structure) soft decoding, to get near-ML soft decoding for RM(2,6) and longer codes
- Successive-cancellation and list decoding, and the connection to polar codes
- Comparison with BCH and polar codes of similar length and rate
