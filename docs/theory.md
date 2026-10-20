# Theory: min-sum RPA in fixed point, and finite-blocklength benchmarks

This note states and proves the results the C++ engine (`cpp/`) and the
benchmarks rely on. Code conventions follow the repository: RM(r, m) has
length n = 2^m, coordinates are points z ∈ E = GF(2)^m, and the Ye–Abbe paper
writes the same code as RM(m, r). BPSK maps bit 0 to +1; y = x + σZ; the
channel LLR is ℓ(z) = 2y(z)/σ².

## 1. The decoder

For r ≥ 2, one RPA round on LLRs L : E → ℝ is

1. **Projection.** For every nonzero b ∈ E and every coset T = {z, z+b},
   P_b(T) = L(z) ⊞ L(z+b) (exact) or L(z) ⊟ L(z+b) := sgn L(z) · sgn L(z+b) · min(|L(z)|, |L(z+b)|) (min-sum).
2. **Recursion.** Decode each P_b as a word of RM(r−1, m−1); write ŷ_b(T) ∈ {±1} for the decision (+1 ↔ bit 0).
   RM(1, ·) is decoded by the FHT: û = argmax_u |P̂(u)|, P̂(u) = Σ_w (−1)^{u·w} P(w), sign = sgn P̂(û).
3. **Aggregation.** L′(z) = κ · Σ_{b≠0} ŷ_b([z+b]) · L(z+b), with κ = 1/(n−1) in floating point.

Rounds repeat until max_z |L′(z) − L(z)| / |L(z)| ≤ θ = 0.05 or ⌊m/2⌋ rounds; the output is sgn L.

**Lemma 1 (projection).** If c = Eval(f) ∈ RM(r, m) and b ≠ 0, then the word
T ↦ c(z) + c(z+b) on the cosets of {0, b} is a codeword of RM(r−1, m−1).

*Proof.* c(z) + c(z+b) = (D_b f)(z) with D_b f(z) = f(z+b) + f(z). For a monomial
x_A, D_b x_A = Π_{i∈A}(z_i + b_i) − Π_{i∈A} z_i has degree ≤ |A| − 1, so
deg D_b f ≤ r − 1. D_b f is constant on cosets of {0, b}, so it is a function on
the quotient E/⟨b⟩ ≅ GF(2)^{m−1}. Picking the representative with z_j = 0
(j = lowest set bit of b) and deleting coordinate j is a linear isomorphism of
the quotient, so the projected word is the evaluation of a polynomial of
degree ≤ r − 1 in m − 1 variables. ∎

## 2. Min-sum versus box-plus

**Lemma 2.** For all a, b ∈ ℝ,
a ⊞ b = a ⊟ b + ln(1 + e^{−|a+b|}) − ln(1 + e^{−|a−b|}).
The correction lies in [−ln 2, ln 2] and has the sign opposite to a ⊟ b, so
|a ⊟ b − a ⊞ b| ≤ ln 2 and |a ⊞ b| ≤ |a ⊟ b|, with sgn(a ⊞ b) = sgn(a ⊟ b).

*Proof.* a ⊞ b = ln(1 + e^{a+b}) − ln(e^a + e^b). Write
ln(1 + e^{a+b}) = max(0, a+b) + ln(1 + e^{−|a+b|}) and
ln(e^a + e^b) = max(a, b) + ln(1 + e^{−|a−b|}). The difference of the maxima
equals a ⊟ b: for a, b ≥ 0 it is a + b − max(a, b) = min(a, b); for a, b ≤ 0 it is
−max(a, b) = min(|a|, |b|); for a ≥ 0 ≥ b it is b = −min(|a|, |b|) if a + b ≥ 0 and
−a = −min(|a|, |b|) otherwise (the case b ≥ 0 ≥ a is symmetric). Both log terms
lie in (0, ln 2], so the correction is in [−ln 2, ln 2]. If sgn a = sgn b then
|a+b| ≥ |a−b|, so the correction is ≤ 0 while a ⊟ b ≥ 0; otherwise the correction
is ≥ 0 while a ⊟ b ≤ 0. Finally |correction| ≤ |a ⊟ b|: by the symmetries
(a, b) ↦ (b, a) and (a, b) ↦ (−a, b) it suffices to take a ≥ b ≥ 0, where the
claim ln(1 + e^{−(a−b)}) − ln(1 + e^{−(a+b)}) ≤ b is equivalent to
1 + e^{b−a} ≤ e^{b} + e^{−a}, i.e. (e^{b} − 1)(1 − e^{−a}) ≥ 0. ∎

(Checked numerically in `tests/test_rpa_bounds.py`.)

## 3. Scale invariance of min-sum RPA

**Theorem 1.** Let D be the min-sum RPA decoder for RM(r, m), r ≥ 1, in exact
arithmetic, with any normalisation constant κ > 0 in the aggregation step
(including a different κ at every level). Then D(cL) = D(L) for every c > 0.

*Proof.* Induction on r. For r = 1, the FHT is linear, so the spectrum of cL is
c P̂; the argmax of |·| and the sign of the maximiser are unchanged. For r ≥ 2,
suppose the claim holds for r − 1. The min-sum projection is positively
homogeneous, (ca) ⊟ (cb) = c (a ⊟ b), so the projections of cL are c P_b, whose
decisions equal those of P_b by induction. The aggregation is linear in L for
fixed decisions, so the next iterate is c L′. The stopping rule compares
|L′ − L| with θ|L| and is invariant under scaling. By induction over rounds, all
iterates scale by c and the final signs agree. ∎

**Corollaries.**
1. *No SNR estimate is needed.* The channel LLR 2y/σ² can be replaced by y (or
   any positive multiple), so a min-sum RPA receiver needs no noise-variance
   estimate. The fixed-point decoder therefore quantises y directly.
2. *Normalised min-sum is identical to min-sum.* Scaling every projected LLR by
   α > 0 changes nothing (apply the theorem to each recursive call), so the usual
   LDPC trick of normalised min-sum cannot help RPA; only non-homogeneous
   corrections (offsets, Lemma 2's exact term) change decisions.
3. *The aggregation constant is free.* Dividing by 2^m (a shift) instead of
   n − 1 gives the same decisions in exact arithmetic; it matters only through
   rounding in fixed point.

Exact box-plus is not homogeneous, so these statements do not hold for it.

## 4. Codeword symmetry

**Theorem 2.** Let c ∈ RM(r, m) and s(z) = (−1)^{c(z)}. For the floating-point
decoders (exact or min-sum), D(s ⊙ L) = s ⊙ D(L) whenever the FHT argmax is
unique. For the fixed-point decoder with an odd channel quantiser, symmetric
rounding, and uniformly random tie-breaking (FHT argmax and zero signs), the
law of D(s ⊙ L) equals the law of s ⊙ D(L). Consequently the block error
probability over any binary-input symmetric channel is the same for every
transmitted codeword.

*Proof sketch.* Induction on r, as in Proposition 4 of Ye–Abbe. Flipping L by s
flips each projection P_b by the projected codeword (Lemma 1), because both ⊞
and ⊟ satisfy (s₁a) ⋆ (s₂b) = s₁s₂(a ⋆ b). For RM(1, m) with
s = (−1)^{a₀ + a·z}, the spectrum transforms as (s ⊙ P)^(u) = (−1)^{a₀} P̂(u ⊕ a):
the set of maximisers is translated by a, so a uniform choice among them
commutes with the flip, and the decision is flipped by s. In aggregation each
vote ŷ_b([z+b]) L(z+b) is multiplied by s(z)s(z+b) · s(z+b) = s(z). The shift
renormalisation x ↦ sgn(x) ⌊(|x| + 2^{m−1}) / 2^m⌋ and the quantiser are odd,
the stopping rule depends only on magnitudes, and a uniformly random sign for a
zero LLR is flip-invariant in law. ∎

Without random tie-breaking (e.g. "first maximiser wins"), the fixed-point
decoder is *not* codeword-symmetric, because integer spectra tie with positive
probability; simulations that send only the all-zero codeword would then be
biased. The engine therefore breaks ties at random and also sends random
codewords.

## 5. Word lengths for fixed-point min-sum RPA

Fixed-point decoder: every stored LLR is an integer in [−A, A] with
A = 2^{b−1} − 1 (b-bit symmetric storage). Channel values are
q(y) = clip(round(y/Δ), −A, A), round-half-away-from-zero. Projections use
min-sum. At a level that decodes RM(r′, m′) with r′ ≥ 2, the aggregation sum
C(z) = Σ_{b≠0} ŷ_b L(z+b) is formed exactly and stored as
L′(z) = sgn C(z) · ⌊(|C(z)| + 2^{m′−1}) / 2^{m′}⌋.

**Theorem 3.** For the fixed-point min-sum RPA decoder of RM(r, m), 2 ≤ r < m:

1. every stored LLR, at every level and round, lies in [−A, A];
2. after t butterfly stages of the FHT that decodes RM(1, m−r+1), every value
   satisfies |·| ≤ 2^t A; in particular all FHT values are ≤ 2^{m−r+1} A;
3. at a level decoding RM(r′, m′), every partial and complete aggregation sum
   satisfies |·| ≤ (2^{m′} − 1) A, the largest being (2^m − 1) A at the top level;
4. all three bounds are attained, by the received word L ≡ A.

Hence b bits of storage, b + m − r + 1 bits for the FHT and b + m bits for the
aggregation accumulator suffice for overflow-free decoding of every input, and
for b ≥ 3 none of these widths can be reduced.

*Proof.* (1) The quantiser saturates at ±A. Min-sum gives
|a ⊟ b| ≤ min(|a|, |b|) ≤ A, so projected LLRs, which are the inputs of the
next level, are also in [−A, A]. For the renormalised aggregate, |C| ≤ (N − 1)A
with N = 2^{m′} by (3), so
|L′| ≤ ⌊((N−1)A + N/2)/N⌋ = ⌊A + 1/2 − A/N⌋.
If A/N < 1/2 then A < A + 1/2 − A/N < A + 1/2 and the floor is A; otherwise
the argument is ≤ A. Either way |L′| ≤ A. By induction over levels and rounds,
(1) holds.
(2) A butterfly maps (x, y) to (x + y, x − y), so the maximum magnitude at most
doubles per stage; the inputs are bounded by A from (1).
(3) There are N − 1 directions; each term ŷ_b L(z+b) has magnitude ≤ A, so every
partial sum of j terms is ≤ jA ≤ (N−1)A.
(4) If L ≡ A, every projection is A ⊟ A = A, every level receives the constant
word A, the FHT returns 2^{m′} A at u = 0 with decision +1 everywhere, and
the aggregation sums to (N − 1)A for every z. These values propagate unchanged
through the rounds, so every bound is met with equality.
Width: a signed word of w bits holds [−(2^{w−1}−1), 2^{w−1}−1]. With
x = 2^{m′}, y = 2^{b−1}: the FHT bound xA = x(y−1) satisfies
xy/2 − 1 < x(y−1) ≤ xy − 1 for y ≥ 2, so it needs exactly b + m′ bits. The
aggregation bound (x−1)(y−1) ≤ xy − 1 always, and (x−1)(y−1) > xy/2 − 1 iff
(x−2)(y−2) > 0, i.e. for m′ ≥ 2 and b ≥ 3; then it needs exactly b + m′ bits. ∎

**Example.** RM(2, 8) with b = 6: storage 6 bits, FHT 13 bits, aggregation
14 bits, so the whole decoder runs in 16-bit integer arithmetic with no
saturation logic after the channel quantiser. With b = 16 (Q7.8 storage),
the FHT needs 23 bits and the aggregation 24 bits, so 16-bit accumulators
overflow. `cpp/sim width` measures the largest values reached and
`tests/test_cpp.py` checks that they never exceed, and do attain, the bounds.

## 6. Finite-blocklength benchmarks (BI-AWGN)

Uniform inputs, information density per channel use (nats)
i(+1; y) = ln 2 − ln(1 + e^{−ℓ}), and its generalisation
i_s(+1; y) = ln 2 − ln(1 + e^{−sℓ}). By symmetry the law of the block sum
S = Σ_j i(x_j; Y_j) does not depend on the codeword; M = 2^k.

* **Normal approximation** (Polyanskiy–Poor–Verdú 2010):
  ln M ≈ nC − √(nV) Q⁻¹(ε) + ½ ln n, with C = E[i], V = Var[i].
* **Meta-converse** (PPV Thm. 27, auxiliary output law Q = uniform-input output):
  every code with average error ε satisfies β_{1−ε}(P_S, Q_S) ≤ 1/M, where
  dQ_S/dP_S = e^{−s}. This is a lower bound on ε for *any* code.
* **RCU_s achievability** (Martínez & Guillén i Fàbregas 2011): for i.i.d. uniform
  random codes and every s > 0,
  ε ≤ E[exp(−(Σ_j i_s(X_j; Y_j) − ln(M−1))⁺)],
  by Markov's inequality applied to the pairwise error event
  {W(Y|X̄) ≥ W(Y|X)}. We minimise over s.
* **DT bound** (PPV Thm. 17): ε ≤ E[exp(−(S − ln((M−1)/2))⁺)].

**Numerics (rigorous).** The map y ↦ i_s(+1; y) is increasing, so its law is
computed exactly on lattice bins [t_j, t_j + h) from the Gaussian CDF at the
preimages. Floor rounding gives S⁻ ≤ S and ceiling rounding S⁺ ≥ S path by
path; n-fold convolutions are done by FFT. Achievability bounds use S⁻ (their
integrands are non-increasing in S) plus n·P(i < t_min) for the truncated tail.
For the converse, the Neyman–Pearson lower bound
β_α ≥ Σ_{s > τ⁺ + nh} e^{−s} P(S⁺ = s) − n·P(i < t_min)·e^{−τ⁺−nh},
with τ⁺ the smallest lattice point satisfying P(S⁺ > τ⁺) ≤ α, holds because
{S > τ} ⊇ {S⁺ − nh > τ⁺} off the truncation event, and e^{−S} ≥ e^{−S⁺}. Both
bounds are therefore rigorous up to floating-point rounding, at a cost of at
most nh nats (0.13 nats for n = 256, h = 5·10⁻⁴; RCU_s uses h = 10⁻³).
`tests/test_fbl.py` checks C = 0.5 bit at the rate-½ Shannon limit
(Eb/N0 = 0.187 dB), the DT and RCU_s values against Monte Carlo, and the
ordering converse ≤ normal approximation ≤ RCU_s ≤ DT.
