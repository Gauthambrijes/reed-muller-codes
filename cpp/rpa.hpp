// Recursive projection-aggregation (RPA) decoding of RM(r, m).
//
// RPAFloat : double precision, exact box-plus or min-sum projection
//            (Ye & Abbe, Algorithms 3-4).
// RPAFixed : integer min-sum RPA for hardware studies. Every stored LLR is a
//            b-bit signed integer in [-A, A], A = 2^(b-1) - 1. Aggregation sums
//            are renormalised by an arithmetic shift of m with symmetric
//            rounding, so stored values never leave [-A, A] (docs/theory.md,
//            Theorem 3). The decoder records the largest magnitude seen in every
//            FHT and aggregation accumulator, per recursion level, so the
//            proven word lengths can be checked empirically.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

using Rng = std::mt19937_64;

// ---------------------------------------------------------------- tables

struct ProjTables {
    int m = 0, n = 0, h = 0;
    std::vector<int> reps;   // (n-1) x h : representative (bit j = 0) of coset w, for direction b
    std::vector<int> coord;  // (n-1) x n : coset coordinate of point z, for direction b
};

inline const ProjTables& proj_tables(int m) {
    static std::array<std::unique_ptr<ProjTables>, 16> cache{};
    if (!cache[m]) {
        auto t = std::make_unique<ProjTables>();
        t->m = m; t->n = 1 << m; t->h = t->n / 2;
        t->reps.resize((size_t)(t->n - 1) * t->h);
        t->coord.resize((size_t)(t->n - 1) * t->n);
        for (int b = 1; b < t->n; ++b) {
            int j = __builtin_ctz(b);
            int* rp = &t->reps[(size_t)(b - 1) * t->h];
            int* cp = &t->coord[(size_t)(b - 1) * t->n];
            int w = 0;
            for (int z = 0; z < t->n; ++z) {
                if (((z >> j) & 1) == 0) rp[w++] = z;  // increasing z == increasing coordinate
                int rep = ((z >> j) & 1) ? (z ^ b) : z;
                cp[z] = (rep & ((1 << j) - 1)) | ((rep >> (j + 1)) << j);
            }
        }
        cache[m] = std::move(t);
    }
    return *cache[m];
}

// Build every table a decoder for RM(r, m) will touch (call before threads start).
inline void warm_tables(int r, int m) {
    for (int mm = m; mm >= m - r + 1 && mm >= 1; --mm) proj_tables(mm);
}

inline int parity(unsigned x) { return __builtin_parity(x); }

// ---------------------------------------------------------------- float RPA

enum class Combine { Exact, MinSum };

inline double sgn0(double x) { return (x > 0) - (x < 0); }

// g(x) = ln(1 + e^{-x}) for x >= 0, tabulated with linear interpolation.
// Step 2^-10 gives |error| <= step^2 / 32 < 3e-8, far below any LLR that matters.
struct SoftplusLUT {
    static constexpr double XMAX = 40.0, STEP = 1.0 / 1024;
    std::vector<double> t;
    SoftplusLUT() {
        int N = (int)(XMAX / STEP) + 2;
        t.resize(N);
        for (int i = 0; i < N; ++i) t[i] = std::log1p(std::exp(-i * STEP));
    }
    double operator()(double x) const {
        if (x >= XMAX) return 0.0;  // ln(1 + e^-40) < 5e-18
        double p = x / STEP;
        int i = (int)p;
        return t[i] + (t[i + 1] - t[i]) * (p - i);
    }
};
inline const SoftplusLUT& softplus_lut() { static const SoftplusLUT L; return L; }

inline double combine(Combine c, double a, double b) {
    double ms = sgn0(a) * sgn0(b) * std::min(std::fabs(a), std::fabs(b));
    if (c == Combine::MinSum) return ms;
    // exact box-plus = min-sum + ln(1 + e^-|a+b|) - ln(1 + e^-|a-b|)
    const SoftplusLUT& g = softplus_lut();
    return ms + g(std::fabs(a + b)) - g(std::fabs(a - b));
}

struct RPAFloat {
    Combine comb = Combine::Exact;
    double theta = 0.05;
    int iters = -1;  // rounds per level; -1 means floor(m/2), at least 1

    // L: LLRs of a word of RM(r, m) (length 2^m); out: +1 / -1 decisions.
    void decode(const double* Lin, int r, int m, int8_t* out) const {
        const int n = 1 << m;
        if (r == 1) { fht_decode(Lin, m, out); return; }
        if (r >= m) { for (int z = 0; z < n; ++z) out[z] = Lin[z] >= 0 ? 1 : -1; return; }
        const ProjTables& T = proj_tables(m);
        const int h = n / 2;
        std::vector<double> L(Lin, Lin + n), cumu(n), proj(h);
        std::vector<int8_t> yh(h);
        int rounds = iters > 0 ? iters : std::max(1, m / 2);
        for (int it = 0; it < rounds; ++it) {
            std::fill(cumu.begin(), cumu.end(), 0.0);
            for (int b = 1; b < n; ++b) {
                const int* rp = &T.reps[(size_t)(b - 1) * h];
                for (int w = 0; w < h; ++w) proj[w] = combine(comb, L[rp[w]], L[rp[w] ^ b]);
                decode(proj.data(), r - 1, m - 1, yh.data());
                const int* cp = &T.coord[(size_t)(b - 1) * n];
                for (int z = 0; z < n; ++z) cumu[z] += yh[cp[z]] * L[z ^ b];
            }
            bool stable = true;
            for (int z = 0; z < n; ++z) {
                double v = cumu[z] / (n - 1);
                if (std::fabs(v - L[z]) > theta * std::fabs(L[z])) stable = false;
                L[z] = v;
            }
            if (stable) break;
        }
        for (int z = 0; z < n; ++z) out[z] = L[z] >= 0 ? 1 : -1;
    }

    static void fht_decode(const double* Lin, int m, int8_t* out) {
        const int n = 1 << m;
        std::vector<double> s(Lin, Lin + n);
        for (int h = 1; h < n; h <<= 1)
            for (int i = 0; i < n; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    double x = s[j], y = s[j + h];
                    s[j] = x + y; s[j + h] = x - y;
                }
        int best = 0;
        for (int u = 1; u < n; ++u) if (std::fabs(s[u]) > std::fabs(s[best])) best = u;
        int sg = s[best] >= 0 ? 1 : -1;
        for (int z = 0; z < n; ++z) out[z] = (int8_t)(parity(best & z) ? -sg : sg);
    }
};

// ---------------------------------------------------------------- fixed RPA

struct WidthStats {
    // Largest |value| seen, indexed by the m of the code being decoded at that
    // level: FHT accumulators (base level, all stages) and aggregation sums.
    std::array<int64_t, 16> max_fht{}, max_cumu{}, max_store{};
    void merge(const WidthStats& o) {
        for (int i = 0; i < 16; ++i) {
            max_fht[i] = std::max(max_fht[i], o.max_fht[i]);
            max_cumu[i] = std::max(max_cumu[i], o.max_cumu[i]);
            max_store[i] = std::max(max_store[i], o.max_store[i]);
        }
    }
};

struct RPAFixed {
    int A = 31;               // stored LLRs lie in [-A, A]
    int theta_num = 1, theta_den = 20;  // early exit: |L' - L| <= theta |L|
    int iters = -1;
    bool random_ties = true;  // break FHT argmax ties and zero signs uniformly at random

    static int64_t sgn(int64_t x) { return (x > 0) - (x < 0); }

    // Symmetric rounding of c / 2^s (round half away from zero): an odd function.
    static int64_t shift_round(int64_t c, int s) {
        int64_t a = c < 0 ? -c : c;
        int64_t q = (a + (int64_t(1) << (s - 1))) >> s;
        return c < 0 ? -q : q;
    }

    int8_t sign_of(int64_t v, Rng& rng) const {
        if (v > 0) return 1;
        if (v < 0) return -1;
        return (random_ties && (rng() & 1)) ? -1 : 1;
    }

    void decode(const int32_t* Lin, int r, int m, int8_t* out, Rng& rng, WidthStats& st) const {
        const int n = 1 << m;
        for (int z = 0; z < n; ++z) st.max_store[m] = std::max<int64_t>(st.max_store[m], std::abs(Lin[z]));
        if (r == 1) { fht_decode(Lin, m, out, rng, st); return; }
        if (r >= m) { for (int z = 0; z < n; ++z) out[z] = sign_of(Lin[z], rng); return; }
        const ProjTables& T = proj_tables(m);
        const int h = n / 2;
        std::vector<int32_t> L(Lin, Lin + n), proj(h);
        std::vector<int64_t> cumu(n);
        std::vector<int8_t> yh(h);
        int rounds = iters > 0 ? iters : std::max(1, m / 2);
        for (int it = 0; it < rounds; ++it) {
            std::fill(cumu.begin(), cumu.end(), 0);
            for (int b = 1; b < n; ++b) {
                const int* rp = &T.reps[(size_t)(b - 1) * h];
                for (int w = 0; w < h; ++w) {
                    int32_t a = L[rp[w]], c = L[rp[w] ^ b];
                    proj[w] = (int32_t)(sgn(a) * sgn(c) * std::min(std::abs(a), std::abs(c)));
                }
                decode(proj.data(), r - 1, m - 1, yh.data(), rng, st);
                const int* cp = &T.coord[(size_t)(b - 1) * n];
                for (int z = 0; z < n; ++z) cumu[z] += yh[cp[z]] * (int64_t)L[z ^ b];
            }
            bool stable = true;
            for (int z = 0; z < n; ++z) {
                st.max_cumu[m] = std::max<int64_t>(st.max_cumu[m], std::abs(cumu[z]));
                int32_t v = (int32_t)shift_round(cumu[z], m);
                if ((int64_t)std::abs(v - L[z]) * theta_den > (int64_t)std::abs(L[z]) * theta_num) stable = false;
                L[z] = v;
                st.max_store[m] = std::max<int64_t>(st.max_store[m], std::abs(v));
            }
            if (stable) break;
        }
        for (int z = 0; z < n; ++z) out[z] = sign_of(L[z], rng);
    }

    void fht_decode(const int32_t* Lin, int m, int8_t* out, Rng& rng, WidthStats& st) const {
        const int n = 1 << m;
        std::vector<int64_t> s(Lin, Lin + n);
        int64_t mx = 0;
        for (int h = 1; h < n; h <<= 1)
            for (int i = 0; i < n; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    int64_t x = s[j], y = s[j + h];
                    s[j] = x + y; s[j + h] = x - y;
                    mx = std::max({mx, std::abs(s[j]), std::abs(s[j + h])});
                }
        st.max_fht[m] = std::max(st.max_fht[m], mx);
        int64_t best_abs = -1; int best = 0, ties = 0;
        for (int u = 0; u < n; ++u) {
            int64_t a = std::abs(s[u]);
            if (a > best_abs) { best_abs = a; best = u; ties = 1; }
            else if (a == best_abs && random_ties) {
                ++ties;
                if (rng() % ties == 0) best = u;  // reservoir sampling: uniform among maxima
            }
        }
        int8_t sg = sign_of(s[best], rng);
        for (int z = 0; z < n; ++z) out[z] = (int8_t)(parity(best & z) ? -sg : sg);
    }
};

// Channel quantiser: round(y / step) clipped to [-A, A], step = clip / A.
// Symmetric rounding and saturation make it an odd function of y.
inline int32_t quantize(double y, double step, int A) {
    long q = std::lround(y / step);
    if (q > A) q = A;
    if (q < -A) q = -A;
    return (int32_t)q;
}
