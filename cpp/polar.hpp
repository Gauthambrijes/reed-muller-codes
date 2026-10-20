// CRC-aided polar codes with successive-cancellation list (SCL) decoding.
//
// Encoder x = u F^{(x)m}, F = [[1,0],[1,1]] (natural order, no bit reversal),
// so x = (enc(u1) + enc(u2), enc(u2)) for the two halves u1, u2 of u.
// Construction: Gaussian approximation of density evolution (Chung's phi) at a
// design Eb/N0; the K = k + c most reliable positions carry k payload bits
// followed by c CRC bits. The decoder is the Tal-Vardy SCL decoder written
// recursively over the code tree, with exact LLR-based path metrics
// PM += ln(1 + exp(-(1 - 2u) L)) and exact box-plus f.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------- CRC

struct CRC {
    int c = 0;
    uint32_t poly = 0;  // generator without the leading x^c term
    explicit CRC(int len = 0) : c(len) {
        switch (len) {
            case 0: break;
            case 4: poly = 0x3; break;        // x^4 + x + 1
            case 6: poly = 0x21; break;       // x^6 + x^5 + 1 (5G NR CRC6)
            case 8: poly = 0xD5; break;       // x^8 + x^7 + x^6 + x^4 + x^2 + 1
            case 11: poly = 0x621; break;     // x^11 + x^10 + x^9 + x^5 + 1 (5G NR CRC11)
            case 16: poly = 0x1021; break;    // CRC-16-CCITT
            default: throw std::invalid_argument("unsupported CRC length");
        }
    }
    // Remainder of bits(x) * x^c modulo g(x); bits[0] is the highest-degree coefficient.
    uint32_t remainder(const uint8_t* bits, int len) const {
        if (c == 0) return 0;
        uint32_t reg = 0, top = 1u << (c - 1), mask = (1u << c) - 1;
        for (int i = 0; i < len; ++i) {
            bool fb = ((reg & top) != 0) ^ (bits[i] != 0);
            reg = (reg << 1) & mask;
            if (fb) reg ^= poly;
        }
        return reg;
    }
    void append(const uint8_t* payload, int k, uint8_t* out_crc) const {
        uint32_t rem = remainder(payload, k);
        for (int i = 0; i < c; ++i) out_crc[i] = (rem >> (c - 1 - i)) & 1;
    }
    bool check(const uint8_t* payload, int k, const uint8_t* crc) const {
        uint32_t rem = remainder(payload, k), got = 0;
        for (int i = 0; i < c; ++i) got = (got << 1) | crc[i];
        return rem == got;
    }
};

// ---------------------------------------------------------------- construction

inline double ga_logphi(double x) {
    if (x <= 0) return 0.0;
    if (x < 10) return -0.4527 * std::pow(x, 0.86) + 0.0218;
    return 0.5 * std::log(M_PI / x) - x / 4 + std::log1p(-10.0 / (7 * x));
}
// Inverse of phi given log(phi) = t (phi is decreasing).
inline double ga_phi_inv_log(double t) {
    double lo = 0, hi = 1e7;
    for (int i = 0; i < 200; ++i) {
        double mid = 0.5 * (lo + hi);
        if (ga_logphi(mid) > t) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}
inline void ga_tree(double mu, int len, std::vector<double>& leaves) {
    if (len == 1) { leaves.push_back(mu); return; }
    // Check node: phi^{-1}(1 - (1 - phi(mu))^2) = phi^{-1}(phi (2 - phi)), in log domain.
    double lp = ga_logphi(mu), phi = std::exp(lp);
    double left = ga_phi_inv_log(lp + std::log(2 - phi));
    ga_tree(left, len / 2, leaves);
    ga_tree(2 * mu, len / 2, leaves);
}

struct PolarCode {
    int m, n, k, c, K;
    std::vector<uint8_t> frozen;
    std::vector<int> info;  // sorted information positions (size K)
    CRC crc;

    PolarCode(int m_, int k_, int c_, double design_ebn0_db)
        : m(m_), n(1 << m_), k(k_), c(c_), K(k_ + c_), crc(c_) {
        double R = double(k) / n;  // energy per payload bit
        double sigma2 = 1.0 / (2 * R * std::pow(10.0, design_ebn0_db / 10));
        std::vector<double> mu;
        ga_tree(2.0 / sigma2, n, mu);
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return mu[a] > mu[b]; });
        frozen.assign(n, 1);
        for (int i = 0; i < K; ++i) frozen[idx[i]] = 0;
        for (int i = 0; i < n; ++i) if (!frozen[i]) info.push_back(i);
    }

    static void transform(uint8_t* a, int n) {  // x = u F^{(x)m}; also its own inverse
        for (int len = 2; len <= n; len <<= 1)
            for (int i = 0; i < n; i += len)
                for (int j = i; j < i + len / 2; ++j) a[j] ^= a[j + len / 2];
    }

    void encode(const uint8_t* payload, uint8_t* x) const {
        std::vector<uint8_t> bits(K);
        std::copy(payload, payload + k, bits.begin());
        crc.append(payload, k, bits.data() + k);
        std::fill(x, x + n, 0);
        for (int i = 0; i < K; ++i) x[info[i]] = bits[i];
        transform(x, n);
    }
};

// ---------------------------------------------------------------- SCL

struct SCLDecoder {
    const PolarCode& pc;
    int L;
    std::vector<double> pm;  // path metrics of the current paths

    SCLDecoder(const PolarCode& p, int list) : pc(p), L(list) {}

    static double f(double a, double b) {
        double ms = ((a > 0) - (a < 0)) * ((b > 0) - (b < 0)) * std::min(std::fabs(a), std::fabs(b));
        return ms + std::log1p(std::exp(-std::fabs(a + b))) - std::log1p(std::exp(-std::fabs(a - b)));
    }
    static double pen(double llr, int u) {  // ln(1 + exp(-(1 - 2u) llr)), stable
        double x = u ? llr : -llr;
        return x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
    }

    // llr: P paths x len (path-major). Outputs beta (P_out x len) and parent (P_out).
    void node(int start, int len, const std::vector<double>& llr, int P,
              std::vector<uint8_t>& beta, std::vector<int>& parent, int& Pout) {
        if (len == 1) {
            if (pc.frozen[start]) {
                Pout = P; parent.resize(P); beta.assign(P, 0);
                for (int p = 0; p < P; ++p) { parent[p] = p; pm[p] += pen(llr[p], 0); }
                return;
            }
            std::vector<std::pair<double, int>> cand(2 * P);
            for (int p = 0; p < P; ++p) {
                cand[2 * p] = {pm[p] + pen(llr[p], 0), 2 * p};
                cand[2 * p + 1] = {pm[p] + pen(llr[p], 1), 2 * p + 1};
            }
            Pout = std::min(2 * P, L);
            std::partial_sort(cand.begin(), cand.begin() + Pout, cand.end());
            parent.resize(Pout); beta.resize(Pout);
            std::vector<double> npm(Pout);
            for (int t = 0; t < Pout; ++t) {
                parent[t] = cand[t].second / 2; beta[t] = cand[t].second & 1; npm[t] = cand[t].first;
            }
            pm = npm;
            return;
        }
        int half = len / 2;
        std::vector<double> l1((size_t)P * half);
        for (int p = 0; p < P; ++p)
            for (int i = 0; i < half; ++i)
                l1[(size_t)p * half + i] = f(llr[(size_t)p * len + i], llr[(size_t)p * len + i + half]);
        std::vector<uint8_t> b1; std::vector<int> par1; int P1;
        node(start, half, l1, P, b1, par1, P1);
        std::vector<double> l2((size_t)P1 * half);
        for (int q = 0; q < P1; ++q) {
            const double* src = &llr[(size_t)par1[q] * len];
            for (int i = 0; i < half; ++i)
                l2[(size_t)q * half + i] = src[i + half] + (b1[(size_t)q * half + i] ? -src[i] : src[i]);
        }
        std::vector<uint8_t> b2; std::vector<int> par2; int P2;
        node(start + half, half, l2, P1, b2, par2, P2);
        Pout = P2; parent.resize(P2); beta.resize((size_t)P2 * len);
        for (int t = 0; t < P2; ++t) {
            parent[t] = par1[par2[t]];
            for (int i = 0; i < half; ++i) {
                uint8_t bb = b2[(size_t)t * half + i];
                beta[(size_t)t * len + i] = b1[(size_t)par2[t] * half + i] ^ bb;
                beta[(size_t)t * len + i + half] = bb;
            }
        }
    }

    // Channel LLRs (length n) -> decoded payload (k bits). Returns true if a
    // path passed the CRC (always true when c = 0).
    bool decode(const double* llr, uint8_t* payload) {
        pm.assign(1, 0.0);
        std::vector<double> l(llr, llr + pc.n);
        std::vector<uint8_t> beta; std::vector<int> parent; int P;
        node(0, pc.n, l, 1, beta, parent, P);
        std::vector<int> order(P);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return pm[a] < pm[b]; });
        std::vector<uint8_t> u(pc.n), bits(pc.K);
        for (int t = 0; t < P; ++t) {
            std::copy(&beta[(size_t)order[t] * pc.n], &beta[(size_t)order[t] * pc.n] + pc.n, u.begin());
            PolarCode::transform(u.data(), pc.n);
            for (int i = 0; i < pc.K; ++i) bits[i] = u[pc.info[i]];
            if (pc.c == 0 || pc.crc.check(bits.data(), pc.k, bits.data() + pc.k) || t == P - 1) {
                std::copy(bits.begin(), bits.begin() + pc.k, payload);
                return pc.c == 0 || pc.crc.check(bits.data(), pc.k, bits.data() + pc.k);
            }
        }
        return false;
    }
};
