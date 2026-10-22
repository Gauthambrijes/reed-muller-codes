// Reed-Muller code RM(r, m): generator matrix and encoder.
//
// Same conventions as rmcodes/code.py: coordinate z in 0..2^m-1 is the point of
// GF(2)^m whose i-th component is bit i of z; message bits are monomial
// coefficients ordered by degree, then lexicographically.
#pragma once
#include <cstdint>
#include <vector>

struct RMCode {
    int r, m, n, k;
    std::vector<std::vector<uint8_t>> G;  // k rows of length n

    RMCode(int r_, int m_) : r(r_), m(m_), n(1 << m_) {
        std::vector<std::vector<int>> monos;
        for (int d = 0; d <= r; ++d) {
            std::vector<int> c(d);
            combos(0, 0, d, c, monos);
        }
        k = static_cast<int>(monos.size());
        G.assign(k, std::vector<uint8_t>(n, 1));
        for (int j = 0; j < k; ++j)
            for (int z = 0; z < n; ++z)
                for (int v : monos[j]) G[j][z] &= (z >> v) & 1;
    }

    void encode(const uint8_t* u, uint8_t* c) const {
        for (int z = 0; z < n; ++z) c[z] = 0;
        for (int j = 0; j < k; ++j)
            if (u[j])
                for (int z = 0; z < n; ++z) c[z] ^= G[j][z];
    }

    // Number of minimum-weight codewords and d_min, for reference.
    int dmin() const { return 1 << (m - r); }

  private:
    void combos(int start, int idx, int d, std::vector<int>& c,
                std::vector<std::vector<int>>& out) {
        if (idx == d) { out.push_back(c); return; }
        for (int v = start; v < m; ++v) { c[idx] = v; combos(v + 1, idx + 1, d, c, out); }
    }
};
