// Monte Carlo simulation and verification tool for RM/RPA and polar/SCL.
//
//   sim rpa   --r 2 --m 8 --dec exact|minsum|fixed [--bits 6 --clip 2.5]
//             --ebn0 1,1.5,2 [--target 100 --max 1000000 --threads 2 --seed 1 --chunk 64]
//   sim polar --m 7 --k 29 --crc 8 --list 32 --design 2.0 --ebn0 ...
//   sim width --r 2 --m 8 --bits 6 --clip 2.5 --ebn0 ... --blocks N
//   sim decode --r 2 --m 6 --dec exact|minsum   (stdin: int32 count, count*n float64
//                                                LLRs; stdout: count*n int8 decisions)
//
// Channel: BPSK (0 -> +1, 1 -> -1) over AWGN, sigma^2 = 1 / (2 R Eb/N0), R = k/n
// (k = payload bits). Output: one CSV row per Eb/N0 point. A point stops once
// --target block errors are seen or --max blocks are sent; work is split into
// fixed rounds of seeded chunks, so results do not depend on the thread count.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include "polar.hpp"
#include "rm.hpp"
#include "rpa.hpp"

struct Args {
    std::map<std::string, std::string> kv;
    std::string get(const std::string& k, const std::string& d = "") const {
        auto it = kv.find(k); return it == kv.end() ? d : it->second;
    }
    double num(const std::string& k, double d) const { auto s = get(k); return s.empty() ? d : std::stod(s); }
    std::vector<double> list(const std::string& k) const {
        std::vector<double> v; std::stringstream ss(get(k)); std::string t;
        while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stod(t));
        return v;
    }
};

static uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

struct PointResult { long blocks = 0, errors = 0; double seconds = 0; WidthStats ws; };

// trial(rng, stats) simulates one block and returns true on a block error.
template <class Trial>
PointResult run_point(Trial trial, uint64_t seed, int point, long target, long max_blocks,
                      int threads, int chunk) {
    const int ROUND = 16;  // chunks per round, fixed so results are thread-count independent
    PointResult res;
    auto t0 = std::chrono::steady_clock::now();
    long next_chunk = 0;
    while (res.errors < target && res.blocks < max_blocks) {
        std::vector<long> err(ROUND, 0), blk(ROUND, 0);
        std::vector<WidthStats> ws(ROUND);
        std::atomic<int> idx{0};
        auto worker = [&]() {
            for (int c; (c = idx.fetch_add(1)) < ROUND;) {
                long gc = next_chunk + c;
                long nb = std::min<long>(chunk, max_blocks - res.blocks - (long)c * chunk);
                if (nb <= 0) continue;
                Rng rng(splitmix(seed * 0x100000001B3ull + (uint64_t)point * 0x9E3779B1ull + (uint64_t)gc));
                for (long b = 0; b < nb; ++b) err[c] += trial(rng, ws[c]) ? 1 : 0;
                blk[c] = nb;
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
        worker();
        for (auto& th : pool) th.join();
        for (int c = 0; c < ROUND; ++c) { res.errors += err[c]; res.blocks += blk[c]; res.ws.merge(ws[c]); }
        next_chunk += ROUND;
    }
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

static double sigma_for(double ebn0_db, double rate) {
    return std::sqrt(1.0 / (2 * rate * std::pow(10.0, ebn0_db / 10)));
}

static int bits_for(int64_t maxabs) {  // signed two's-complement width holding [-maxabs, maxabs]
    int w = 1;
    while ((int64_t(1) << (w - 1)) - 1 < maxabs) ++w;
    return w;
}

// ---------------------------------------------------------------- rpa

int cmd_rpa(const Args& a) {
    int r = (int)a.num("--r", 2), m = (int)a.num("--m", 8);
    std::string dec = a.get("--dec", "exact");
    int bits = (int)a.num("--bits", 6);
    double clip = a.num("--clip", 2.5);
    // --clip-rms c: clip level c * sqrt(E[y^2]) = c * sqrt(1 + sigma^2), i.e. relative to
    // the received RMS that an AGC measures (no noise-variance estimate needed).
    double clip_rms = a.num("--clip-rms", 0.0);
    long target = (long)a.num("--target", 100), maxb = (long)a.num("--max", 1e6);
    int threads = (int)a.num("--threads", 2), chunk = (int)a.num("--chunk", 64);
    uint64_t seed = (uint64_t)a.num("--seed", 1);
    RMCode code(r, m);
    warm_tables(r, m);
    const double R = double(code.k) / code.n;
    RPAFloat fl; fl.comb = dec == "exact" ? Combine::Exact : Combine::MinSum;
    RPAFixed fx; fx.A = (1 << (bits - 1)) - 1;
    std::string label = dec != "fixed" ? dec
        : clip_rms > 0 ? "fixed" + std::to_string(bits) + "_rms" + a.get("--clip-rms")
                       : "fixed" + std::to_string(bits) + "_clip" + a.get("--clip", "2.5");
    std::printf("scheme,code,n,k,decoder,ebn0_db,blocks,block_errors,bler,seconds\n");
    int pi = 0;
    for (double eb : a.list("--ebn0")) {
        double sigma = sigma_for(eb, R);
        auto trial = [&](Rng& rng, WidthStats& ws) {
            std::normal_distribution<double> N(0.0, 1.0);
            std::vector<uint8_t> u(code.k), c(code.n);
            for (auto& x : u) x = rng() & 1;
            code.encode(u.data(), c.data());
            std::vector<int8_t> out(code.n);
            if (dec == "fixed") {
                std::vector<int32_t> q(code.n);
                double T = clip_rms > 0 ? clip_rms * std::sqrt(1.0 + sigma * sigma) : clip;
                double step = T / fx.A;
                for (int z = 0; z < code.n; ++z) q[z] = quantize(1.0 - 2.0 * c[z] + sigma * N(rng), step, fx.A);
                fx.decode(q.data(), r, m, out.data(), rng, ws);
            } else {
                std::vector<double> L(code.n);
                for (int z = 0; z < code.n; ++z) L[z] = 2.0 * (1.0 - 2.0 * c[z] + sigma * N(rng)) / (sigma * sigma);
                fl.decode(L.data(), r, m, out.data());
            }
            for (int z = 0; z < code.n; ++z) if (out[z] != 1 - 2 * c[z]) return true;
            return false;
        };
        PointResult pr = run_point(trial, seed, pi++, target, maxb, threads, chunk);
        std::printf("RM,RM(%d.%d),%d,%d,rpa_%s,%.2f,%ld,%ld,%.6e,%.1f\n", r, m, code.n, code.k,
                    label.c_str(), eb, pr.blocks, pr.errors, double(pr.errors) / pr.blocks, pr.seconds);
        std::fflush(stdout);
        if (pr.errors == 0) break;
    }
    return 0;
}

// ---------------------------------------------------------------- polar

int cmd_polar(const Args& a) {
    int m = (int)a.num("--m", 7), k = (int)a.num("--k", 29), c = (int)a.num("--crc", 8);
    int list = (int)a.num("--list", 32);
    double design = a.num("--design", 2.0);
    long target = (long)a.num("--target", 100), maxb = (long)a.num("--max", 1e6);
    int threads = (int)a.num("--threads", 2), chunk = (int)a.num("--chunk", 64);
    uint64_t seed = (uint64_t)a.num("--seed", 1);
    PolarCode pc(m, k, c, design);
    const double R = double(k) / pc.n;
    std::printf("scheme,code,n,k,decoder,ebn0_db,blocks,block_errors,bler,seconds\n");
    int pi = 0;
    for (double eb : a.list("--ebn0")) {
        double sigma = sigma_for(eb, R);
        auto trial = [&](Rng& rng, WidthStats&) {
            std::normal_distribution<double> N(0.0, 1.0);
            std::vector<uint8_t> u(k), x(pc.n), uh(k);
            for (auto& b : u) b = rng() & 1;
            pc.encode(u.data(), x.data());
            std::vector<double> L(pc.n);
            for (int z = 0; z < pc.n; ++z) L[z] = 2.0 * (1.0 - 2.0 * x[z] + sigma * N(rng)) / (sigma * sigma);
            SCLDecoder dec(pc, list);
            dec.decode(L.data(), uh.data());
            return u != uh;
        };
        PointResult pr = run_point(trial, seed, pi++, target, maxb, threads, chunk);
        std::printf("polar,polar(%d.%d),%d,%d,scl%d_crc%d_d%.2f,%.2f,%ld,%ld,%.6e,%.1f\n", pc.n, k, pc.n, k,
                    list, c, design, eb, pr.blocks, pr.errors, double(pr.errors) / pr.blocks, pr.seconds);
        std::fflush(stdout);
        if (pr.errors == 0) break;
    }
    return 0;
}

// ---------------------------------------------------------------- width check

int cmd_width(const Args& a) {
    int r = (int)a.num("--r", 2), m = (int)a.num("--m", 8), bits = (int)a.num("--bits", 6);
    double clip = a.num("--clip", 2.5);
    long blocks = (long)a.num("--blocks", 2000);
    RMCode code(r, m);
    warm_tables(r, m);
    RPAFixed fx; fx.A = (1 << (bits - 1)) - 1;
    const int A = fx.A;
    std::printf("level_m,quantity,bound,bound_bits,observed_adversarial,observed_random,observed_bits\n");
    // Adversarial inputs: +-A on a random codeword pattern (maximally reliable word),
    // and the constant word +A.
    WidthStats adv, rnd;
    Rng rng(12345);
    std::vector<int8_t> out(code.n);
    for (int t = 0; t < 64; ++t) {
        std::vector<uint8_t> u(code.k), c(code.n);
        if (t > 0) for (auto& x : u) x = rng() & 1;
        code.encode(u.data(), c.data());
        std::vector<int32_t> q(code.n);
        for (int z = 0; z < code.n; ++z) q[z] = c[z] ? -A : A;
        fx.decode(q.data(), r, m, out.data(), rng, adv);
    }
    const double R = double(code.k) / code.n;
    for (double eb : a.list("--ebn0")) {
        double sigma = sigma_for(eb, R), step = clip / A;
        std::normal_distribution<double> N(0.0, 1.0);
        for (long b = 0; b < blocks; ++b) {
            std::vector<uint8_t> u(code.k), c(code.n);
            for (auto& x : u) x = rng() & 1;
            code.encode(u.data(), c.data());
            std::vector<int32_t> q(code.n);
            for (int z = 0; z < code.n; ++z) q[z] = quantize(1.0 - 2.0 * c[z] + sigma * N(rng), step, A);
            fx.decode(q.data(), r, m, out.data(), rng, rnd);
        }
    }
    // Bounds (docs/theory.md, Theorem 3): stored |L| <= A at every level; the FHT on
    // RM(1, m') reaches at most 2^{m'} A; aggregation for RM(r', m') reaches (2^{m'} - 1) A.
    for (int lv = m; lv >= m - r + 1; --lv) {
        int64_t sb = A;
        std::printf("%d,store,%lld,%d,%lld,%lld,%d\n", lv, (long long)sb, bits_for(sb),
                    (long long)adv.max_store[lv], (long long)rnd.max_store[lv],
                    bits_for(std::max(adv.max_store[lv], rnd.max_store[lv])));
        if (lv == m - r + 1) {
            int64_t fb = (int64_t(1) << lv) * A;
            std::printf("%d,fht,%lld,%d,%lld,%lld,%d\n", lv, (long long)fb, bits_for(fb),
                        (long long)adv.max_fht[lv], (long long)rnd.max_fht[lv],
                        bits_for(std::max(adv.max_fht[lv], rnd.max_fht[lv])));
        } else {
            int64_t cb = ((int64_t(1) << lv) - 1) * A;
            std::printf("%d,aggregate,%lld,%d,%lld,%lld,%d\n", lv, (long long)cb, bits_for(cb),
                        (long long)adv.max_cumu[lv], (long long)rnd.max_cumu[lv],
                        bits_for(std::max(adv.max_cumu[lv], rnd.max_cumu[lv])));
        }
    }
    return 0;
}

// ---------------------------------------------------------------- decode (cross-check)

int cmd_decode(const Args& a) {
    int r = (int)a.num("--r", 2), m = (int)a.num("--m", 6);
    std::string dec = a.get("--dec", "exact");
    RMCode code(r, m);
    warm_tables(r, m);
    RPAFloat fl; fl.comb = dec == "exact" ? Combine::Exact : Combine::MinSum;
    int32_t count = 0;
    if (std::fread(&count, sizeof count, 1, stdin) != 1) return 1;
    std::vector<double> L(code.n);
    std::vector<int8_t> out(code.n);
    for (int i = 0; i < count; ++i) {
        if (std::fread(L.data(), sizeof(double), code.n, stdin) != (size_t)code.n) return 1;
        fl.decode(L.data(), r, m, out.data());
        std::fwrite(out.data(), 1, code.n, stdout);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: sim rpa|polar|width|decode [--key value ...]\n"); return 2; }
    Args a;
    for (int i = 2; i + 1 < argc; i += 2) a.kv[argv[i]] = argv[i + 1];
    std::string cmd = argv[1];
    if (cmd == "rpa") return cmd_rpa(a);
    if (cmd == "polar") return cmd_polar(a);
    if (cmd == "width") return cmd_width(a);
    if (cmd == "decode") return cmd_decode(a);
    std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
    return 2;
}
