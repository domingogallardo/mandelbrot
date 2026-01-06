/*
 * MANDELBROT EXPLORER
 * Ultra-fast console-based fractal explorer
 *
 * Controls:
 *   Arrow Keys          - Pan view
 *   SHIFT + Up/Down     - Zoom in/out
 *   SHIFT + Left/Right  - Rotate view angle
 *   C/V                 - Rotate color palette
 *   1-9                 - Switch color schemes
 *   +/-                 - Increase/decrease max iterations
 *   I                   - Toggle iTerm2 image mode (higher resolution)
 *   R                   - Reset view
 *   Q/ESC               - Quit
 *
 * CLI Arguments:
 *   --pos <re+imi>      - Target position (e.g., -0.5+0.0i)
 *   --zoom <value>      - Target zoom level
 *   --angle <degrees>   - Target view angle
 *   --auto [N]          - Auto mode, or trajectory to --pos/--zoom over N sec
 *   --image [WxH]       - Enable iTerm2 inline image mode
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <csignal>
#include <string>
#include <algorithm>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include <getopt.h>

#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// ═══════════════════════════════════════════════════════════════════════════
// DOUBLE-DOUBLE ARITHMETIC (for high-precision reference orbit)
// ═══════════════════════════════════════════════════════════════════════════

// CRITICAL: DD arithmetic requires strict IEEE rounding - disable fast-math.
// For GCC: use pragma. For Clang: use function attributes (see functions below).
// If compiling with -ffast-math, these functions MUST retain strict IEEE semantics.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math")
#endif

// Clang attribute to disable fast-math optimizations for specific functions
#if defined(__clang__)
#define STRICT_FP __attribute__((optnone))
#else
#define STRICT_FP
#endif

// Double-double: represents value as hi + lo where |lo| << |hi|
struct DD {
    double hi, lo;

    DD() : hi(0), lo(0) {}
    DD(double h) : hi(h), lo(0) {}
    DD(double h, double l) : hi(h), lo(l) {}

    explicit operator double() const { return hi + lo; }
};

// Error-free transformations (require strict IEEE rounding)

// Two-Sum: a + b = s + e exactly
STRICT_FP inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    double v = s - a;
    e = (a - (s - v)) + (b - v);
}

// Quick-Two-Sum: when |a| >= |b| is guaranteed
STRICT_FP inline void quick_two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    e = b - (s - a);
}

// Two-Prod: a * b = p + e exactly (uses FMA when available)
STRICT_FP inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
#ifdef FP_FAST_FMA
    e = std::fma(a, b, -p);
#else
    // Dekker splitting fallback for systems without FMA
    constexpr double SPLIT = 134217729.0; // 2^27 + 1
    double ca = SPLIT * a;
    double cb = SPLIT * b;
    double a_hi = ca - (ca - a);
    double a_lo = a - a_hi;
    double b_hi = cb - (cb - b);
    double b_lo = b - b_hi;
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
#endif
}

// DD + DD
inline DD dd_add(DD a, DD b) {
    double s1, s2, t1, t2;
    two_sum(a.hi, b.hi, s1, s2);
    two_sum(a.lo, b.lo, t1, t2);
    s2 += t1;
    quick_two_sum(s1, s2, s1, s2);
    s2 += t2;
    quick_two_sum(s1, s2, s1, s2);
    return DD(s1, s2);
}

// DD + double
inline DD dd_add(DD a, double b) {
    double s1, s2;
    two_sum(a.hi, b, s1, s2);
    s2 += a.lo;
    quick_two_sum(s1, s2, s1, s2);
    return DD(s1, s2);
}

// DD - DD
inline DD dd_sub(DD a, DD b) {
    return dd_add(a, DD(-b.hi, -b.lo));
}

// DD - double
inline DD dd_sub(DD a, double b) {
    return dd_add(a, -b);
}

// DD * DD
inline DD dd_mul(DD a, DD b) {
    double p1, p2;
    two_prod(a.hi, b.hi, p1, p2);
    p2 += a.hi * b.lo + a.lo * b.hi;
    quick_two_sum(p1, p2, p1, p2);
    return DD(p1, p2);
}

// DD * double
inline DD dd_mul(DD a, double b) {
    double p1, p2;
    two_prod(a.hi, b, p1, p2);
    p2 += a.lo * b;
    quick_two_sum(p1, p2, p1, p2);
    return DD(p1, p2);
}

// DD / double (simplified division)
inline DD dd_div(DD a, double b) {
    double q1 = a.hi / b;
    double p1, p2;
    two_prod(q1, b, p1, p2);
    double s, e;
    two_sum(a.hi, -p1, s, e);
    e += a.lo;
    e -= p2;
    double q2 = (s + e) / b;
    quick_two_sum(q1, q2, q1, q2);
    return DD(q1, q2);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

// ═══════════════════════════════════════════════════════════════════════════
// DD COMPLEX TYPE (for reference orbit computation)
// ═══════════════════════════════════════════════════════════════════════════

struct DDComplex {
    DD re, im;

    DDComplex() : re(), im() {}
    DDComplex(DD r, DD i) : re(r), im(i) {}
    DDComplex(double r, double i) : re(r), im(i) {}

    // Square: (a + bi)^2 = a^2 - b^2 + 2abi
    DDComplex square() const {
        DD re2 = dd_mul(re, re);
        DD im2 = dd_mul(im, im);
        DD new_re = dd_sub(re2, im2);
        DD two_re_im = dd_mul(dd_mul(re, im), 2.0);
        return DDComplex(new_re, two_re_im);
    }

    DDComplex operator+(const DDComplex& other) const {
        return DDComplex(dd_add(re, other.re), dd_add(im, other.im));
    }

    // Approximate |z|^2 using only hi parts (for fast escape check)
    double norm_approx() const {
        return re.hi * re.hi + im.hi * im.hi;
    }

    // Full DD precision |z|^2 (for guaranteed correct escape check)
    double norm_full() const {
        DD re2 = dd_mul(re, re);
        DD im2 = dd_mul(im, im);
        DD norm_dd = dd_add(re2, im2);
        return norm_dd.hi + norm_dd.lo;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE ORBIT (for perturbation theory)
// ═══════════════════════════════════════════════════════════════════════════

struct ReferenceOrbit {
    // Store full DD precision: hi and lo parts separately
    std::vector<double> Zr_hi, Zr_lo;  // DD real parts
    std::vector<double> Zi_hi, Zi_lo;  // DD imaginary parts
    // Precomputed sums for fast access (avoids hi+lo in inner loop)
    std::vector<double> Zr_sum;        // Zr_hi + Zr_lo
    std::vector<double> Zi_sum;        // Zi_hi + Zi_lo
    std::vector<double> Z_norm;        // |Z_n|^2 computed from sums
    int length;                        // Actual orbit length (may be < max_iter)
    int escape_iter;                   // Iteration at escape, or -1 if bounded
    DD center_re, center_im;           // High-precision center for this orbit

    // ═══════════════════════════════════════════════════════════════════════
    // SERIES APPROXIMATION (SA) COEFFICIENTS
    // ═══════════════════════════════════════════════════════════════════════
    // For pixel at C = C_ref + δC, the orbit deviation is:
    //   δZ_n ≈ A_n * δC + B_n * δC² + C_n * δC³ + ...
    // This allows skipping many iterations by evaluating the polynomial.
    // Coefficients are complex, stored as real/imag pairs.
    std::vector<double> SA_Ar, SA_Ai;  // A coefficient (linear term)
    std::vector<double> SA_Br, SA_Bi;  // B coefficient (quadratic term)
    std::vector<double> SA_Cr, SA_Ci;  // C coefficient (cubic term)
    std::vector<double> SA_Dr, SA_Di;  // D coefficient (quartic term)
    std::vector<double> SA_A_norm;     // |A_n|² for validity checks
    std::vector<double> SA_D_norm;     // |D_n|² for 4-term validity checks
    std::vector<double> sa_max_dC_norm; // Precomputed max |δC|² for SA validity at each n
    std::vector<double> sa_seg_tree;    // Segment tree for max-threshold queries (rightmost valid n)
    bool sa_enabled = false;           // Whether SA was computed
    bool sa_thresholds_built = false;  // Whether sa_max_dC_norm is computed

    ReferenceOrbit() : length(0), escape_iter(-1), sa_enabled(false), sa_thresholds_built(false) {}

    void clear() {
        Zr_hi.clear();
        Zr_lo.clear();
        Zi_hi.clear();
        Zi_lo.clear();
        Zr_sum.clear();
        Zi_sum.clear();
        Z_norm.clear();
        SA_Ar.clear(); SA_Ai.clear();
        SA_Br.clear(); SA_Bi.clear();
        SA_Cr.clear(); SA_Ci.clear();
        SA_Dr.clear(); SA_Di.clear();
        SA_A_norm.clear();
        SA_D_norm.clear();
        sa_max_dC_norm.clear();
        sa_seg_tree.clear();
        length = 0;
        escape_iter = -1;
        sa_enabled = false;
        sa_thresholds_built = false;
    }
};

// Glitch detection constants
// Key insight: δz grows exponentially, so it WILL become larger than Z.
// This is expected behavior, not a glitch. A true glitch is when precision
// is lost in the δz computation itself. We use very lenient thresholds
// and primarily rely on proper reference orbit fallback handling.
constexpr double GLITCH_BASE_RATIO = 1e6;       // Only glitch if δz dominates by 1000x
constexpr double GLITCH_ESCAPE_RATIO = 1e8;     // Even more lenient near escape
constexpr double GLITCH_FLOOR = 1e-30;          // Prevent div-by-zero
constexpr int GLITCH_MIN_ITER = 50;             // Skip early iterations
constexpr double NEAR_ESCAPE_THRESHOLD = 3.5;   // |Z|² threshold for "near escape"
constexpr int ESCAPE_WINDOW = 20;               // Iterations before escape_iter to relax
constexpr int MAX_REFERENCES = 8;               // Allow multiple reference orbits
constexpr double GLITCH_RATE_LIMIT = 0.15;      // Max glitch rate before DD fallback

// Compute reference orbit at high precision, store with full DD precision
// Also computes Series Approximation (SA) coefficients for iteration skipping
inline void compute_reference_orbit(ReferenceOrbit& orbit,
                                    DD center_x, DD center_y,
                                    int max_iter,
                                    bool enable_sa = true) {
    orbit.clear();
    orbit.center_re = center_x;
    orbit.center_im = center_y;

    // Reserve space for all arrays
    orbit.Zr_hi.reserve(max_iter + 1);
    orbit.Zr_lo.reserve(max_iter + 1);
    orbit.Zi_hi.reserve(max_iter + 1);
    orbit.Zi_lo.reserve(max_iter + 1);
    orbit.Zr_sum.reserve(max_iter + 1);
    orbit.Zi_sum.reserve(max_iter + 1);
    orbit.Z_norm.reserve(max_iter + 1);

    // Reserve space for SA coefficients (only if enabled)
    if (enable_sa) {
        orbit.SA_Ar.reserve(max_iter + 1);
        orbit.SA_Ai.reserve(max_iter + 1);
        orbit.SA_Br.reserve(max_iter + 1);
        orbit.SA_Bi.reserve(max_iter + 1);
        orbit.SA_Cr.reserve(max_iter + 1);
        orbit.SA_Ci.reserve(max_iter + 1);
        orbit.SA_Dr.reserve(max_iter + 1);
        orbit.SA_Di.reserve(max_iter + 1);
        orbit.SA_A_norm.reserve(max_iter + 1);
        orbit.SA_D_norm.reserve(max_iter + 1);
    }

    // Initial Z = 0
    orbit.Zr_hi.push_back(0.0);
    orbit.Zr_lo.push_back(0.0);
    orbit.Zi_hi.push_back(0.0);
    orbit.Zi_lo.push_back(0.0);
    orbit.Zr_sum.push_back(0.0);
    orbit.Zi_sum.push_back(0.0);
    orbit.Z_norm.push_back(0.0);

    // Initial SA coefficients: A_0 = B_0 = C_0 = D_0 = 0 (since δZ_0 = 0)
    if (enable_sa) {
        orbit.SA_Ar.push_back(0.0);
        orbit.SA_Ai.push_back(0.0);
        orbit.SA_Br.push_back(0.0);
        orbit.SA_Bi.push_back(0.0);
        orbit.SA_Cr.push_back(0.0);
        orbit.SA_Ci.push_back(0.0);
        orbit.SA_Dr.push_back(0.0);
        orbit.SA_Di.push_back(0.0);
        orbit.SA_A_norm.push_back(0.0);
        orbit.SA_D_norm.push_back(0.0);
    }

    DDComplex C(center_x, center_y);
    DDComplex Z(0.0, 0.0);

    // SA coefficients (complex numbers) - only used if enable_sa
    // δZ_n = A_n*δC + B_n*δC² + C_n*δC³ + D_n*δC⁴
    double Ar = 0.0, Ai = 0.0;  // A_n
    double Br = 0.0, Bi = 0.0;  // B_n
    double Cr = 0.0, Ci = 0.0;  // C_n
    double Dr = 0.0, Di = 0.0;  // D_n

    for (int n = 0; n < max_iter; n++) {
        Z = Z.square() + C;

        // Store full DD precision
        orbit.Zr_hi.push_back(Z.re.hi);
        orbit.Zr_lo.push_back(Z.re.lo);
        orbit.Zi_hi.push_back(Z.im.hi);
        orbit.Zi_lo.push_back(Z.im.lo);

        // Precompute sums for fast access in perturbation loop
        double zr_sum = Z.re.hi + Z.re.lo;
        double zi_sum = Z.im.hi + Z.im.lo;
        orbit.Zr_sum.push_back(zr_sum);
        orbit.Zi_sum.push_back(zi_sum);

        // Compute norm from full-precision sums
        double norm = zr_sum * zr_sum + zi_sum * zi_sum;
        orbit.Z_norm.push_back(norm);

        // ═══════════════════════════════════════════════════════════════════
        // SA COEFFICIENT RECURRENCE (only if SA is enabled)
        // ═══════════════════════════════════════════════════════════════════
        if (enable_sa) {
            // Given: δZ_{n+1} = 2*Z_n*δZ_n + δZ_n² + δC
            // With: δZ_n = A_n*δC + B_n*δC² + C_n*δC³ + D_n*δC⁴ + ...
            //
            // Recurrence relations (complex multiplication):
            //   A_{n+1} = 2*Z_n*A_n + 1
            //   B_{n+1} = 2*Z_n*B_n + A_n²
            //   C_{n+1} = 2*Z_n*C_n + 2*A_n*B_n
            //   D_{n+1} = 2*Z_n*D_n + (2*A_n*C_n + B_n²)

            // Use Z from previous iteration (Z_n before the square+C above)
            double Zr_prev = (n == 0) ? 0.0 : orbit.Zr_sum[n];
            double Zi_prev = (n == 0) ? 0.0 : orbit.Zi_sum[n];

            // A_{n+1} = 2*Z_n*A_n + 1
            double new_Ar = 2.0 * (Zr_prev * Ar - Zi_prev * Ai) + 1.0;
            double new_Ai = 2.0 * (Zr_prev * Ai + Zi_prev * Ar);

            // B_{n+1} = 2*Z_n*B_n + A_n²
            double A2r = Ar * Ar - Ai * Ai;
            double A2i = 2.0 * Ar * Ai;
            double new_Br = 2.0 * (Zr_prev * Br - Zi_prev * Bi) + A2r;
            double new_Bi = 2.0 * (Zr_prev * Bi + Zi_prev * Br) + A2i;

            // C_{n+1} = 2*Z_n*C_n + 2*A_n*B_n
            double ABr = Ar * Br - Ai * Bi;
            double ABi = Ar * Bi + Ai * Br;
            double new_Cr = 2.0 * (Zr_prev * Cr - Zi_prev * Ci) + 2.0 * ABr;
            double new_Ci = 2.0 * (Zr_prev * Ci + Zi_prev * Cr) + 2.0 * ABi;

            // D_{n+1} = 2*Z_n*D_n + (2*A_n*C_n + B_n²)
            // A*C = (Ar*Cr - Ai*Ci) + (Ar*Ci + Ai*Cr)*i
            double ACr = Ar * Cr - Ai * Ci;
            double ACi = Ar * Ci + Ai * Cr;
            // B² = (Br² - Bi²) + (2*Br*Bi)*i
            double B2r = Br * Br - Bi * Bi;
            double B2i = 2.0 * Br * Bi;
            double new_Dr = 2.0 * (Zr_prev * Dr - Zi_prev * Di) + 2.0 * ACr + B2r;
            double new_Di = 2.0 * (Zr_prev * Di + Zi_prev * Dr) + 2.0 * ACi + B2i;

            Ar = new_Ar; Ai = new_Ai;
            Br = new_Br; Bi = new_Bi;
            Cr = new_Cr; Ci = new_Ci;
            Dr = new_Dr; Di = new_Di;

            // Store coefficients
            orbit.SA_Ar.push_back(Ar);
            orbit.SA_Ai.push_back(Ai);
            orbit.SA_Br.push_back(Br);
            orbit.SA_Bi.push_back(Bi);
            orbit.SA_Cr.push_back(Cr);
            orbit.SA_Ci.push_back(Ci);
            orbit.SA_Dr.push_back(Dr);
            orbit.SA_Di.push_back(Di);
            orbit.SA_A_norm.push_back(Ar * Ar + Ai * Ai);
            orbit.SA_D_norm.push_back(Dr * Dr + Di * Di);
        }

        // Use larger escape radius for reference (extends orbit length)
        if (norm > 1e6) {
            orbit.escape_iter = n + 1;
            orbit.length = n + 2;
            orbit.sa_enabled = enable_sa;
            return;
        }
    }

    orbit.escape_iter = -1;  // Didn't escape
    orbit.length = max_iter + 1;
    orbit.sa_enabled = enable_sa;
}

// ═══════════════════════════════════════════════════════════════════════════
// PRECOMPUTED SA THRESHOLDS (Optimization 1.1)
// ═══════════════════════════════════════════════════════════════════════════
// Precompute maximum allowed |δC|² at each iteration for SA validity.
// This allows O(log n) skip lookup per tile instead of O(n) per pixel.

constexpr double SA_TOLERANCE_PRECOMPUTE = 0.001;  // Conservative tolerance for SA validity

// Build segment tree for max-threshold queries
// Tree[1] = max of entire range, Tree[2*i] = left child, Tree[2*i+1] = right child
static void build_seg_tree(std::vector<double>& tree, const std::vector<double>& arr, int node, int lo, int hi) {
    if (lo == hi) {
        tree[node] = arr[lo];
        return;
    }
    int mid = (lo + hi) / 2;
    build_seg_tree(tree, arr, 2 * node, lo, mid);
    build_seg_tree(tree, arr, 2 * node + 1, mid + 1, hi);
    tree[node] = std::max(tree[2 * node], tree[2 * node + 1]);
}

// Query segment tree for rightmost index where threshold >= query_val
// Returns -1 if no such index exists in range [lo, hi]
static int query_rightmost_valid(const std::vector<double>& tree, const std::vector<double>& arr,
                                  int node, int lo, int hi, double query_val, int range_lo, int range_hi) {
    // If current segment is outside query range or max in segment < query_val, no valid index here
    if (hi < range_lo || lo > range_hi || tree[node] < query_val) {
        return -1;
    }
    // Leaf node
    if (lo == hi) {
        return (arr[lo] >= query_val) ? lo : -1;
    }
    int mid = (lo + hi) / 2;
    // Search right child first (we want rightmost)
    int right_result = query_rightmost_valid(tree, arr, 2 * node + 1, mid + 1, hi, query_val, range_lo, range_hi);
    if (right_result != -1) return right_result;
    // If not found in right, search left
    return query_rightmost_valid(tree, arr, 2 * node, lo, mid, query_val, range_lo, range_hi);
}

inline void build_sa_thresholds(ReferenceOrbit& orbit, double tol = SA_TOLERANCE_PRECOMPUTE) {
    if (!orbit.sa_enabled || orbit.length < 2) {
        orbit.sa_thresholds_built = false;
        return;
    }

    int N = orbit.length;
    orbit.sa_max_dC_norm.resize(N);
    orbit.sa_max_dC_norm[0] = 0.0;  // n=0 is not valid for SA

    double tol_sq = tol * tol;

    for (int n = 1; n < N; n++) {
        double A2 = orbit.SA_A_norm[n];
        double D2 = orbit.SA_D_norm[n];
        double thr = 0.0;

        if (std::isfinite(A2) && A2 > 0.0) {
            if (!std::isfinite(D2) || D2 == 0.0) {
                // D² == 0 or invalid: SA validity |D|²*|δC|⁶ < tol²*|A|² is ALWAYS TRUE
                // (0 * anything < positive), so threshold is infinity
                thr = std::numeric_limits<double>::infinity();
            } else {
                // Normal case: solve |δC|² < (tol² * |A|² / |D|²)^(1/3)
                thr = std::cbrt(tol_sq * (A2 / D2));
            }
        }
        // If A2 is invalid/zero, SA approximation is degenerate, keep thr = 0

        orbit.sa_max_dC_norm[n] = thr;
    }

    // Build segment tree for max queries (size 4*N is safe for any N)
    orbit.sa_seg_tree.resize(4 * N, 0.0);
    build_seg_tree(orbit.sa_seg_tree, orbit.sa_max_dC_norm, 1, 0, N - 1);

    orbit.sa_thresholds_built = true;
}

// Fast SA skip lookup using segment tree - O(log n) rightmost-valid query
// Returns the largest n where sa_max_dC_norm[n] >= dC_norm (i.e., SA is valid at n)
inline int sa_max_skip_from_dC_norm(const ReferenceOrbit& orbit, double dC_norm, int max_iter) {
    if (!orbit.sa_thresholds_built || dC_norm <= 0.0) {
        return 0;
    }

    int N = orbit.length;
    int hi = std::min(max_iter, N - 1);
    if (hi < 1) return 0;

    // Query segment tree for rightmost index in [1, hi] where threshold >= dC_norm
    int result = query_rightmost_valid(orbit.sa_seg_tree, orbit.sa_max_dC_norm,
                                        1, 0, N - 1, dC_norm, 1, hi);
    return (result > 0) ? result : 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SKIP CACHING: Use previous pixel's skip as hint for O(1) common case
// ═══════════════════════════════════════════════════════════════════════════
// At high zoom, all pixels have essentially the same dC_norm, so they all get
// the same skip value. Check if hint is still valid AND optimal (hint+1 invalid).
// If so, return hint immediately (O(1)). Otherwise fall back to segment tree.
inline int sa_max_skip_with_hint(const ReferenceOrbit& orbit, double dC_norm, int max_iter, int hint) {
    if (!orbit.sa_thresholds_built || dC_norm <= 0.0) {
        return 0;
    }

    int N = orbit.length;
    int hi = std::min(max_iter, N - 1);
    if (hi < 1) return 0;

    // Fast path: check if hint is valid AND optimal (hint+1 is invalid or out of range)
    if (hint >= 1 && hint <= hi && orbit.sa_max_dC_norm[hint] >= dC_norm) {
        // Hint is valid - check if it's the rightmost valid
        if (hint >= hi || orbit.sa_max_dC_norm[hint + 1] < dC_norm) {
            return hint;  // Hint is optimal
        }
    }

    // Fall back to segment tree query
    int result = query_rightmost_valid(orbit.sa_seg_tree, orbit.sa_max_dC_norm,
                                        1, 0, N - 1, dC_norm, 1, hi);
    return (result > 0) ? result : 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// VECTORIZED SA EVALUATION (Optimization 1.2)
// ═══════════════════════════════════════════════════════════════════════════

#if USE_AVX2
// Complex multiply with optional FMA: (ar + ai*i) * (br + bi*i)
static inline void cmul_avx2(__m256d ar, __m256d ai, __m256d br, __m256d bi,
                              __m256d& rr, __m256d& ri) {
#ifdef __FMA__
    rr = _mm256_fmsub_pd(ar, br, _mm256_mul_pd(ai, bi));
    ri = _mm256_fmadd_pd(ar, bi, _mm256_mul_pd(ai, br));
#else
    rr = _mm256_sub_pd(_mm256_mul_pd(ar, br), _mm256_mul_pd(ai, bi));
    ri = _mm256_add_pd(_mm256_mul_pd(ar, bi), _mm256_mul_pd(ai, br));
#endif
}

// Vectorized 4-term SA Horner evaluation for 4 pixels
// δZ = A*δC + B*δC² + C*δC³ + D*δC⁴ = (((D*δC + C)*δC + B)*δC + A)*δC
// Returns true if all lanes are finite, false if any lane has NaN/inf
static inline bool sa_eval4(const ReferenceOrbit& orbit, int n,
                            __m256d dCr, __m256d dCi,
                            __m256d& dzr, __m256d& dzi) {
    // Broadcast coefficients for iteration n
    __m256d hr = _mm256_set1_pd(orbit.SA_Dr[n]);
    __m256d hi = _mm256_set1_pd(orbit.SA_Di[n]);
    __m256d tr, ti;

    // Horner step 1: h = D*δC + C
    cmul_avx2(hr, hi, dCr, dCi, tr, ti);
    hr = _mm256_add_pd(tr, _mm256_set1_pd(orbit.SA_Cr[n]));
    hi = _mm256_add_pd(ti, _mm256_set1_pd(orbit.SA_Ci[n]));

    // Horner step 2: h = h*δC + B
    cmul_avx2(hr, hi, dCr, dCi, tr, ti);
    hr = _mm256_add_pd(tr, _mm256_set1_pd(orbit.SA_Br[n]));
    hi = _mm256_add_pd(ti, _mm256_set1_pd(orbit.SA_Bi[n]));

    // Horner step 3: h = h*δC + A
    cmul_avx2(hr, hi, dCr, dCi, tr, ti);
    hr = _mm256_add_pd(tr, _mm256_set1_pd(orbit.SA_Ar[n]));
    hi = _mm256_add_pd(ti, _mm256_set1_pd(orbit.SA_Ai[n]));

    // Final: δZ = h*δC
    cmul_avx2(hr, hi, dCr, dCi, dzr, dzi);

    // Check for NaN/inf in result - if any lane is bad, return false
    // A value is finite iff it equals itself and subtracting it from itself gives 0
    __m256d sum = _mm256_add_pd(_mm256_mul_pd(dzr, dzr), _mm256_mul_pd(dzi, dzi));
    __m256d is_finite = _mm256_cmp_pd(sum, sum, _CMP_EQ_OQ);  // NaN != NaN
    int finite_mask = _mm256_movemask_pd(is_finite);

    if (finite_mask != 0xF) {
        // At least one lane has NaN/inf - zero out bad lanes
        dzr = _mm256_and_pd(dzr, is_finite);
        dzi = _mm256_and_pd(dzi, is_finite);
        return false;
    }
    return true;
}
#endif

// Scalar SA evaluation using Horner's method (for non-AVX2 or scalar paths)
// Returns true if result is finite, false if NaN/inf (sets dz to 0 in that case)
static inline bool sa_eval_scalar(const ReferenceOrbit& orbit, int n,
                                   double dCr, double dCi,
                                   double& dzr, double& dzi) {
    double hr = orbit.SA_Dr[n], hi = orbit.SA_Di[n];
    double tr, ti;

    // Horner step 1: h = D*δC + C
    tr = hr * dCr - hi * dCi;
    ti = hr * dCi + hi * dCr;
    hr = tr + orbit.SA_Cr[n];
    hi = ti + orbit.SA_Ci[n];

    // Horner step 2: h = h*δC + B
    tr = hr * dCr - hi * dCi;
    ti = hr * dCi + hi * dCr;
    hr = tr + orbit.SA_Br[n];
    hi = ti + orbit.SA_Bi[n];

    // Horner step 3: h = h*δC + A
    tr = hr * dCr - hi * dCi;
    ti = hr * dCi + hi * dCr;
    hr = tr + orbit.SA_Ar[n];
    hi = ti + orbit.SA_Ai[n];

    // Final: δZ = h*δC
    dzr = hr * dCr - hi * dCi;
    dzi = hr * dCi + hi * dCr;

    // Check for NaN/inf
    if (!std::isfinite(dzr) || !std::isfinite(dzi)) {
        dzr = 0.0;
        dzi = 0.0;
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// SERIES APPROXIMATION (SA) EVALUATION
// ═══════════════════════════════════════════════════════════════════════════

// SA validity criterion: truncation error < tolerance * approximation value
// We use 4 terms: δZ ≈ A*δC + B*δC² + C*δC³ + D*δC⁴
// Error is dominated by E*δC⁵ (the next uncomputed term)
// Since we don't compute E, we use D as proxy for validity:
//   |D*δC⁴| < ε * |A*δC|
//   |D|*|δC|³ < ε * |A|
// Squaring both sides for numerical convenience:
//   |D|²*|δC|⁶ < ε²*|A|²
constexpr double SA_TOLERANCE = 0.001;  // Conservative tolerance for accuracy

// Find maximum iteration where SA approximation is valid for given δC
// Returns the iteration to skip to, and computes the initial δz at that point
inline int sa_find_skip_iteration(const ReferenceOrbit& orbit,
                                   double dCr, double dCi,
                                   double& dzr_out, double& dzi_out,
                                   int max_iter) {
    if (!orbit.sa_enabled || orbit.length < 2) {
        dzr_out = 0.0;
        dzi_out = 0.0;
        return 0;
    }

    double dC_norm = dCr * dCr + dCi * dCi;
    if (dC_norm == 0.0 || !std::isfinite(dC_norm)) {
        dzr_out = 0.0;
        dzi_out = 0.0;
        return 0;
    }

    // Binary search for maximum valid skip iteration
    // Note: The validity predicate may not be strictly monotonic near periodic
    // regions where |A_n|/|C_n| oscillates. We use binary search for speed but
    // validate the result afterward.
    int max_check = std::min(max_iter, orbit.length - 1);
    int best_skip = 0;

    // Lambda to check validity at iteration n
    // For 4-term SA: |D|² * |δC|⁶ < ε² * |A|²
    auto is_valid = [&](int n) -> bool {
        if (n < 1 || n > max_check) return false;
        double D_norm = orbit.SA_D_norm[n];
        double A_norm = orbit.SA_A_norm[n];
        if (!std::isfinite(D_norm) || !std::isfinite(A_norm) || A_norm == 0.0)
            return false;
        // Validity check: |D|² * |δC|⁶ < ε² * |A|²
        // |δC|⁶ = (|δC|²)³ = dC_norm³
        double dC_norm_cubed = dC_norm * dC_norm * dC_norm;
        double lhs = D_norm * dC_norm_cubed;
        double rhs = SA_TOLERANCE * SA_TOLERANCE * A_norm;
        return lhs < rhs;
    };

    // Binary search to find approximate maximum valid skip
    int lo = 1, hi = max_check;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (is_valid(mid)) {
            best_skip = mid;
            lo = mid + 1;  // Try to find larger valid skip
        } else {
            hi = mid - 1;  // Current is invalid, search lower
        }
    }

    // Validate best_skip - if non-monotonic, scan backward to find true maximum
    // This handles cases where validity oscillates near periodic regions
    if (best_skip > 0 && !is_valid(best_skip)) {
        // Binary search gave invalid result, scan backward
        while (best_skip > 0 && !is_valid(best_skip)) {
            best_skip--;
        }
    }

    // If we found a valid skip point, evaluate the SA polynomial using Horner form
    // δZ = A*δC + B*δC² + C*δC³ + D*δC⁴ = (((D*δC + C)*δC + B)*δC + A)*δC
    // Horner form reduces 7 complex muls to 4
    if (best_skip > 0) {
        double Ar = orbit.SA_Ar[best_skip];
        double Ai = orbit.SA_Ai[best_skip];
        double Br = orbit.SA_Br[best_skip];
        double Bi = orbit.SA_Bi[best_skip];
        double Cr = orbit.SA_Cr[best_skip];
        double Ci = orbit.SA_Ci[best_skip];
        double Dr = orbit.SA_Dr[best_skip];
        double Di = orbit.SA_Di[best_skip];

        // Horner: result = D
        double hr = Dr, hi = Di;

        // result = result * δC + C
        double tr = hr * dCr - hi * dCi;
        double ti = hr * dCi + hi * dCr;
        hr = tr + Cr;
        hi = ti + Ci;

        // result = result * δC + B
        tr = hr * dCr - hi * dCi;
        ti = hr * dCi + hi * dCr;
        hr = tr + Br;
        hi = ti + Bi;

        // result = result * δC + A
        tr = hr * dCr - hi * dCi;
        ti = hr * dCi + hi * dCr;
        hr = tr + Ar;
        hi = ti + Ai;

        // result = result * δC (final multiplication)
        dzr_out = hr * dCr - hi * dCi;
        dzi_out = hr * dCi + hi * dCr;

        // Guard against non-finite results from coefficient overflow
        if (!std::isfinite(dzr_out) || !std::isfinite(dzi_out)) {
            dzr_out = 0.0;
            dzi_out = 0.0;
            return 0;  // Fall back to no skip
        }
    } else {
        dzr_out = 0.0;
        dzi_out = 0.0;
    }

    return best_skip;
}

// ═══════════════════════════════════════════════════════════════════════════
// ANSI ESCAPE CODES
// ═══════════════════════════════════════════════════════════════════════════

#define ESC "\x1b"
#define CSI ESC "["

#define CLEAR_SCREEN    CSI "2J"
#define CURSOR_HOME     CSI "H"
#define CURSOR_HIDE     CSI "?25l"
#define CURSOR_SHOW     CSI "?25h"
#define ALT_BUFFER_ON   CSI "?1049h"
#define ALT_BUFFER_OFF  CSI "?1049l"
#define BOLD            CSI "1m"
#define RESET           CSI "0m"

// ═══════════════════════════════════════════════════════════════════════════
// COLOR SCHEMES
// ═══════════════════════════════════════════════════════════════════════════

struct RGB { uint8_t r, g, b; };

// Clamp and convert double [0,1] to uint8_t [0,255]
inline uint8_t clamp_to_u8(double v) {
    return static_cast<uint8_t>(std::clamp(v, 0.0, 1.0) * 255.0);
}

// Smooth color interpolation
inline RGB lerp_color(RGB a, RGB b, double t) {
    return {
        (uint8_t)(a.r + t * (b.r - a.r)),
        (uint8_t)(a.g + t * (b.g - a.g)),
        (uint8_t)(a.b + t * (b.b - a.b))
    };
}

// Color scheme definitions
namespace ColorSchemes {
    // 1. Electric Blue
    RGB electric_blue(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double h = t * 360.0;
        double s = 0.9, v = 0.9 + 0.1 * sin(t * 6.28);

        double c = v * s;
        double x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
        double m = v - c;

        double r, g, b;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        return {clamp_to_u8(r + m), clamp_to_u8(g + m), clamp_to_u8(b + m)};
    }

    // 2. Fire
    RGB fire(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double r = fmin(1.0, t * 3);
        double g = fmax(0.0, fmin(1.0, (t - 0.33) * 3));
        double b = fmax(0.0, (t - 0.67) * 3);
        return {clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b)};
    }

    // 3. Ocean
    RGB ocean(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double r = 0.1 + 0.2 * sin(t * 3.14159);
        double g = 0.3 + 0.4 * t;
        double b = 0.5 + 0.5 * t;
        return {clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b)};
    }

    // 4. Neon
    RGB neon(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double phase = t * 6.28318;
        double r = 0.5 + 0.5 * sin(phase);
        double g = 0.5 + 0.5 * sin(phase + 2.094);
        double b = 0.5 + 0.5 * sin(phase + 4.188);
        // Boost saturation
        double max_c = fmax(r, fmax(g, b));
        if (max_c > 0) { r /= max_c; g /= max_c; b /= max_c; }
        return {clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b)};
    }

    // 5. Grayscale
    RGB grayscale(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        uint8_t v = clamp_to_u8(t);
        return {v, v, v};
    }

    // 6. Plasma
    RGB plasma(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double r = sin(t * 3.14159 * 2) * 0.5 + 0.5;
        double g = sin(t * 3.14159 * 4 + 1) * 0.5 + 0.5;
        double b = sin(t * 3.14159 * 6 + 2) * 0.5 + 0.5;
        return {clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b)};
    }

    // 7. Ice
    RGB ice(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double r = 0.7 + 0.3 * t;
        double g = 0.85 + 0.15 * t;
        double b = 1.0;
        return {clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b)};
    }

    // 8. Toxic
    RGB toxic(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double r = 0.2 * sin(t * 6.28) + 0.2;
        double g = 0.8 + 0.2 * sin(t * 12.56);
        double b = 0.1 + 0.3 * t;
        return {clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b)};
    }

    // 9. Sunset
    RGB sunset(double t, double rotation) {
        t = fmod(t + rotation, 1.0);
        if (t < 0) t += 1.0;
        double r = 1.0;
        double g = 0.3 + 0.5 * (1 - t);
        double b = 0.1 + 0.4 * (1 - t) * (1 - t);
        return {clamp_to_u8(r), clamp_to_u8(g), clamp_to_u8(b)};
    }
}

typedef RGB (*ColorFunc)(double, double);
ColorFunc color_schemes[] = {
    ColorSchemes::electric_blue,
    ColorSchemes::fire,
    ColorSchemes::ocean,
    ColorSchemes::neon,
    ColorSchemes::grayscale,
    ColorSchemes::plasma,
    ColorSchemes::ice,
    ColorSchemes::toxic,
    ColorSchemes::sunset
};
const char* scheme_names[] = {
    "Electric", "Fire", "Ocean", "Neon", "Grayscale",
    "Plasma", "Ice", "Toxic", "Sunset"
};
const int NUM_SCHEMES = 9;

// ═══════════════════════════════════════════════════════════════════════════
// MANDELBROT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

struct MandelbrotState {
    double center_x = -0.5;
    double center_y = 0.0;
    double zoom = 1.0;
    double angle = 0.0;  // View rotation in radians
    int max_iter = 256;
    int color_scheme = 0;
    double color_rotation = 0.0;
#if defined(__APPLE__)
    bool zoom_mode = false;  // macOS: toggle arrow keys between pan and zoom/rotate
#endif

    int width = 80;
    int height = 24;

    std::vector<double> iterations;
    std::string output_buffer;

    std::atomic<bool> needs_redraw{true};
    std::atomic<bool> running{true};

    // Perturbation theory support
    DD center_x_dd{-0.5};          // High-precision center X
    DD center_y_dd{0.0};           // High-precision center Y
    bool use_perturbation = false; // Auto-enabled at deep zoom
    bool dd_authoritative = false; // True if DD was set via --pos with full precision
    ReferenceOrbit primary_orbit;  // Primary reference orbit
    std::vector<std::pair<int,int>> glitch_pixels;  // Pixels needing recompute

    // Pan offset accumulator for deep zoom navigation
    // At extreme zoom, adding small pan deltas to center_dd loses precision.
    // Instead, we accumulate pan offsets separately and apply to δC computation.
    // Stored in SCREEN coordinates - rotated to world coords when committed or displayed.
    DD pan_offset_x{0.0};
    DD pan_offset_y{0.0};

    // iTerm2 inline image mode - renders as actual pixels instead of block characters
    bool iterm_image_mode = false;
    int image_width = 640;   // Pixel width when in image mode
    int image_height = 400;  // Pixel height when in image mode
    std::vector<uint8_t> image_buffer;  // RGB buffer for image output

    // Series Approximation control
    bool disable_sa = false;  // Disable SA for benchmarking

    // Reference orbit cache - avoid recomputing when center hasn't changed
    bool orbit_cache_valid = false;
    DD cached_center_x{0.0};
    DD cached_center_y{0.0};
    int cached_max_iter = 0;
    bool cached_sa_enabled = true;

    // Sync DD and double centers (for display/compatibility)
    // Note: center_x/y are doubles, so we sum hi+lo for best approximation
    void sync_centers_to_dd() {
        center_x = center_x_dd.hi + center_x_dd.lo;
        center_y = center_y_dd.hi + center_y_dd.lo;
    }
    void sync_centers_from_double() {
        center_x_dd = DD(center_x);
        center_y_dd = DD(center_y);
        dd_authoritative = false;  // Double is now authoritative
    }

    // Commit accumulated pan offset to center coordinates
    // Pan offset is in SCREEN coords, must rotate to world coords before adding
    // Call on zoom change, mode transition, reset, etc.
    void commit_pan_offset() {
        if (pan_offset_x.hi != 0.0 || pan_offset_x.lo != 0.0 ||
            pan_offset_y.hi != 0.0 || pan_offset_y.lo != 0.0) {
            // Rotate screen coords to world coords
            double cos_a = cos(angle);
            double sin_a = sin(angle);
            double pan_sx = pan_offset_x.hi + pan_offset_x.lo;
            double pan_sy = pan_offset_y.hi + pan_offset_y.lo;
            double pan_world_x = pan_sx * cos_a - pan_sy * sin_a;
            double pan_world_y = pan_sx * sin_a + pan_sy * cos_a;
            center_x_dd = dd_add(center_x_dd, pan_world_x);
            center_y_dd = dd_add(center_y_dd, pan_world_y);
            pan_offset_x = DD(0.0);
            pan_offset_y = DD(0.0);
            sync_centers_to_dd();
        }
    }

    // Get effective center (center + pan_offset) for display
    // Pan offset is in screen coords, rotate to world coords
    DD effective_center_x() const {
        double cos_a = cos(angle);
        double sin_a = sin(angle);
        double pan_sx = pan_offset_x.hi + pan_offset_x.lo;
        double pan_sy = pan_offset_y.hi + pan_offset_y.lo;
        double pan_world_x = pan_sx * cos_a - pan_sy * sin_a;
        return dd_add(center_x_dd, pan_world_x);
    }
    DD effective_center_y() const {
        double cos_a = cos(angle);
        double sin_a = sin(angle);
        double pan_sx = pan_offset_x.hi + pan_offset_x.lo;
        double pan_sy = pan_offset_y.hi + pan_offset_y.lo;
        double pan_world_y = pan_sx * sin_a + pan_sy * cos_a;
        return dd_add(center_y_dd, pan_world_y);
    }
};

// Smooth iteration count for anti-aliased coloring
inline double smooth_iter(double zr, double zi, int iter, int max_iter) {
    if (iter >= max_iter) return -1.0;
    double log_zn = log(zr * zr + zi * zi) / 2.0;
    double nu = log(log_zn / log(2.0)) / log(2.0);
    return iter + 1 - nu;
}

#if USE_AVX2
// AVX2 optimized computation - processes 4 pixels at once
void compute_mandelbrot_avx2(MandelbrotState& state, int start_row, int end_row) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);

    __m256d four = _mm256_set1_pd(4.0);
    __m256d one = _mm256_set1_pd(1.0);

    for (int y = start_row; y < end_row; y++) {
        double dy = (y - state.height / 2.0) / state.height * scale;

        for (int x = 0; x < state.width; x += 4) {
            // Compute rotated coordinates for 4 pixels
            double dx0 = (x - state.width / 2.0) / state.width * scale * aspect;
            double dx1 = ((x + 1) - state.width / 2.0) / state.width * scale * aspect;
            double dx2 = ((x + 2) - state.width / 2.0) / state.width * scale * aspect;
            double dx3 = ((x + 3) - state.width / 2.0) / state.width * scale * aspect;

            __m256d cx = _mm256_set_pd(
                state.center_x + dx3 * cos_a - dy * sin_a,
                state.center_x + dx2 * cos_a - dy * sin_a,
                state.center_x + dx1 * cos_a - dy * sin_a,
                state.center_x + dx0 * cos_a - dy * sin_a
            );
            __m256d cy_v = _mm256_set_pd(
                state.center_y + dx3 * sin_a + dy * cos_a,
                state.center_y + dx2 * sin_a + dy * cos_a,
                state.center_y + dx1 * sin_a + dy * cos_a,
                state.center_y + dx0 * sin_a + dy * cos_a
            );

            __m256d zr = _mm256_setzero_pd();
            __m256d zi = _mm256_setzero_pd();
            __m256d iter = _mm256_setzero_pd();
            // Store zr/zi at escape for smooth coloring
            __m256d escaped_zr = _mm256_setzero_pd();
            __m256d escaped_zi = _mm256_setzero_pd();
            __m256d has_escaped = _mm256_setzero_pd();

            for (int i = 0; i < state.max_iter; i++) {
                __m256d zr2 = _mm256_mul_pd(zr, zr);
                __m256d zi2 = _mm256_mul_pd(zi, zi);
                __m256d mag2 = _mm256_add_pd(zr2, zi2);

                __m256d active = _mm256_cmp_pd(mag2, four, _CMP_LT_OQ);
                if (_mm256_testz_pd(active, active)) break;

                // Capture zr/zi at the moment of first escape
                __m256d newly_escaped = _mm256_andnot_pd(has_escaped,
                    _mm256_cmp_pd(mag2, four, _CMP_GE_OQ));
                escaped_zr = _mm256_blendv_pd(escaped_zr, zr, newly_escaped);
                escaped_zi = _mm256_blendv_pd(escaped_zi, zi, newly_escaped);
                has_escaped = _mm256_or_pd(has_escaped, newly_escaped);

                __m256d zri = _mm256_mul_pd(zr, zi);
                __m256d new_zr = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cx);
                __m256d new_zi = _mm256_add_pd(_mm256_add_pd(zri, zri), cy_v);

                // Only update zr/zi for active (non-escaped) lanes
                zr = _mm256_blendv_pd(zr, new_zr, active);
                zi = _mm256_blendv_pd(zi, new_zi, active);

                iter = _mm256_add_pd(iter, _mm256_and_pd(active, one));
            }

            // Use escaped values for lanes that escaped, final values for those that didn't
            __m256d final_zr = _mm256_blendv_pd(zr, escaped_zr, has_escaped);
            __m256d final_zi = _mm256_blendv_pd(zi, escaped_zi, has_escaped);

            double iter_arr[4], zr_arr[4], zi_arr[4];
            _mm256_storeu_pd(iter_arr, iter);
            _mm256_storeu_pd(zr_arr, final_zr);
            _mm256_storeu_pd(zi_arr, final_zi);

            for (int i = 0; i < 4 && x + i < state.width; i++) {
                int idx = y * state.width + x + i;
                state.iterations[idx] = smooth_iter(zr_arr[i], zi_arr[i],
                    (int)iter_arr[i], state.max_iter);
            }
        }
    }
}
#endif

// Scalar fallback
void compute_mandelbrot_scalar(MandelbrotState& state, int start_row, int end_row) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);

    for (int y = start_row; y < end_row; y++) {
        double dy = (y - state.height / 2.0) / state.height * scale;

        for (int x = 0; x < state.width; x++) {
            double dx = (x - state.width / 2.0) / state.width * scale * aspect;

            // Apply rotation
            double cx = state.center_x + dx * cos_a - dy * sin_a;
            double cy = state.center_y + dx * sin_a + dy * cos_a;

            double zr = 0, zi = 0;
            int iter = 0;

            while (zr * zr + zi * zi < 4.0 && iter < state.max_iter) {
                double tmp = zr * zr - zi * zi + cx;
                zi = 2 * zr * zi + cy;
                zr = tmp;
                iter++;
            }

            int idx = y * state.width + x;
            state.iterations[idx] = smooth_iter(zr, zi, iter, state.max_iter);
        }
    }
}

void compute_mandelbrot_threaded(MandelbrotState& state) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    std::vector<std::thread> threads;
    int rows_per_thread = state.height / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * rows_per_thread;
        int end = (t == num_threads - 1) ? state.height : start + rows_per_thread;

        threads.emplace_back([&state, start, end]() {
#if USE_AVX2
            compute_mandelbrot_avx2(state, start, end);
#else
            compute_mandelbrot_scalar(state, start, end);
#endif
        });
    }

    for (auto& t : threads) t.join();
}

// ═══════════════════════════════════════════════════════════════════════════
// PERTURBATION COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

// Check if perturbation mode is needed based on precision requirements
// Double precision has ~15-16 significant digits. Beyond that, we need perturbation.
inline bool needs_perturbation(const MandelbrotState& state) {
    double pixel_size = 3.0 / state.zoom / state.width;
    // Use machine epsilon scaled by the magnitude of coordinates
    // DBL_EPSILON ≈ 2.2e-16, but we use 1e-13 as a conservative threshold
    // to account for accumulated rounding errors in iteration
    double max_coord = std::max({std::abs(state.center_x), std::abs(state.center_y), 1.0});
    double precision_limit = 1e-13 * max_coord;
    return pixel_size < precision_limit;
}

// Scalar perturbation computation with optimizations:
// - Incremental dC generation (Optimization 2.1)
// - Per-pixel SA skip using precomputed thresholds (Optimization 1.1)
// Reference orbit is at center_dd. Pan offset is added to δC to shift the view.
// At extreme zoom, both pixel_offset and pan_offset are tiny values (~1e-38),
// so adding them in double precision preserves accuracy.
void compute_perturbation_scalar(MandelbrotState& state,
                                 const ReferenceOrbit& orbit,
                                 int start_row, int end_row,
                                 bool glitch_only = false) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);

    // Pan offset in SCREEN coordinates - collapse DD to double
    // This is safe because pan_offset and pixel_offset are both tiny at extreme zoom
    double pan_x = state.pan_offset_x.hi + state.pan_offset_x.lo;
    double pan_y = state.pan_offset_y.hi + state.pan_offset_y.lo;

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIMIZATION 2.1: Incremental dC generation
    // dC is affine in x: dC = base + x * step (avoids per-pixel recomputation)
    // ═══════════════════════════════════════════════════════════════════════
    double pixel_step = scale * aspect / state.width;  // Screen-space step per pixel
    double step_r = pixel_step * cos_a;  // Rotated step in real direction
    double step_i = pixel_step * sin_a;  // Rotated step in imag direction

    int max_ref_iter = orbit.length - 1;
    bool use_fast_sa = orbit.sa_thresholds_built && orbit.sa_enabled;

    for (int y = start_row; y < end_row; y++) {
        double dy = (y - state.height / 2.0) / state.height * scale;
        double total_y = dy + pan_y;  // Screen-space y with pan

        // Base dC for x=0 (rotated screen coords)
        double dx0_base = (-state.width / 2.0) / state.width * scale * aspect + pan_x;
        double base_r = dx0_base * cos_a - total_y * sin_a;
        double base_i = dx0_base * sin_a + total_y * cos_a;

        // Current dC (incremental)
        double dCr = base_r;
        double dCi = base_i;

        // Skip hint from previous pixel (for O(1) amortized SA lookup)
        int prev_skip = 0;

        for (int x = 0; x < state.width; x++) {
            int idx = y * state.width + x;

            // Skip if doing glitch-only pass and this isn't a glitch
            if (glitch_only && state.iterations[idx] != -2.0) {
                dCr += step_r;
                dCi += step_i;
                continue;
            }

            // ═══════════════════════════════════════════════════════════════
            // OPTIMIZATION 1.1: Fast SA skip using precomputed thresholds
            // With skip caching: use previous pixel's skip as hint for O(1) lookup
            // ═══════════════════════════════════════════════════════════════
            double dzr = 0.0, dzi = 0.0;
            int iter;

            if (use_fast_sa) {
                double dC_norm = dCr * dCr + dCi * dCi;
                // Use hint-based lookup for O(1) amortized performance
                iter = sa_max_skip_with_hint(orbit, dC_norm, state.max_iter, prev_skip);
                prev_skip = iter;  // Update hint for next pixel
                if (iter > 0) {
                    if (!sa_eval_scalar(orbit, iter, dCr, dCi, dzr, dzi)) {
                        // SA coefficients overflowed, fall back to no skip
                        iter = 0;
                    }
                }
            } else {
                iter = sa_find_skip_iteration(orbit, dCr, dCi, dzr, dzi, state.max_iter);
                prev_skip = iter;
            }

            bool escaped = false;
            bool glitched = false;
            double final_zr = 0, final_zi = 0;

            // Check if SA already found escape
            if (iter > 0) {
                double full_zr = orbit.Zr_sum[iter] + dzr;
                double full_zi = orbit.Zi_sum[iter] + dzi;
                double mag2 = full_zr * full_zr + full_zi * full_zi;
                if (mag2 > 4.0) {
                    escaped = true;
                    final_zr = full_zr;
                    final_zi = full_zi;
                }
            }

            while (!escaped && iter < state.max_iter && iter < max_ref_iter) {
                // Use precomputed sums for full DD precision
                double Zr = orbit.Zr_sum[iter];
                double Zi = orbit.Zi_sum[iter];

                // δz_{n+1} = (2·Z_n + δz_n)·δz_n + δC
                double two_Zr = 2.0 * Zr;
                double two_Zi = 2.0 * Zi;

                double temp_r = two_Zr + dzr;
                double temp_i = two_Zi + dzi;

                // Complex multiply: (temp_r + i*temp_i) * (dzr + i*dzi)
                double new_dzr = temp_r * dzr - temp_i * dzi + dCr;
                double new_dzi = temp_r * dzi + temp_i * dzr + dCi;

                dzr = new_dzr;
                dzi = new_dzi;
                iter++;

                // Full Z = Z_ref + δz (using precomputed sums)
                double full_zr = orbit.Zr_sum[iter] + dzr;
                double full_zi = orbit.Zi_sum[iter] + dzi;
                double mag2 = full_zr * full_zr + full_zi * full_zi;

                // Escape check
                if (mag2 > 4.0) {
                    escaped = true;
                    final_zr = full_zr;
                    final_zi = full_zi;
                    break;
                }

                // Adaptive glitch detection (skip early iters and near-escape)
                if (iter > GLITCH_MIN_ITER && mag2 < NEAR_ESCAPE_THRESHOLD) {
                    double dz_mag2 = dzr * dzr + dzi * dzi;
                    double Z_mag2 = orbit.Z_norm[iter] + GLITCH_FLOOR;

                    // Adaptive ratio based on escape status
                    double ratio_threshold = GLITCH_BASE_RATIO;
                    if (orbit.escape_iter >= 0 && iter > orbit.escape_iter - ESCAPE_WINDOW) {
                        ratio_threshold = GLITCH_ESCAPE_RATIO;  // Very lenient near escape
                    }
                    // Also scale with |Z| to be more lenient as |Z| grows
                    double adaptive_ratio = ratio_threshold * (1.0 + 0.1 * sqrt(Z_mag2));

                    if (dz_mag2 / Z_mag2 > adaptive_ratio) {
                        glitched = true;
                        break;
                    }
                }
            }

            // Handle reference escape fallback
            if (!escaped && !glitched && iter >= max_ref_iter && iter < state.max_iter) {
                // Reference escaped before pixel - fall back to scalar with DD precision
                double zr = orbit.Zr_sum[iter] + dzr;
                double zi = orbit.Zi_sum[iter] + dzi;

                // Compute pixel coordinate using DD arithmetic to preserve precision
                // Use effective center (center_dd + pan_offset) for correct position
                // Recompute dx/dy from x/y for DD precision in fallback path
                double dx_local = (x - state.width / 2.0) / state.width * scale * aspect;
                double dy_local = (y - state.height / 2.0) / state.height * scale;
                double pixel_offset_x = dx_local * cos_a - dy_local * sin_a;
                double pixel_offset_y = dx_local * sin_a + dy_local * cos_a;
                DD effective_x = dd_add(state.center_x_dd, state.pan_offset_x);
                DD effective_y = dd_add(state.center_y_dd, state.pan_offset_y);
                DD cx_dd = dd_add(effective_x, pixel_offset_x);
                DD cy_dd = dd_add(effective_y, pixel_offset_y);
                double cx = cx_dd.hi + cx_dd.lo;
                double cy = cy_dd.hi + cy_dd.lo;

                while (zr * zr + zi * zi < 4.0 && iter < state.max_iter) {
                    double tmp = zr * zr - zi * zi + cx;
                    zi = 2 * zr * zi + cy;
                    zr = tmp;
                    iter++;
                }

                if (iter < state.max_iter) {
                    escaped = true;
                    final_zr = zr;
                    final_zi = zi;
                }
            }

            // Store result
            if (glitched) {
                state.iterations[idx] = -2.0;  // Mark for glitch handling
            } else if (escaped) {
                state.iterations[idx] = smooth_iter(final_zr, final_zi, iter, state.max_iter);
            } else {
                state.iterations[idx] = -1.0;  // Bounded
            }

            // Increment dC for next pixel (Optimization 2.1)
            dCr += step_r;
            dCi += step_i;
        }
    }
}

#if USE_AVX2
// AVX2 perturbation computation with optimizations:
// - Incremental dC generation (Optimization 2.1)
// - Per-tile SA skip lookup (Optimization 1.1)
// - Vectorized SA evaluation (Optimization 1.2)
void compute_perturbation_avx2(MandelbrotState& state,
                               const ReferenceOrbit& orbit,
                               int start_row, int end_row) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);

    // Pan offset in screen coordinates
    double pan_x = state.pan_offset_x.hi + state.pan_offset_x.lo;
    double pan_y = state.pan_offset_y.hi + state.pan_offset_y.lo;

    // ═══════════════════════════════════════════════════════════════════════
    // OPTIMIZATION 2.1: Incremental dC generation
    // dC is affine in x: dC = base + x * step (avoids per-pixel recomputation)
    // ═══════════════════════════════════════════════════════════════════════
    double pixel_step = scale * aspect / state.width;  // Screen-space step per pixel
    double step_r = pixel_step * cos_a;  // Rotated step in real direction
    double step_i = pixel_step * sin_a;  // Rotated step in imag direction

    // For vectorized incremental: build [0,1,2,3] index vector
    __m256d idx4 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    __m256d step_r_vec = _mm256_set1_pd(step_r);
    __m256d step_i_vec = _mm256_set1_pd(step_i);
    __m256d step4_r = _mm256_set1_pd(4.0 * step_r);
    __m256d step4_i = _mm256_set1_pd(4.0 * step_i);

    __m256d four = _mm256_set1_pd(4.0);
    __m256d two = _mm256_set1_pd(2.0);
    __m256d one = _mm256_set1_pd(1.0);
    __m256d near_escape_thresh = _mm256_set1_pd(NEAR_ESCAPE_THRESHOLD);
    __m256d floor_val = _mm256_set1_pd(GLITCH_FLOOR);

    int max_ref_iter = orbit.length - 1;

    // Use original SA method - precomputed thresholds had accuracy issues
    bool use_fast_sa = orbit.sa_thresholds_built && orbit.sa_enabled;

    for (int y = start_row; y < end_row; y++) {
        double dy = (y - state.height / 2.0) / state.height * scale;
        double total_y = dy + pan_y;  // Screen-space y with pan

        // Base dC for x=0 (rotated screen coords)
        double dx0_base = (-state.width / 2.0) / state.width * scale * aspect + pan_x;
        double base_r = dx0_base * cos_a - total_y * sin_a;
        double base_i = dx0_base * sin_a + total_y * cos_a;

        // Initialize SIMD base vectors
        __m256d dC_base_r = _mm256_add_pd(_mm256_set1_pd(base_r), _mm256_mul_pd(idx4, step_r_vec));
        __m256d dC_base_i = _mm256_add_pd(_mm256_set1_pd(base_i), _mm256_mul_pd(idx4, step_i_vec));

        // Skip hint from previous tile (for O(1) amortized SA lookup)
        int prev_skip = 0;

        for (int x = 0; x < state.width; x += 4) {
            // Current dC values for 4 pixels (incremental)
            __m256d dC_r = dC_base_r;
            __m256d dC_i = dC_base_i;

            // Extract scalar values for SA lookup (needed for skip calculation)
            double dCr0, dCr1, dCr2, dCr3;
            double dCi0, dCi1, dCi2, dCi3;
            {
                alignas(32) double dCr_arr[4], dCi_arr[4];
                _mm256_store_pd(dCr_arr, dC_r);
                _mm256_store_pd(dCi_arr, dC_i);
                dCr0 = dCr_arr[0]; dCr1 = dCr_arr[1]; dCr2 = dCr_arr[2]; dCr3 = dCr_arr[3];
                dCi0 = dCi_arr[0]; dCi1 = dCi_arr[1]; dCi2 = dCi_arr[2]; dCi3 = dCi_arr[3];
            }

            // ═══════════════════════════════════════════════════════════════
            // OPTIMIZATION 1.1: Per-tile SA skip using precomputed thresholds
            // With skip caching: use previous tile's skip as hint for O(1) lookup
            // ═══════════════════════════════════════════════════════════════
            int min_skip = 0;
            __m256d dzr, dzi;

            if (use_fast_sa) {
                // Compute max |δC|² among the 4 pixels (worst case for tile)
                double dC_norm0 = dCr0*dCr0 + dCi0*dCi0;
                double dC_norm1 = dCr1*dCr1 + dCi1*dCi1;
                double dC_norm2 = dCr2*dCr2 + dCi2*dCi2;
                double dC_norm3 = dCr3*dCr3 + dCi3*dCi3;
                double max_dC_norm = std::max({dC_norm0, dC_norm1, dC_norm2, dC_norm3});

                // Use hint-based lookup for O(1) amortized performance
                min_skip = sa_max_skip_with_hint(orbit, max_dC_norm, state.max_iter, prev_skip);
                prev_skip = min_skip;  // Update hint for next tile

                // ═══════════════════════════════════════════════════════════
                // OPTIMIZATION 1.2: Vectorized SA evaluation with sa_eval4()
                // ═══════════════════════════════════════════════════════════
                if (min_skip > 0) {
                    if (!sa_eval4(orbit, min_skip, dC_r, dC_i, dzr, dzi)) {
                        // SA coefficients overflowed on at least one lane
                        // Bad lanes were zeroed, but we continue (they'll iterate from 0)
                    }
                } else {
                    dzr = _mm256_setzero_pd();
                    dzi = _mm256_setzero_pd();
                }
            } else {
                // Fallback to old per-pixel SA (when thresholds not built)
                double dzr_arr[4] = {0, 0, 0, 0};
                double dzi_arr[4] = {0, 0, 0, 0};
                int skip0 = sa_find_skip_iteration(orbit, dCr0, dCi0, dzr_arr[0], dzi_arr[0], state.max_iter);
                int skip1 = sa_find_skip_iteration(orbit, dCr1, dCi1, dzr_arr[1], dzi_arr[1], state.max_iter);
                int skip2 = sa_find_skip_iteration(orbit, dCr2, dCi2, dzr_arr[2], dzi_arr[2], state.max_iter);
                int skip3 = sa_find_skip_iteration(orbit, dCr3, dCi3, dzr_arr[3], dzi_arr[3], state.max_iter);
                min_skip = std::min({skip0, skip1, skip2, skip3});
                prev_skip = min_skip;  // Update hint for next tile

                if (min_skip > 0) {
                    sa_find_skip_iteration(orbit, dCr0, dCi0, dzr_arr[0], dzi_arr[0], min_skip);
                    sa_find_skip_iteration(orbit, dCr1, dCi1, dzr_arr[1], dzi_arr[1], min_skip);
                    sa_find_skip_iteration(orbit, dCr2, dCi2, dzr_arr[2], dzi_arr[2], min_skip);
                    sa_find_skip_iteration(orbit, dCr3, dCi3, dzr_arr[3], dzi_arr[3], min_skip);
                } else {
                    dzr_arr[0] = dzr_arr[1] = dzr_arr[2] = dzr_arr[3] = 0.0;
                    dzi_arr[0] = dzi_arr[1] = dzi_arr[2] = dzi_arr[3] = 0.0;
                }
                dzr = _mm256_set_pd(dzr_arr[3], dzr_arr[2], dzr_arr[1], dzr_arr[0]);
                dzi = _mm256_set_pd(dzi_arr[3], dzi_arr[2], dzi_arr[1], dzi_arr[0]);
            }

            // Increment base for next iteration (incremental dC)
            dC_base_r = _mm256_add_pd(dC_base_r, step4_r);
            dC_base_i = _mm256_add_pd(dC_base_i, step4_i);
            __m256d iter = _mm256_set1_pd((double)min_skip);
            __m256d active = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));  // All true

            // For smooth coloring
            __m256d escaped_zr = _mm256_setzero_pd();
            __m256d escaped_zi = _mm256_setzero_pd();
            __m256d has_escaped = _mm256_setzero_pd();
            __m256d is_glitched = _mm256_setzero_pd();

            // Check for early escape after SA skip (matches scalar path behavior)
            if (min_skip > 0) {
                __m256d Zr_skip = _mm256_set1_pd(orbit.Zr_sum[min_skip]);
                __m256d Zi_skip = _mm256_set1_pd(orbit.Zi_sum[min_skip]);
                __m256d full_zr = _mm256_add_pd(Zr_skip, dzr);
                __m256d full_zi = _mm256_add_pd(Zi_skip, dzi);
                __m256d mag2 = _mm256_add_pd(_mm256_mul_pd(full_zr, full_zr),
                                             _mm256_mul_pd(full_zi, full_zi));
                __m256d early_escape = _mm256_cmp_pd(mag2, four, _CMP_GT_OQ);
                has_escaped = _mm256_or_pd(has_escaped, early_escape);
                escaped_zr = _mm256_blendv_pd(escaped_zr, full_zr, early_escape);
                escaped_zi = _mm256_blendv_pd(escaped_zi, full_zi, early_escape);
                active = _mm256_andnot_pd(early_escape, active);
            }

            int max_n = std::min(state.max_iter, max_ref_iter);

            for (int n = min_skip; n < max_n; n++) {
                // Broadcast reference values (using precomputed sums for precision)
                __m256d Zr = _mm256_set1_pd(orbit.Zr_sum[n]);
                __m256d Zi = _mm256_set1_pd(orbit.Zi_sum[n]);
                __m256d Zr_next = _mm256_set1_pd(orbit.Zr_sum[n + 1]);
                __m256d Zi_next = _mm256_set1_pd(orbit.Zi_sum[n + 1]);
                __m256d Z_norm_next = _mm256_set1_pd(orbit.Z_norm[n + 1]);

                // δz_{n+1} = (2·Z_n + δz_n)·δz_n + δC
                __m256d two_Zr = _mm256_mul_pd(two, Zr);
                __m256d two_Zi = _mm256_mul_pd(two, Zi);

                __m256d temp_r = _mm256_add_pd(two_Zr, dzr);
                __m256d temp_i = _mm256_add_pd(two_Zi, dzi);

                // Complex multiply
                __m256d tr_dr = _mm256_mul_pd(temp_r, dzr);
                __m256d ti_di = _mm256_mul_pd(temp_i, dzi);
                __m256d tr_di = _mm256_mul_pd(temp_r, dzi);
                __m256d ti_dr = _mm256_mul_pd(temp_i, dzr);

                __m256d new_dzr = _mm256_add_pd(_mm256_sub_pd(tr_dr, ti_di), dC_r);
                __m256d new_dzi = _mm256_add_pd(_mm256_add_pd(tr_di, ti_dr), dC_i);

                // Update only active lanes
                dzr = _mm256_blendv_pd(dzr, new_dzr, active);
                dzi = _mm256_blendv_pd(dzi, new_dzi, active);
                iter = _mm256_add_pd(iter, _mm256_and_pd(active, one));

                // Full Z = Z_ref + δz
                __m256d full_zr = _mm256_add_pd(Zr_next, dzr);
                __m256d full_zi = _mm256_add_pd(Zi_next, dzi);
                __m256d mag2 = _mm256_add_pd(
                    _mm256_mul_pd(full_zr, full_zr),
                    _mm256_mul_pd(full_zi, full_zi)
                );

                // Escape check
                __m256d escaped = _mm256_cmp_pd(mag2, four, _CMP_GT_OQ);
                __m256d newly_escaped = _mm256_andnot_pd(has_escaped, escaped);
                escaped_zr = _mm256_blendv_pd(escaped_zr, full_zr, newly_escaped);
                escaped_zi = _mm256_blendv_pd(escaped_zi, full_zi, newly_escaped);
                has_escaped = _mm256_or_pd(has_escaped, escaped);

                // Adaptive glitch detection (skip early iters and near-escape)
                // Use n+1 for indexing to match scalar path (after iter++)
                if (n >= GLITCH_MIN_ITER) {
                    // Only check glitch if not near escape
                    __m256d not_near_escape = _mm256_cmp_pd(mag2, near_escape_thresh, _CMP_LT_OQ);

                    __m256d dz_mag2 = _mm256_add_pd(
                        _mm256_mul_pd(dzr, dzr),
                        _mm256_mul_pd(dzi, dzi)
                    );
                    __m256d Z_mag2 = _mm256_add_pd(Z_norm_next, floor_val);

                    // Compute adaptive ratio threshold
                    // base_ratio * (1 + 0.1 * sqrt(Z_mag2))
                    double base_ratio = GLITCH_BASE_RATIO;
                    if (orbit.escape_iter >= 0 && n + 1 > orbit.escape_iter - ESCAPE_WINDOW) {
                        base_ratio = GLITCH_ESCAPE_RATIO;
                    }
                    __m256d ratio_base = _mm256_set1_pd(base_ratio);
                    __m256d sqrt_Z = _mm256_sqrt_pd(Z_mag2);
                    __m256d scale_factor = _mm256_add_pd(one, _mm256_mul_pd(_mm256_set1_pd(0.1), sqrt_Z));
                    __m256d adaptive_ratio = _mm256_mul_pd(ratio_base, scale_factor);

                    // threshold = Z_mag2 * adaptive_ratio
                    __m256d threshold = _mm256_mul_pd(Z_mag2, adaptive_ratio);
                    __m256d glitch = _mm256_cmp_pd(dz_mag2, threshold, _CMP_GT_OQ);

                    // Only mark as glitched if not near escape
                    glitch = _mm256_and_pd(glitch, not_near_escape);

                    __m256d newly_glitched = _mm256_andnot_pd(
                        _mm256_or_pd(has_escaped, is_glitched),
                        glitch
                    );
                    is_glitched = _mm256_or_pd(is_glitched, newly_glitched);
                }

                // Update active mask
                active = _mm256_andnot_pd(
                    _mm256_or_pd(has_escaped, is_glitched),
                    active
                );

                if (_mm256_testz_pd(active, active)) break;
            }

            // Extract and store results
            double iter_arr[4], zr_arr[4], zi_arr[4];
            double escaped_arr[4], glitched_arr[4];
            double dzr_out[4], dzi_out[4];
            _mm256_storeu_pd(iter_arr, iter);
            _mm256_storeu_pd(zr_arr, escaped_zr);
            _mm256_storeu_pd(zi_arr, escaped_zi);
            _mm256_storeu_pd(escaped_arr, has_escaped);
            _mm256_storeu_pd(glitched_arr, is_glitched);
            _mm256_storeu_pd(dzr_out, dzr);
            _mm256_storeu_pd(dzi_out, dzi);

            for (int i = 0; i < 4 && x + i < state.width; i++) {
                int idx = y * state.width + x + i;
                if (glitched_arr[i] != 0.0) {
                    state.iterations[idx] = -2.0;  // Glitch marker
                } else if (escaped_arr[i] != 0.0) {
                    state.iterations[idx] = smooth_iter(
                        zr_arr[i], zi_arr[i],
                        (int)iter_arr[i], state.max_iter
                    );
                } else {
                    // Reference-escape fallback: continue scalar from end of orbit
                    int cur_iter = (int)iter_arr[i];
                    if (cur_iter >= max_ref_iter && cur_iter < state.max_iter) {
                        // Get current Z = Z_ref[end] + δz (using precomputed sums)
                        double zr = orbit.Zr_sum[max_ref_iter] + dzr_out[i];
                        double zi = orbit.Zi_sum[max_ref_iter] + dzi_out[i];

                        // Compute pixel offset (without pan_offset, we add it via DD below)
                        double dxi = ((x + i) - state.width / 2.0) / state.width * scale * aspect;
                        double pixel_offset_x = dxi * cos_a - dy * sin_a;
                        double pixel_offset_y = dxi * sin_a + dy * cos_a;

                        // Get C using DD arithmetic with effective center (includes pan_offset)
                        DD effective_x = dd_add(state.center_x_dd, state.pan_offset_x);
                        DD effective_y = dd_add(state.center_y_dd, state.pan_offset_y);
                        DD cx_dd = dd_add(effective_x, pixel_offset_x);
                        DD cy_dd = dd_add(effective_y, pixel_offset_y);
                        double cx = cx_dd.hi + cx_dd.lo;
                        double cy = cy_dd.hi + cy_dd.lo;

                        // Continue standard iteration
                        while (zr * zr + zi * zi < 4.0 && cur_iter < state.max_iter) {
                            double tmp = zr * zr - zi * zi + cx;
                            zi = 2 * zr * zi + cy;
                            zr = tmp;
                            cur_iter++;
                        }

                        if (cur_iter < state.max_iter) {
                            state.iterations[idx] = smooth_iter(zr, zi, cur_iter, state.max_iter);
                        } else {
                            state.iterations[idx] = -1.0;  // Bounded
                        }
                    } else {
                        state.iterations[idx] = -1.0;  // Bounded
                    }
                }
            }
        }
    }
}
#endif

// Threaded perturbation computation
void compute_perturbation_threaded(MandelbrotState& state, const ReferenceOrbit& orbit) {
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;

    std::vector<std::thread> threads;
    int rows_per_thread = state.height / num_threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * rows_per_thread;
        int end = (t == num_threads - 1) ? state.height : start + rows_per_thread;

        threads.emplace_back([&state, &orbit, start, end]() {
#if USE_AVX2
            compute_perturbation_avx2(state, orbit, start, end);
#else
            compute_perturbation_scalar(state, orbit, start, end);
#endif
        });
    }

    for (auto& t : threads) t.join();
}

// Collect glitched pixels after computation
void collect_glitch_pixels(MandelbrotState& state) {
    state.glitch_pixels.clear();
    for (int y = 0; y < state.height; y++) {
        for (int x = 0; x < state.width; x++) {
            if (state.iterations[y * state.width + x] == -2.0) {
                state.glitch_pixels.push_back({x, y});
            }
        }
    }
}

// Find center of glitch cluster
std::pair<int, int> find_glitch_center(const MandelbrotState& state) {
    if (state.glitch_pixels.empty()) return {state.width / 2, state.height / 2};

    long sum_x = 0, sum_y = 0;
    for (const auto& p : state.glitch_pixels) {
        sum_x += p.first;
        sum_y += p.second;
    }
    return {
        (int)(sum_x / state.glitch_pixels.size()),
        (int)(sum_y / state.glitch_pixels.size())
    };
}

// Convert pixel to DD coordinates (with rotation and pan_offset)
// Uses effective center (center_dd + pan_offset) to get actual pixel coordinate
DD pixel_to_dd_x(int px, int py, const MandelbrotState& state) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);
    double dx = (px - state.width / 2.0) / state.width * scale * aspect;
    double dy = (py - state.height / 2.0) / state.height * scale;
    double offset = dx * cos_a - dy * sin_a;
    // Use effective center (includes pan_offset)
    DD effective_x = dd_add(state.center_x_dd, state.pan_offset_x);
    return dd_add(effective_x, offset);
}

DD pixel_to_dd_y(int px, int py, const MandelbrotState& state) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);
    double dx = (px - state.width / 2.0) / state.width * scale * aspect;
    double dy = (py - state.height / 2.0) / state.height * scale;
    double offset = dx * sin_a + dy * cos_a;
    // Use effective center (includes pan_offset)
    DD effective_y = dd_add(state.center_y_dd, state.pan_offset_y);
    return dd_add(effective_y, offset);
}

// Direct DD iteration for fallback (guaranteed correct but slow)
double compute_direct_dd(int px, int py, const MandelbrotState& state) {
    DD cx = pixel_to_dd_x(px, py, state);
    DD cy = pixel_to_dd_y(px, py, state);

    DDComplex C(cx, cy);
    DDComplex Z(0.0, 0.0);

    for (int n = 0; n < state.max_iter; n++) {
        Z = Z.square() + C;
        // Use full DD precision for escape check (critical for deep zoom correctness)
        double mag2 = Z.norm_full();

        if (mag2 > 4.0) {
            return smooth_iter(Z.re.hi, Z.im.hi, n + 1, state.max_iter);
        }
    }
    return -1.0;  // Bounded
}

// Handle glitches with additional references and DD fallback
void handle_glitches(MandelbrotState& state) {
    collect_glitch_pixels(state);

    // Check glitch rate - if too high, reference is bad
    double glitch_rate = (double)state.glitch_pixels.size() /
                         (state.width * state.height);

    // If glitch rate is very high, the primary reference is unsuitable
    // In this case, use DD fallback for all glitches directly
    if (glitch_rate > GLITCH_RATE_LIMIT) {
        // Fall back to DD for all glitched pixels
        for (const auto& [px, py] : state.glitch_pixels) {
            int idx = py * state.width + px;
            state.iterations[idx] = compute_direct_dd(px, py, state);
        }
        state.glitch_pixels.clear();
        return;
    }

    std::vector<ReferenceOrbit> additional_orbits;
    int passes = 0;

    while (!state.glitch_pixels.empty() && passes < MAX_REFERENCES - 1) {
        passes++;

        // Find glitch cluster center
        auto [gx, gy] = find_glitch_center(state);

        // Compute new reference at glitch center
        ReferenceOrbit new_orbit;
        DD new_center_x = pixel_to_dd_x(gx, gy, state);
        DD new_center_y = pixel_to_dd_y(gx, gy, state);
        compute_reference_orbit(new_orbit, new_center_x, new_center_y, state.max_iter, !state.disable_sa);

        // Recompute only glitched pixels with new reference
        double aspect = (double)state.width / (state.height * 2.0);
        double scale = 3.0 / state.zoom;
        double cos_a = cos(state.angle);
        double sin_a = sin(state.angle);

        for (const auto& [px, py] : state.glitch_pixels) {
            int idx = py * state.width + px;

            double dx = (px - state.width / 2.0) / state.width * scale * aspect;
            double dy = (py - state.height / 2.0) / state.height * scale;
            double offset_x = dx * cos_a - dy * sin_a;
            double offset_y = dx * sin_a + dy * cos_a;

            // Use effective center (center_dd + pan_offset) to preserve precision
            DD effective_x = dd_add(state.center_x_dd, state.pan_offset_x);
            DD effective_y = dd_add(state.center_y_dd, state.pan_offset_y);
            DD ppx = dd_add(effective_x, offset_x);
            DD ppy = dd_add(effective_y, offset_y);
            DD dCr_dd = dd_sub(ppx, new_orbit.center_re);
            DD dCi_dd = dd_sub(ppy, new_orbit.center_im);

            double dCr = dCr_dd.hi + dCr_dd.lo;
            double dCi = dCi_dd.hi + dCi_dd.lo;

            double dzr = 0.0, dzi = 0.0;
            int iter = 0;
            int max_ref = new_orbit.length - 1;
            bool escaped = false;
            double final_zr = 0, final_zi = 0;

            while (iter < state.max_iter && iter < max_ref) {
                // Use precomputed sums for full DD precision
                double Zr = new_orbit.Zr_sum[iter];
                double Zi = new_orbit.Zi_sum[iter];

                double temp_r = 2.0 * Zr + dzr;
                double temp_i = 2.0 * Zi + dzi;

                double new_dzr = temp_r * dzr - temp_i * dzi + dCr;
                double new_dzi = temp_r * dzi + temp_i * dzr + dCi;

                dzr = new_dzr;
                dzi = new_dzi;
                iter++;

                double full_zr = new_orbit.Zr_sum[iter] + dzr;
                double full_zi = new_orbit.Zi_sum[iter] + dzi;
                double mag2 = full_zr * full_zr + full_zi * full_zi;

                if (mag2 > 4.0) {
                    escaped = true;
                    final_zr = full_zr;
                    final_zi = full_zi;
                    break;
                }

                // Adaptive glitch check (may still glitch with new reference)
                if (iter > GLITCH_MIN_ITER && mag2 < NEAR_ESCAPE_THRESHOLD) {
                    double dz_mag2 = dzr * dzr + dzi * dzi;
                    double Z_mag2 = new_orbit.Z_norm[iter] + GLITCH_FLOOR;

                    double ratio_threshold = GLITCH_BASE_RATIO;
                    if (new_orbit.escape_iter >= 0 && iter > new_orbit.escape_iter - ESCAPE_WINDOW) {
                        ratio_threshold = GLITCH_ESCAPE_RATIO;
                    }
                    double adaptive_ratio = ratio_threshold * (1.0 + 0.1 * sqrt(Z_mag2));

                    if (dz_mag2 / Z_mag2 > adaptive_ratio) {
                        break;  // Still glitched, will be collected again
                    }
                }
            }

            // Handle reference-escape fallback (critical fix from Codex review)
            if (!escaped && iter >= max_ref && iter < state.max_iter) {
                // Reference orbit ran out - continue with direct iteration
                double zr = new_orbit.Zr_sum[max_ref] + dzr;
                double zi = new_orbit.Zi_sum[max_ref] + dzi;

                // Compute C using DD arithmetic with effective center (includes pan_offset)
                DD eff_x = dd_add(state.center_x_dd, state.pan_offset_x);
                DD eff_y = dd_add(state.center_y_dd, state.pan_offset_y);
                DD ppx = dd_add(eff_x, offset_x);
                DD ppy = dd_add(eff_y, offset_y);
                double cx = ppx.hi + ppx.lo;
                double cy = ppy.hi + ppy.lo;

                while (zr * zr + zi * zi < 4.0 && iter < state.max_iter) {
                    double tmp = zr * zr - zi * zi + cx;
                    zi = 2 * zr * zi + cy;
                    zr = tmp;
                    iter++;
                }

                if (iter < state.max_iter) {
                    escaped = true;
                    final_zr = zr;
                    final_zi = zi;
                }
            }

            if (escaped) {
                state.iterations[idx] = smooth_iter(final_zr, final_zi, iter, state.max_iter);
            } else if (iter >= state.max_iter) {
                state.iterations[idx] = -1.0;  // Truly bounded
            }
            // else: still -2.0 (glitched), will be collected again
        }

        additional_orbits.push_back(std::move(new_orbit));
        collect_glitch_pixels(state);
    }

    // Any remaining glitches use DD fallback (guaranteed correct)
    for (const auto& [px, py] : state.glitch_pixels) {
        int idx = py * state.width + px;
        state.iterations[idx] = compute_direct_dd(px, py, state);
    }
    state.glitch_pixels.clear();
}

// Unified computation dispatcher
void compute_mandelbrot_unified(MandelbrotState& state) {
    if (!needs_perturbation(state)) {
        // Standard double-precision path
        // Commit any accumulated pan_offset before leaving perturbation mode
        if (state.use_perturbation) {
            state.commit_pan_offset();
        }
        state.use_perturbation = false;
        compute_mandelbrot_threaded(state);
    } else {
        // Perturbation path
        bool was_perturbation = state.use_perturbation;
        state.use_perturbation = true;

        // Sync DD centers from double on mode transition
        // This ensures DD centers match double centers if user panned/zoomed
        // before entering perturbation mode
        // BUT: skip if DD is authoritative (position was specified with full DD precision)
        if (!was_perturbation && !state.dd_authoritative) {
            state.sync_centers_from_double();
        }

        // Auto-scale iterations for deep zoom (deep areas need more iterations)
        // This applies even in manual mode, not just auto-explore
        int min_iter_for_zoom = 256 + (int)(log10(std::max(1.0, state.zoom)) * 64);
        int effective_max_iter = std::max(state.max_iter, std::min(4096, min_iter_for_zoom));

        // Check if we can reuse the cached reference orbit
        // Cache is valid if center, max_iter, and SA settings match
        bool cache_hit = state.orbit_cache_valid &&
                         state.cached_center_x.hi == state.center_x_dd.hi &&
                         state.cached_center_x.lo == state.center_x_dd.lo &&
                         state.cached_center_y.hi == state.center_y_dd.hi &&
                         state.cached_center_y.lo == state.center_y_dd.lo &&
                         state.cached_max_iter >= effective_max_iter &&
                         state.cached_sa_enabled == !state.disable_sa;

        if (!cache_hit) {
            // Compute reference orbit at center_dd (not effective_center)
            // Pan offset is added to δC in perturbation loop - this allows panning without
            // recomputing reference orbit, and works because both pixel_offset and pan_offset
            // are tiny values at extreme zoom (same order of magnitude)
            compute_reference_orbit(state.primary_orbit,
                                   state.center_x_dd, state.center_y_dd,
                                   effective_max_iter,
                                   !state.disable_sa);

            // Build SA thresholds for fast per-tile skip lookup (Optimization 1.1)
            if (!state.disable_sa) {
                build_sa_thresholds(state.primary_orbit);
            }

            // Update cache metadata
            state.orbit_cache_valid = true;
            state.cached_center_x = state.center_x_dd;
            state.cached_center_y = state.center_y_dd;
            state.cached_max_iter = effective_max_iter;
            state.cached_sa_enabled = !state.disable_sa;
        }
        // else: reuse state.primary_orbit from cache

        // Temporarily use scaled iterations for perturbation
        int saved_max_iter = state.max_iter;
        state.max_iter = effective_max_iter;

        // Compute perturbation for all pixels
        compute_perturbation_threaded(state, state.primary_orbit);

        // Handle glitches with additional references
        handle_glitches(state);

        // Restore original max_iter (display shows user's setting)
        state.max_iter = saved_max_iter;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RENDERING
// ═══════════════════════════════════════════════════════════════════════════

// Block characters for half-height pixels
const char* BLOCKS[] = {" ", "▄", "▀", "█"};

// ═══════════════════════════════════════════════════════════════════════════
// iTerm2 INLINE IMAGE SUPPORT
// ═══════════════════════════════════════════════════════════════════════════

// Base64 encoding for iTerm2 inline images
static const char base64_chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode(const uint8_t* data, size_t len) {
    std::string result;
    result.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = data[i] << 16;
        if (i + 1 < len) n |= data[i + 1] << 8;
        if (i + 2 < len) n |= data[i + 2];

        result += base64_chars[(n >> 18) & 0x3F];
        result += base64_chars[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? base64_chars[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? base64_chars[n & 0x3F] : '=';
    }
    return result;
}

// Check if running in iTerm2 (works even through tmux)
bool detect_iterm2() {
    const char* lc_terminal = getenv("LC_TERMINAL");
    if (lc_terminal && strcmp(lc_terminal, "iTerm2") == 0) return true;
    const char* iterm_session = getenv("ITERM_SESSION_ID");
    if (iterm_session && strlen(iterm_session) > 0) return true;
    return false;
}

// Compute mandelbrot at image resolution (higher than terminal cells)
void compute_mandelbrot_image(MandelbrotState& state) {
    // Save original dimensions
    int orig_width = state.width;
    int orig_height = state.height;
    auto orig_iterations = std::move(state.iterations);

    // Set to image dimensions
    state.width = state.image_width;
    state.height = state.image_height;
    state.iterations.resize(state.width * state.height);

    // Compute using existing infrastructure
    compute_mandelbrot_unified(state);

    // Generate RGB buffer
    ColorFunc get_color = color_schemes[state.color_scheme];
    state.image_buffer.resize(state.width * state.height * 3);

    for (int y = 0; y < state.height; y++) {
        for (int x = 0; x < state.width; x++) {
            double iter = state.iterations[y * state.width + x];
            RGB color;
            if (iter < 0) {
                color = {0, 0, 0};
            } else {
                double t = fmod(iter / 64.0, 1.0);
                color = get_color(t, state.color_rotation);
            }
            size_t idx = (y * state.width + x) * 3;
            state.image_buffer[idx] = color.r;
            state.image_buffer[idx + 1] = color.g;
            state.image_buffer[idx + 2] = color.b;
        }
    }

    // Restore original dimensions for terminal rendering fallback
    state.width = orig_width;
    state.height = orig_height;
    state.iterations = std::move(orig_iterations);
}

// Create PPM image data (simple format, no library needed)
std::vector<uint8_t> create_ppm(const uint8_t* rgb, int width, int height) {
    std::vector<uint8_t> ppm;
    char header[64];
    int hlen = snprintf(header, sizeof(header), "P6\n%d %d\n255\n", width, height);
    ppm.insert(ppm.end(), header, header + hlen);
    ppm.insert(ppm.end(), rgb, rgb + width * height * 3);
    return ppm;
}

// Output image using iTerm2 inline image protocol
// Protocol: ESC ] 1337 ; File = [args] : base64_data BEL
void render_iterm2_image(MandelbrotState& state) {
    // Compute at image resolution
    compute_mandelbrot_image(state);

    // Create PPM image
    auto ppm = create_ppm(state.image_buffer.data(), state.image_width, state.image_height);

    // Base64 encode
    std::string b64 = base64_encode(ppm.data(), ppm.size());

    // Output using iTerm2 protocol
    // Move cursor home, then output image
    printf(CURSOR_HOME);
    printf("\033]1337;File=inline=1;width=auto;height=auto;preserveAspectRatio=1:");
    printf("%s", b64.c_str());
    printf("\007");  // BEL character

    // Status bar below image (move down based on terminal height)
    char status[640];
    const char* mode_str = state.use_perturbation ? "PERTURB" : "DOUBLE";
    double angle_deg = state.angle * 180.0 / M_PI;
    DD eff_x = state.effective_center_x();
    DD eff_y = state.effective_center_y();
    double display_x = eff_x.hi + eff_x.lo;
    double display_y = eff_y.hi + eff_y.lo;
    snprintf(status, sizeof(status),
        "\n" BOLD " ═══ MANDELBROT [%dx%d] ═══ " RESET
        " │ Pos: %.10g%+.10gi │ Zoom: %.2e │ Angle: %.1f° │ [%s]",
        state.image_width, state.image_height,
        display_x, display_y, state.zoom, angle_deg, mode_str);
    printf("%s", status);
#if defined(__APPLE__)
    // macOS: Show PAN/ZOOM mode indicator
    printf(CSI "K" "\x1b[%dG%s", std::max(1, state.width - 3), state.zoom_mode ? "ZOOM" : "PAN ");
#endif
    fflush(stdout);
}

void render_frame(MandelbrotState& state) {
    // Use iTerm2 inline image mode if enabled
    if (state.iterm_image_mode) {
        render_iterm2_image(state);
        return;
    }

    ColorFunc get_color = color_schemes[state.color_scheme];

    std::string& out = state.output_buffer;
    out.clear();
    out.reserve(state.width * state.height * 30);

    out += CURSOR_HOME;

    // Render two rows at a time using half-block characters
    for (int y = 0; y < state.height - 1; y += 2) {
        for (int x = 0; x < state.width; x++) {
            double iter_top = state.iterations[y * state.width + x];
            double iter_bot = state.iterations[(y + 1) * state.width + x];

            RGB top_color, bot_color;

            if (iter_top < 0) {
                top_color = {0, 0, 0};
            } else {
                double t = fmod(iter_top / 64.0, 1.0);
                top_color = get_color(t, state.color_rotation);
            }

            if (iter_bot < 0) {
                bot_color = {0, 0, 0};
            } else {
                double t = fmod(iter_bot / 64.0, 1.0);
                bot_color = get_color(t, state.color_rotation);
            }

            // Use ▄ with foreground=bottom, background=top
            char buf[64];
            snprintf(buf, sizeof(buf), "\x1b[38;2;%d;%d;%d;48;2;%d;%d;%dm▄",
                bot_color.r, bot_color.g, bot_color.b,
                top_color.r, top_color.g, top_color.b);
            out += buf;
        }
        out += RESET "\n";
    }

    // Status bar - show effective position (includes pan_offset)
    char status[640];
    const char* mode_str = state.use_perturbation ? "PERTURB" : "DOUBLE";
    double angle_deg = state.angle * 180.0 / M_PI;
    // Use effective center for display (center + pan_offset)
    DD eff_x = state.effective_center_x();
    DD eff_y = state.effective_center_y();
    double display_x = eff_x.hi + eff_x.lo;
    double display_y = eff_y.hi + eff_y.lo;
    snprintf(status, sizeof(status),
        BOLD " ═══ MANDELBROT EXPLORER ═══ " RESET
        " │ Pos: %.10g%+.10gi │ Zoom: %.2e │ Angle: %.1f° │ Iter: %d │ [%s] │ %s",
        display_x, display_y, state.zoom, angle_deg, state.max_iter,
        scheme_names[state.color_scheme], mode_str);
    out += status;
#if defined(__APPLE__)
    // macOS: Show PAN/ZOOM mode indicator
    char nav_buf[32];
    snprintf(nav_buf, sizeof(nav_buf), CSI "K" "\x1b[%dG%s",
             std::max(1, state.width - 3), state.zoom_mode ? "ZOOM" : "PAN ");
    out += nav_buf;
#endif

    printf("%s", out.c_str());
    fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════════════
// INPUT HANDLING
// ═══════════════════════════════════════════════════════════════════════════

static struct termios orig_termios;
static bool terminal_raw = false;

void restore_terminal() {
    if (terminal_raw) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
        terminal_raw = false;
    }
    printf(CURSOR_SHOW ALT_BUFFER_OFF);
    fflush(stdout);
}

void setup_terminal() {
    // Only set up terminal if stdin is a TTY
    if (isatty(STDIN_FILENO)) {
        tcgetattr(STDIN_FILENO, &orig_termios);
        atexit(restore_terminal);

        struct termios raw = orig_termios;
        raw.c_lflag &= ~(ECHO | ICANON);
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 1;
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        terminal_raw = true;
    }

    printf(ALT_BUFFER_ON CURSOR_HIDE CLEAR_SCREEN);
    fflush(stdout);
}

void handle_resize(MandelbrotState& state) {
    struct winsize ws;
    if (isatty(STDOUT_FILENO) && ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        state.width = ws.ws_col;
        state.height = (ws.ws_row - 1) * 2;  // -1 for status bar, *2 for half-blocks
    }
    // Ensure iterations vector is properly sized
    size_t required_size = state.width * state.height;
    if (state.iterations.size() != required_size) {
        state.iterations.resize(required_size);
    }
    state.needs_redraw = true;
}

static MandelbrotState* g_state = nullptr;
static volatile sig_atomic_t resize_pending = 0;
static volatile sig_atomic_t sigint_pending = 0;

void sigwinch_handler(int) {
    resize_pending = 1;
}

void sigint_handler(int) {
    sigint_pending = 1;
}

enum Key {
    KEY_NONE = 0,
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_SHIFT_UP, KEY_SHIFT_DOWN, KEY_SHIFT_LEFT, KEY_SHIFT_RIGHT,
    KEY_Q, KEY_R, KEY_ESC,
    KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9,
    KEY_PLUS, KEY_MINUS,
    KEY_C, KEY_V, KEY_I
};

Key read_key() {
    char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return KEY_NONE;

    if (c == 'q' || c == 'Q') return KEY_Q;
    if (c == 'r' || c == 'R') return KEY_R;
    if (c == 'c' || c == 'C') return KEY_C;
    if (c == 'v' || c == 'V') return KEY_V;
#if defined(__APPLE__)
    if (c == 'z' || c == 'Z') {
        // macOS: Toggle arrow key mode between pan and zoom/rotate
        if (g_state) {
            g_state->zoom_mode = !g_state->zoom_mode;
            g_state->needs_redraw = true;
        }
        return KEY_NONE;
    }
#endif
    if (c == 'i' || c == 'I') return KEY_I;
    if (c == '+' || c == '=') return KEY_PLUS;
    if (c == '-' || c == '_') return KEY_MINUS;
    if (c >= '1' && c <= '9') return (Key)(KEY_1 + (c - '1'));
    if (c == 27) {  // ESC or escape sequence
        char seq[5];
        if (read(STDIN_FILENO, &seq[0], 1) != 1) return KEY_ESC;
        if (read(STDIN_FILENO, &seq[1], 1) != 1) return KEY_ESC;

        if (seq[0] == '[') {
            // Check for extended sequences (shift+arrow)
            if (seq[1] == '1') {
                if (read(STDIN_FILENO, &seq[2], 1) == 1 && seq[2] == ';') {
                    if (read(STDIN_FILENO, &seq[3], 1) == 1 && seq[3] == '2') {
                        if (read(STDIN_FILENO, &seq[4], 1) == 1) {
                            switch (seq[4]) {
                                case 'A': return KEY_SHIFT_UP;
                                case 'B': return KEY_SHIFT_DOWN;
                                case 'C': return KEY_SHIFT_RIGHT;
                                case 'D': return KEY_SHIFT_LEFT;
                            }
                        }
                    }
                }
            }
            // Regular arrows
            switch (seq[1]) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
        return KEY_ESC;
    }

    return KEY_NONE;
}

void handle_input(MandelbrotState& state) {
    Key key = read_key();
    if (key == KEY_NONE) return;

    double zoom_factor = 1.5;
    double angle_step = 5.0 * M_PI / 180.0;  // 5 degrees in radians
    double color_rotation_step = 0.05;

    // Use pan_offset for movement when in perturbation mode (preserves precision at deep zoom)
    // Otherwise use DD/double arithmetic directly on center
    bool use_pan_offset = needs_perturbation(state);

    switch (key) {
#if defined(__APPLE__)
        // macOS zoom mode: arrow keys do zoom/rotate instead of pan
        case KEY_UP:
            if (state.zoom_mode) {
                state.commit_pan_offset();  // Commit before zoom change
                state.zoom *= zoom_factor;
                state.needs_redraw = true;
                break;
            }
            // Fall through to pan handling
            if (use_pan_offset) {
                double move = 0.1 / state.zoom;
                state.pan_offset_y = dd_add(state.pan_offset_y, -move);
            } else {
                state.center_y -= 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_DOWN:
            if (state.zoom_mode) {
                state.commit_pan_offset();
                state.zoom /= zoom_factor;
                state.needs_redraw = true;
                break;
            }
            if (use_pan_offset) {
                double move = 0.1 / state.zoom;
                state.pan_offset_y = dd_add(state.pan_offset_y, move);
            } else {
                state.center_y += 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_LEFT:
            if (state.zoom_mode) {
                state.angle -= angle_step;
                state.needs_redraw = true;
                break;
            }
            if (use_pan_offset) {
                double move = 0.1 / state.zoom;
                state.pan_offset_x = dd_add(state.pan_offset_x, -move);
            } else {
                state.center_x -= 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_RIGHT:
            if (state.zoom_mode) {
                state.angle += angle_step;
                state.needs_redraw = true;
                break;
            }
            if (use_pan_offset) {
                double move = 0.1 / state.zoom;
                state.pan_offset_x = dd_add(state.pan_offset_x, move);
            } else {
                state.center_x += 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
#else
        case KEY_UP:
            if (use_pan_offset) {
                // Accumulate in pan_offset in SCREEN coords (not rotated)
                // Rotation is applied in perturbation loop along with pixel offset
                double move = 0.1 / state.zoom;
                // Screen-up is -Y in screen coords
                state.pan_offset_y = dd_add(state.pan_offset_y, -move);
            } else {
                state.center_y -= 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_DOWN:
            if (use_pan_offset) {
                double move = 0.1 / state.zoom;
                // Screen-down is +Y in screen coords
                state.pan_offset_y = dd_add(state.pan_offset_y, move);
            } else {
                state.center_y += 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_LEFT:
            if (use_pan_offset) {
                double move = 0.1 / state.zoom;
                // Screen-left is -X in screen coords
                state.pan_offset_x = dd_add(state.pan_offset_x, -move);
            } else {
                state.center_x -= 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_RIGHT:
            if (use_pan_offset) {
                double move = 0.1 / state.zoom;
                // Screen-right is +X in screen coords
                state.pan_offset_x = dd_add(state.pan_offset_x, move);
            } else {
                state.center_x += 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
#endif

        case KEY_SHIFT_UP:
            state.commit_pan_offset();  // Commit before zoom change
            state.zoom *= zoom_factor;
            state.needs_redraw = true;
            break;
        case KEY_SHIFT_DOWN:
            state.commit_pan_offset();  // Commit before zoom change
            state.zoom /= zoom_factor;
            state.needs_redraw = true;
            break;
        case KEY_SHIFT_LEFT:
            state.angle -= angle_step;
            state.needs_redraw = true;
            break;
        case KEY_SHIFT_RIGHT:
            state.angle += angle_step;
            state.needs_redraw = true;
            break;

        case KEY_C:
            state.color_rotation -= color_rotation_step;
            state.needs_redraw = true;
            break;
        case KEY_V:
            state.color_rotation += color_rotation_step;
            state.needs_redraw = true;
            break;

        case KEY_PLUS:
            state.max_iter = std::min(4096, state.max_iter + 64);
            state.needs_redraw = true;
            break;
        case KEY_MINUS:
            state.max_iter = std::max(64, state.max_iter - 64);
            state.needs_redraw = true;
            break;

        case KEY_1: case KEY_2: case KEY_3: case KEY_4: case KEY_5:
        case KEY_6: case KEY_7: case KEY_8: case KEY_9:
            state.color_scheme = key - KEY_1;
            state.needs_redraw = true;
            break;

        case KEY_R:
            state.center_x = -0.5;
            state.center_y = 0.0;
            state.center_x_dd = DD(-0.5);
            state.center_y_dd = DD(0.0);
            state.pan_offset_x = DD(0.0);  // Clear pan offset on reset
            state.pan_offset_y = DD(0.0);
            state.dd_authoritative = false;  // Clear DD authority on reset
            state.zoom = 1.0;
            state.angle = 0.0;
            state.max_iter = 256;
            state.color_rotation = 0.0;
            state.use_perturbation = false;
            state.needs_redraw = true;
            break;

        case KEY_I:
            // Toggle iTerm2 image mode (only if iTerm2 is available)
            if (detect_iterm2()) {
                state.iterm_image_mode = !state.iterm_image_mode;
                state.needs_redraw = true;
            }
            break;

        case KEY_Q:
        case KEY_ESC:
            state.running = false;
            break;

        default:
            break;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AUTO EXPLORATION MODE
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration for cinematic path
struct CinematicPath;

struct AutoExplorer {
    bool enabled = false;
    double target_zoom = 1e14;           // How deep to zoom
    double zoom_factor = 1.008;          // Per-frame zoom multiplier (~60 fps)
    double rotation_speed = 0.3;         // Degrees per frame
    double current_rotation_dir = 1.0;   // 1 or -1
    std::mt19937 rng;

    // Trajectory mode: animate from start to target over duration seconds
    bool trajectory_mode = false;
    double trajectory_duration = 30.0;   // Duration in seconds (default 30)
    std::chrono::steady_clock::time_point trajectory_start;  // Use steady_clock for monotonic timing
    bool trajectory_started = false;

    // Start point (defaults)
    double start_x = -0.5;
    double start_y = 0.0;
    double start_zoom = 1.0;
    double start_angle = 0.0;

    // Target point (from CLI)
    double traj_target_x = -0.5;
    double traj_target_y = 0.0;
    double traj_target_zoom = 1.0;
    double traj_target_angle = 0.0;

    // DD precision targets for deep zoom (set from --pos DD format)
    DD traj_target_x_dd{-0.5};
    DD traj_target_y_dd{0.0};
    bool has_dd_target = false;  // True if DD targets were explicitly set

    // Cached log values for zoom interpolation (computed once at trajectory start)
    double log_start_zoom = 0.0;
    double log_target_zoom = 0.0;

    // Cinematic path for interesting trajectories
    std::unique_ptr<CinematicPath> cinematic_path;

    AutoExplorer() : rng(std::random_device{}()) {}
};

// ═══════════════════════════════════════════════════════════════════════════
// CINEMATIC TRAJECTORY PLANNING
// ═══════════════════════════════════════════════════════════════════════════

struct TrajectoryWaypoint {
    double x, y;           // Position in complex plane
    double zoom;           // Zoom level
    double angle;          // View rotation
    double interest_score; // For debugging/visualization
};

struct CinematicPath {
    std::vector<TrajectoryWaypoint> waypoints;
    double total_duration;
    bool valid = false;

    // Catmull-Rom spline evaluation at time t
    TrajectoryWaypoint evaluate(double t) const;
};

// Forward declarations
double calculate_interest_score_at_zoom(double cx, double cy, double zoom, int sample_radius = 5);
TrajectoryWaypoint find_best_waypoint(double center_x, double center_y, double zoom,
                                       double search_radius, double prev_x, double prev_y,
                                       std::mt19937& rng);
CinematicPath plan_cinematic_path(double start_x, double start_y, double start_zoom, double start_angle,
                                   double target_x, double target_y, double target_zoom, double target_angle,
                                   double duration, std::mt19937& rng);

// Quick iteration count for a single point
inline int quick_iterate(double cx, double cy, int max_iter) {
    double zr = 0, zi = 0;
    for (int i = 0; i < max_iter; i++) {
        double zr2 = zr * zr, zi2 = zi * zi;
        if (zr2 + zi2 > 4.0) return i;
        zi = 2 * zr * zi + cy;
        zr = zr2 - zi2 + cx;
    }
    return max_iter;
}

// Calculate "interestingness" score for a point
// Higher score = more interesting visually
double calculate_interest_score(double cx, double cy, int sample_radius = 3) {
    const int max_iter = 256;
    const double sample_scale = 0.001;  // Distance between samples

    std::vector<int> iterations;
    int bounded_count = 0;
    int escaped_count = 0;

    for (int dy = -sample_radius; dy <= sample_radius; dy++) {
        for (int dx = -sample_radius; dx <= sample_radius; dx++) {
            double px = cx + dx * sample_scale;
            double py = cy + dy * sample_scale;
            int iter = quick_iterate(px, py, max_iter);
            iterations.push_back(iter);
            if (iter >= max_iter) bounded_count++;
            else escaped_count++;
        }
    }

    // Score based on:
    // 1. Being on the boundary (mix of escaped and bounded)
    // 2. High variance in iteration counts (complex structure)
    // 3. Higher average iterations (near but not in set)

    double on_boundary_score = 0;
    if (bounded_count > 0 && escaped_count > 0) {
        double ratio = std::min(bounded_count, escaped_count) /
                       (double)std::max(bounded_count, escaped_count);
        on_boundary_score = ratio * 50.0;  // Up to 50 points for being on boundary
    }

    // Calculate variance
    double sum = 0, sum_sq = 0;
    for (int iter : iterations) {
        sum += iter;
        sum_sq += iter * iter;
    }
    double n = iterations.size();
    double mean = sum / n;
    double variance = (sum_sq / n) - (mean * mean);
    // Clamp to avoid negative variance from floating-point rounding
    variance = std::max(0.0, variance);
    double variance_score = std::min(50.0, sqrt(variance));  // Up to 50 points

    // Bonus for high average iteration (complex area)
    double avg_score = std::min(20.0, mean / 12.0);  // Up to 20 points

    return on_boundary_score + variance_score + avg_score;
}

// Find an interesting point along the Mandelbrot boundary
std::pair<double, double> find_interesting_point(AutoExplorer& explorer) {
    const int NUM_CANDIDATES = 50;
    const int BOUNDARY_SAMPLES = 200;

    std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
    std::uniform_real_distribution<double> radius_dist(0.3, 0.8);

    std::vector<std::tuple<double, double, double>> candidates;  // x, y, score

    // Strategy 1: Sample along the main cardioid and bulb boundaries
    for (int i = 0; i < BOUNDARY_SAMPLES; i++) {
        double theta = 2.0 * M_PI * i / BOUNDARY_SAMPLES;

        // Main cardioid boundary: r = (1 - cos(θ))/2
        double r = (1.0 - cos(theta)) / 2.0;
        double cx = r * cos(theta) - 0.25;
        double cy = r * sin(theta);

        // Add some randomness to escape exact boundary
        std::uniform_real_distribution<double> jitter(-0.02, 0.02);
        cx += jitter(explorer.rng);
        cy += jitter(explorer.rng);

        double score = calculate_interest_score(cx, cy);
        candidates.push_back({cx, cy, score});
    }

    // Strategy 2: Sample around known interesting regions
    const double interesting_regions[][2] = {
        {-0.75, 0.1},      // Seahorse valley
        {-0.75, -0.1},     // Seahorse valley (lower)
        {-1.25, 0.0},      // Antenna tip area
        {-0.16, 1.04},     // Top spiral
        {-0.16, -1.04},    // Bottom spiral
        {0.28, 0.53},      // Side bulb detail
        {0.28, -0.53},     // Side bulb detail (lower)
        {-1.77, 0.0},      // Far left antenna
        {-0.56, 0.64},     // Upper left detail
        {-0.56, -0.64},    // Lower left detail
    };

    for (const auto& region : interesting_regions) {
        std::uniform_real_distribution<double> local_dist(-0.05, 0.05);
        for (int j = 0; j < 5; j++) {
            double cx = region[0] + local_dist(explorer.rng);
            double cy = region[1] + local_dist(explorer.rng);
            double score = calculate_interest_score(cx, cy);
            candidates.push_back({cx, cy, score});
        }
    }

    // Strategy 3: Random boundary search - find points where iteration changes rapidly
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        double theta = angle_dist(explorer.rng);
        double r = radius_dist(explorer.rng);

        // Start from a random point and binary search toward the boundary
        double cx = r * cos(theta) - 0.5;
        double cy = r * sin(theta);

        // Binary search to find boundary
        double step = 0.1;
        int iter_prev = quick_iterate(cx, cy, 256);

        for (int s = 0; s < 20; s++) {
            double nx = cx + step * cos(theta);
            double ny = cy + step * sin(theta);
            int iter_new = quick_iterate(nx, ny, 256);

            if ((iter_prev >= 256) != (iter_new >= 256)) {
                // Found boundary crossing
                step *= -0.5;
            } else if (abs(iter_new - iter_prev) > 10) {
                // High iteration gradient
                step *= 0.5;
            }

            cx = nx;
            cy = ny;
            iter_prev = iter_new;
        }

        double score = calculate_interest_score(cx, cy);
        candidates.push_back({cx, cy, score});
    }

    // Sort by score and pick randomly from top candidates
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });

    // Pick randomly from top 10
    std::uniform_int_distribution<int> top_dist(0, std::min(9, (int)candidates.size() - 1));
    int pick = top_dist(explorer.rng);

    return {std::get<0>(candidates[pick]), std::get<1>(candidates[pick])};
}

// ═══════════════════════════════════════════════════════════════════════════
// CINEMATIC PATH IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════

// Calculate interest score at a specific zoom level
// The sample scale adapts to the zoom level for proper scale-aware scoring
double calculate_interest_score_at_zoom(double cx, double cy, double zoom, int sample_radius) {
    // FIX: Scale max_iter with zoom - deep zooms need more iterations
    // At zoom 1: 256, at zoom 1e6: 512, at zoom 1e12: 768
    const int max_iter = 256 + (int)(log10(std::max(1.0, zoom)) * 50);
    // FIX: Sample scale covers full viewport (3/zoom wide)
    // With sample_radius=5, we have 11 samples spanning viewport width
    const double viewport_width = 3.0 / zoom;
    const double sample_scale = viewport_width / (2.0 * sample_radius);

    std::vector<int> iterations;
    int bounded_count = 0;
    int escaped_count = 0;

    for (int dy = -sample_radius; dy <= sample_radius; dy++) {
        for (int dx = -sample_radius; dx <= sample_radius; dx++) {
            double px = cx + dx * sample_scale;
            double py = cy + dy * sample_scale;
            int iter = quick_iterate(px, py, max_iter);
            iterations.push_back(iter);
            if (iter >= max_iter) bounded_count++;
            else escaped_count++;
        }
    }

    // FIX: Strong penalty for uniform regions (all black or all escaped)
    // Single-color regions are BORING - never select them
    int total_samples = (int)iterations.size();
    if (bounded_count == total_samples || escaped_count == total_samples) {
        return -100.0;  // Strong negative score for uniform regions
    }

    // Score based on boundary proximity - ratio of minority to majority type
    // Higher ratio = closer to boundary = more interesting
    double ratio = std::min(bounded_count, escaped_count) /
                   (double)std::max(bounded_count, escaped_count);
    double on_boundary_score = ratio * 50.0;

    // Calculate variance for complexity
    double sum = 0, sum_sq = 0;
    for (int iter : iterations) {
        sum += iter;
        sum_sq += iter * iter;
    }
    double n = iterations.size();
    double mean = sum / n;
    double variance = std::max(0.0, (sum_sq / n) - (mean * mean));
    double variance_score = std::min(50.0, sqrt(variance));

    // Bonus for high average iteration (but NOT for uniform max_iter regions)
    double avg_score = std::min(20.0, mean / 12.0);

    return on_boundary_score + variance_score + avg_score;
}

// Find the best waypoint in a search region (EXPLORATORY mode)
TrajectoryWaypoint find_best_waypoint(double center_x, double center_y, double zoom,
                                       double search_radius, double prev_x, double prev_y,
                                       std::mt19937& rng) {
    const int NUM_CANDIDATES = 100;  // More candidates for thorough exploration
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // EXPLORATORY: Double the search radius for scenic options
    double effective_radius = search_radius * 2.0;
    int search_attempts = 0;
    const int MAX_SEARCH_ATTEMPTS = 3;  // Expand up to 3x if needed

    while (search_attempts < MAX_SEARCH_ATTEMPTS) {
        std::vector<TrajectoryWaypoint> candidates;

        for (int i = 0; i < NUM_CANDIDATES; i++) {
            double dx = dist(rng) * effective_radius;
            double dy = dist(rng) * effective_radius;
            double cx = center_x + dx;
            double cy = center_y + dy;

            double score = calculate_interest_score_at_zoom(cx, cy, zoom);

            // FIX: Scale-invariant continuity penalty (relative to search radius)
            double jump_dist = hypot(cx - prev_x, cy - prev_y);
            double jump_relative = jump_dist / effective_radius;
            double continuity_penalty = std::min(20.0, jump_relative * 3.0);
            score -= continuity_penalty;

            candidates.push_back({cx, cy, zoom, 0.0, score});
        }

        // Find best candidate
        auto best = std::max_element(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.interest_score < b.interest_score; });

        // FIX: If best score is too low (likely in boring region), expand search
        if (best->interest_score >= 5.0 || search_attempts == MAX_SEARCH_ATTEMPTS - 1) {
            return *best;
        }

        // Expand search radius and try again
        effective_radius *= 3.0;
        search_attempts++;
    }

    // Should never reach here, but return something safe
    return {center_x, center_y, zoom, 0.0, 0.0};
}

// Catmull-Rom spline interpolation helper
inline double catmull_rom(double p0, double p1, double p2, double p3, double t) {
    double t2 = t * t;
    double t3 = t2 * t;
    return 0.5 * ((2.0 * p1) +
                  (-p0 + p2) * t +
                  (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                  (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

// Smooth easing function (ease-in-out cubic)
inline double ease_in_out(double t) {
    return t < 0.5 ? 4 * t * t * t : 1 - pow(-2 * t + 2, 3) / 2;
}

// Evaluate cinematic path at time t using Catmull-Rom splines
TrajectoryWaypoint CinematicPath::evaluate(double t) const {
    if (!valid || waypoints.size() < 2) {
        return waypoints.empty() ? TrajectoryWaypoint{0, 0, 1, 0, 0} : waypoints[0];
    }

    // Normalize time to [0, 1] and apply easing
    double normalized = std::max(0.0, std::min(1.0, t / total_duration));
    double eased = ease_in_out(normalized);

    int n = waypoints.size() - 1;
    double segment_pos = eased * n;
    int segment = std::min((int)segment_pos, n - 1);
    double local_t = segment_pos - segment;

    // Get control points for Catmull-Rom (clamp at boundaries)
    int i0 = std::max(0, segment - 1);
    int i1 = segment;
    int i2 = std::min(segment + 1, n);
    int i3 = std::min(segment + 2, n);

    const auto& p0 = waypoints[i0];
    const auto& p1 = waypoints[i1];
    const auto& p2 = waypoints[i2];
    const auto& p3 = waypoints[i3];

    // Interpolate each component using Catmull-Rom spline
    double x = catmull_rom(p0.x, p1.x, p2.x, p3.x, local_t);
    double y = catmull_rom(p0.y, p1.y, p2.y, p3.y, local_t);

    // Interpolate zoom in log space for natural feel
    double log_zoom = catmull_rom(log(p0.zoom), log(p1.zoom), log(p2.zoom), log(p3.zoom), local_t);
    double zoom = exp(log_zoom);

    // Linear interpolation for angle (simpler, avoids wrap-around issues)
    double angle = p1.angle + local_t * (p2.angle - p1.angle);

    // Note: We removed the per-frame interest score check here.
    // With simplified paths (fixed position, zoom-only), there's no position
    // overshoot possible. The expensive score calculation was a performance bug.
    return {x, y, zoom, angle, 100.0};
}

// Plan a cinematic path - simple zoom/rotate at target position
// This creates a smooth zoom-in animation without jittery position changes
CinematicPath plan_cinematic_path(double start_x, double start_y, double start_zoom, double start_angle,
                                   double target_x, double target_y, double target_zoom, double target_angle,
                                   double duration, std::mt19937& /*rng*/) {
    CinematicPath path;
    path.total_duration = duration;

    // Simple approach: stay at target position, only interpolate zoom and angle
    // This avoids the jittery waypoint-hopping of complex path planning
    // Just two waypoints: start zoom/angle and end zoom/angle, both at target position
    path.waypoints.push_back({target_x, target_y, start_zoom, start_angle, 100.0});
    path.waypoints.push_back({target_x, target_y, target_zoom, target_angle, 100.0});

    path.valid = true;
    return path;
}


// Update auto exploration state
void update_auto_exploration(MandelbrotState& state, AutoExplorer& explorer) {
    if (!explorer.enabled) return;

    // Trajectory mode: interpolate from start to target over duration
    if (explorer.trajectory_mode) {
        // Plan the cinematic path on first call
        if (!explorer.trajectory_started) {
            // Plan the path through interesting waypoints
            explorer.cinematic_path = std::make_unique<CinematicPath>(
                plan_cinematic_path(
                    explorer.start_x, explorer.start_y, explorer.start_zoom, explorer.start_angle,
                    explorer.traj_target_x, explorer.traj_target_y,
                    explorer.traj_target_zoom, explorer.traj_target_angle,
                    explorer.trajectory_duration, explorer.rng
                )
            );
            explorer.trajectory_start = std::chrono::steady_clock::now();
            explorer.trajectory_started = true;
        }

        // Calculate elapsed time
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - explorer.trajectory_start).count();

        // Evaluate the cinematic path at current time
        TrajectoryWaypoint wp = explorer.cinematic_path->evaluate(elapsed);

        // Update state from waypoint
        state.center_x = wp.x;
        state.center_y = wp.y;
        state.center_x_dd = DD(wp.x);
        state.center_y_dd = DD(wp.y);
        state.zoom = wp.zoom;
        state.angle = wp.angle;

        // Clear pan offset to match cinematic path
        state.pan_offset_x = DD(0.0);
        state.pan_offset_y = DD(0.0);

        // Increase iterations as we zoom deeper
        int suggested_iter = 256 + (int)(log10(std::max(1.0, wp.zoom)) * 50);
        state.max_iter = std::min(2048, std::max(256, suggested_iter));

        state.needs_redraw = true;

        // Check if trajectory is complete
        if (elapsed >= explorer.trajectory_duration) {
            // Snap to exact target with full DD precision if available
            state.center_x = explorer.traj_target_x;
            state.center_y = explorer.traj_target_y;
            if (explorer.has_dd_target) {
                // Use full DD precision from --pos
                state.center_x_dd = explorer.traj_target_x_dd;
                state.center_y_dd = explorer.traj_target_y_dd;
                state.dd_authoritative = true;
            } else {
                state.center_x_dd = DD(explorer.traj_target_x);
                state.center_y_dd = DD(explorer.traj_target_y);
            }
            state.zoom = explorer.traj_target_zoom;
            state.angle = explorer.traj_target_angle;

            // Trajectory finished - disable auto mode
            explorer.enabled = false;
            explorer.trajectory_mode = false;
            explorer.cinematic_path.reset();
        }
        return;
    }

    // Regular auto exploration mode
    // Check if we've reached target zoom - find new point
    if (state.zoom >= explorer.target_zoom) {
        // Find new interesting point
        auto [new_x, new_y] = find_interesting_point(explorer);
        state.center_x = new_x;
        state.center_y = new_y;
        state.center_x_dd = DD(new_x);
        state.center_y_dd = DD(new_y);
        state.zoom = 1.0;

        // Randomize new target zoom (1e8 to 1e14)
        std::uniform_real_distribution<double> zoom_dist(8, 14);
        explorer.target_zoom = pow(10.0, zoom_dist(explorer.rng));

        // Randomize rotation direction
        std::uniform_int_distribution<int> dir_dist(0, 1);
        explorer.current_rotation_dir = dir_dist(explorer.rng) ? 1.0 : -1.0;

        // Randomize rotation speed (0.1 to 0.5 degrees per frame)
        std::uniform_real_distribution<double> speed_dist(0.1, 0.5);
        explorer.rotation_speed = speed_dist(explorer.rng);
    }

    // Apply zoom and rotation
    state.zoom *= explorer.zoom_factor;
    state.angle += explorer.rotation_speed * explorer.current_rotation_dir * M_PI / 180.0;

    // Increase iterations as we zoom deeper
    int suggested_iter = 256 + (int)(log10(state.zoom) * 50);
    state.max_iter = std::min(2048, std::max(256, suggested_iter));

    state.needs_redraw = true;
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI ARGUMENT PARSING
// ═══════════════════════════════════════════════════════════════════════════

// Parse complex number from string: "-0.5+0.3i", "-0.5-0.3i", "-0.5", "0.5i", "i", "-i"
// Formats: [real], [real][+-][imag]i, [imag]i
bool parse_complex(const char* str, double& re, double& im) {
    re = 0;
    im = 0;

    // Skip leading whitespace
    while (*str == ' ' || *str == '\t') str++;

    if (*str == '\0') return false;

    // Check for pure imaginary: "i", "-i", "+i", "0.5i", etc.
    // These have 'i' at the end with no +/- separator before it (except leading sign)
    const char* i_pos = strchr(str, 'i');
    if (!i_pos) i_pos = strchr(str, 'I');

    if (i_pos) {
        // Check if this is pure imaginary (no real part)
        // Pure imaginary: the only +/- are at the start or there's just "i"
        const char* p = str;
        bool has_separator = false;

        // Skip leading sign
        if (*p == '+' || *p == '-') p++;

        // Look for a +/- that would separate real and imaginary parts
        while (p < i_pos) {
            if ((*p == '+' || *p == '-') && p != str) {
                has_separator = true;
                break;
            }
            p++;
        }

        if (!has_separator) {
            // Pure imaginary: "i", "-i", "+i", "0.5i", "-.5i"
            if (i_pos == str || (i_pos == str + 1 && (*str == '+' || *str == '-'))) {
                // Just "i", "+i", or "-i"
                im = (*str == '-') ? -1.0 : 1.0;
            } else {
                // Something like "0.5i" or "-.5i"
                char* end;
                im = strtod(str, &end);
                if (end != i_pos) return false;  // Didn't parse up to 'i'
            }
            // Check nothing after 'i' (except whitespace)
            const char* after_i = i_pos + 1;
            while (*after_i == ' ' || *after_i == '\t') after_i++;
            return (*after_i == '\0');
        }
    }

    // Parse real part
    char* end;
    re = strtod(str, &end);
    if (end == str) return false;

    // Check for imaginary part
    if (*end == '+' || *end == '-') {
        char sign = *end;
        const char* imag_start = end + 1;

        // Handle "+i" or "-i" (implicit 1)
        if (*imag_start == 'i' || *imag_start == 'I') {
            im = (sign == '-') ? -1.0 : 1.0;
            end = const_cast<char*>(imag_start);
        } else {
            im = strtod(imag_start, &end);
            if (end == imag_start) return false;  // No number after +/-
            if (sign == '-') im = -im;
        }

        // Require 'i' suffix for imaginary part
        if (*end != 'i' && *end != 'I') return false;
        end++;
    }

    // Skip trailing whitespace
    while (*end == ' ' || *end == '\t') end++;

    return (*end == '\0');
}

// Parse complex number with DD (double-double) precision
// Formats supported:
//   DD format: <re_hi>:<re_lo><sign><im_hi>:<im_lo>i
//   Example:   -0.7115114743:1.2e-17+-0.3078112463:3.4e-18i
//   Regular:   -0.5+0.3i (falls back to regular parsing, .lo = 0)
bool parse_complex_dd(const char* str, DD& re_dd, DD& im_dd) {
    // Check for DD format indicator (colon in the string)
    if (strchr(str, ':') == nullptr) {
        // No colon = regular format, fall back to double parsing
        double re, im;
        if (parse_complex(str, re, im)) {
            re_dd = DD(re);
            im_dd = DD(im);
            return true;
        }
        return false;
    }

    // DD format: re_hi:re_lo+im_hi:im_lo i  or  re_hi:re_lo-im_hi:im_lo i
    // Parse real part: re_hi:re_lo
    char* end;
    double re_hi = strtod(str, &end);
    if (end == str || *end != ':') return false;

    const char* re_lo_start = end + 1;
    double re_lo = strtod(re_lo_start, &end);
    if (end == re_lo_start) return false;

    // Parse sign separator between real and imaginary
    if (*end != '+' && *end != '-') return false;
    char sign = *end;
    end++;

    // Parse imaginary part: im_hi:im_lo i
    double im_hi = strtod(end, &end);
    if (*end != ':') return false;

    const char* im_lo_start = end + 1;
    double im_lo = strtod(im_lo_start, &end);
    if (end == im_lo_start) return false;

    // Require 'i' suffix
    if (*end != 'i' && *end != 'I') return false;
    end++;

    // Skip trailing whitespace
    while (*end == ' ' || *end == '\t') end++;
    if (*end != '\0') return false;

    // Apply sign to imaginary part
    if (sign == '-') {
        im_hi = -im_hi;
        im_lo = -im_lo;
    }

    re_dd = DD{re_hi, re_lo};
    im_dd = DD{im_hi, im_lo};
    return true;
}

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --pos <position>   Target position in complex format:\n");
    printf("                     Standard: re+imi (e.g., -0.5+0.3i)\n");
    printf("                     DD:       re_hi:re_lo+im_hi:im_lo i (deep zoom)\n");
    printf("  --zoom <value>     Target zoom level (e.g., 1e6)\n");
    printf("  --angle <degrees>  Target view angle (e.g., 45)\n");
    printf("  --auto [N]         Enable automatic exploration, or with --pos/--zoom:\n");
    printf("                     animate from default to target over N seconds (default 30)\n");
    printf("  --image [WxH]      Enable iTerm2 inline image mode (requires iTerm2)\n");
    printf("                     Optional resolution, e.g., --image=800x600 (default 640x400)\n");
    printf("  --benchmark        Compute one frame and print timing (no interactive mode)\n");
    printf("  --output <file>    Save rendered image to file (PPM or PNG)\n");
    printf("                     Use with --image for resolution, e.g.:\n");
    printf("                     --image=1200x800 --output=out.png\n");
    printf("  --no-sa            Disable Series Approximation (for benchmarking)\n");
    printf("  --help             Show this help message\n");
    printf("\nDebug options:\n");
    printf("  --debug            Print DD precision values to stderr at parse and exit\n");
    printf("  --exit-now         Exit immediately after parsing (for testing DD round-trip)\n");
    printf("\nExamples:\n");
    printf("  %s --pos -0.7+0.3i --zoom 1e6\n", prog);
    printf("      Start at specified position and zoom\n");
    printf("  %s --auto\n", prog);
    printf("      Automatic exploration mode\n");
    printf("  %s --pos -0.7+0.3i --zoom 1e11 --auto 60\n", prog);
    printf("      Animate from default to target over 60 seconds\n");
    printf("\nControls:\n");
    printf("  Arrow Keys          - Pan view\n");
    printf("  SHIFT + Up/Down     - Zoom in/out\n");
    printf("  SHIFT + Left/Right  - Rotate view\n");
    printf("  C/V                 - Rotate color palette\n");
    printf("  1-9                 - Switch color schemes\n");
    printf("  +/-                 - Adjust max iterations\n");
    printf("  I                   - Toggle iTerm2 image mode (higher resolution)\n");
#if defined(__APPLE__)
    printf("  Z                   - Toggle arrow mode (Pan <-> Zoom/Rotate)\n");
#endif
    printf("  R                   - Reset view\n");
    printf("  Q/ESC               - Quit\n");
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    MandelbrotState state;
    AutoExplorer explorer;
    g_state = &state;

    // Parse command line arguments
    // Track if --pos or --zoom were specified (for trajectory mode)
    bool has_target_pos = false;
    bool has_target_zoom = false;
    double cli_target_x = -0.5, cli_target_y = 0.0;
    DD cli_target_x_dd{-0.5}, cli_target_y_dd{0.0};  // DD precision for deep zoom
    bool cli_has_dd_pos = false;  // True if --pos used DD format
    double cli_target_zoom = 1.0;
    double cli_target_angle = 0.0;
    bool debug_dd = false;
    bool exit_now = false;  // Exit immediately after parsing (for testing)
    bool benchmark_mode = false;  // Compute one frame and print timing
    bool disable_sa = false;      // Disable Series Approximation
    std::string output_file;      // Output file path (PPM or PNG)

    static struct option long_options[] = {
        {"pos",   required_argument, 0, 'p'},
        {"zoom",  required_argument, 0, 'z'},
        {"angle", required_argument, 0, 'a'},
        {"auto",  optional_argument, 0, 'A'},
        {"image", optional_argument, 0, 'I'},
        {"benchmark", no_argument,   0, 'B'},
        {"output", required_argument, 0, 'O'},
        {"no-sa", no_argument,       0, 'S'},
        {"debug", no_argument,       0, 'D'},
        {"exit-now", no_argument,    0, 'X'},
        {"help",  no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:z:a:ADh", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p': {
                DD re_dd, im_dd;
                if (parse_complex_dd(optarg, re_dd, im_dd)) {
                    // Set DD values with full precision
                    state.center_x_dd = re_dd;
                    state.center_y_dd = im_dd;
                    // Sync double values from DD (hi + lo for best approximation)
                    state.center_x = re_dd.hi + re_dd.lo;
                    state.center_y = im_dd.hi + im_dd.lo;
                    // Mark DD as authoritative if .lo components are non-zero
                    // (i.e., DD format was used, not regular double format)
                    state.dd_authoritative = (re_dd.lo != 0.0 || im_dd.lo != 0.0);
                    // Set trajectory targets (both double and DD)
                    cli_target_x = state.center_x;
                    cli_target_y = state.center_y;
                    cli_target_x_dd = re_dd;
                    cli_target_y_dd = im_dd;
                    cli_has_dd_pos = state.dd_authoritative;
                    has_target_pos = true;
                } else {
                    fprintf(stderr, "Error: Invalid position format '%s'\n", optarg);
                    fprintf(stderr, "Expected formats:\n");
                    fprintf(stderr, "  Standard: re+imi (e.g., -0.5+0.3i)\n");
                    fprintf(stderr, "  DD:       re_hi:re_lo+im_hi:im_lo i (for deep zoom)\n");
                    return 1;
                }
                break;
            }
            case 'z': {
                char* end;
                double zoom = strtod(optarg, &end);
                if (end == optarg || zoom <= 0) {
                    fprintf(stderr, "Error: Invalid zoom value '%s'\n", optarg);
                    return 1;
                }
                cli_target_zoom = zoom;
                has_target_zoom = true;
                state.zoom = zoom;
                break;
            }
            case 'a': {
                char* end;
                double angle_deg = strtod(optarg, &end);
                if (end == optarg) {
                    fprintf(stderr, "Error: Invalid angle value '%s'\n", optarg);
                    return 1;
                }
                cli_target_angle = angle_deg * M_PI / 180.0;
                state.angle = cli_target_angle;
                break;
            }
            case 'A':
                explorer.enabled = true;
                // Parse optional duration argument (e.g., --auto 60 or --auto=60)
                if (optarg) {
                    // Handle --auto=60 syntax (optarg is set)
                    char* end;
                    double duration = strtod(optarg, &end);
                    if (end != optarg && duration > 0) {
                        explorer.trajectory_duration = duration;
                    }
                } else if (optind < argc && argv[optind] && argv[optind][0] != '-') {
                    // Handle --auto 60 syntax (optarg not set, check next argv)
                    char* end;
                    double duration = strtod(argv[optind], &end);
                    if (end != argv[optind] && *end == '\0' && duration > 0) {
                        explorer.trajectory_duration = duration;
                        optind++;  // Consume the duration argument
                    }
                }
                break;
            case 'I':
                // Enable iTerm2 image mode (auto-detect or force)
                if (detect_iterm2()) {
                    state.iterm_image_mode = true;
                    // Parse optional resolution argument (e.g., --image=800x600)
                    if (optarg) {
                        int w, h;
                        if (sscanf(optarg, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                            state.image_width = w;
                            state.image_height = h;
                        }
                    }
                    fprintf(stderr, "iTerm2 detected: image mode enabled (%dx%d)\n",
                            state.image_width, state.image_height);
                } else {
                    fprintf(stderr, "Warning: iTerm2 not detected, --image requires iTerm2\n");
                }
                break;
            case 'B':
                benchmark_mode = true;
                break;
            case 'O':
                output_file = optarg;
                break;
            case 'S':
                disable_sa = true;
                break;
            case 'D':
                debug_dd = true;
                break;
            case 'X':
                exit_now = true;
                debug_dd = true;  // Also enable debug output
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Apply disable_sa to state
    state.disable_sa = disable_sa;

    // Debug: Print parsed DD values
    if (debug_dd && has_target_pos) {
        fprintf(stderr, "\n=== DD DEBUG: After parsing --pos ===\n");
        fprintf(stderr, "center_x_dd: hi=%.17g lo=%.17g (0x%llx, 0x%llx)\n",
                state.center_x_dd.hi, state.center_x_dd.lo,
                *(unsigned long long*)&state.center_x_dd.hi, *(unsigned long long*)&state.center_x_dd.lo);
        fprintf(stderr, "center_y_dd: hi=%.17g lo=%.17g (0x%llx, 0x%llx)\n",
                state.center_y_dd.hi, state.center_y_dd.lo,
                *(unsigned long long*)&state.center_y_dd.hi, *(unsigned long long*)&state.center_y_dd.lo);
        fprintf(stderr, "dd_authoritative: %s\n", state.dd_authoritative ? "true" : "false");
        fprintf(stderr, "NOTE: Run 'q' immediately and check terminal dims at exit to debug visual shift\n");
        fprintf(stderr, "=====================================\n\n");
        if (exit_now) {
            fprintf(stderr, "Exiting immediately (--exit-now)\n");
            return 0;
        }
    }

    // Set up trajectory mode if --auto combined with --pos or --zoom
    if (explorer.enabled && (has_target_pos || has_target_zoom)) {
        // Trajectory mode: animate from default start to specified target
        explorer.trajectory_mode = true;

        // Store target values (both double and DD for precision)
        explorer.traj_target_x = cli_target_x;
        explorer.traj_target_y = cli_target_y;
        explorer.traj_target_x_dd = cli_target_x_dd;
        explorer.traj_target_y_dd = cli_target_y_dd;
        explorer.has_dd_target = cli_has_dd_pos;
        explorer.traj_target_zoom = cli_target_zoom;
        explorer.traj_target_angle = cli_target_angle;

        // Cache log values for zoom interpolation (avoids per-frame log() calls)
        explorer.log_start_zoom = log(explorer.start_zoom);
        explorer.log_target_zoom = log(explorer.traj_target_zoom);

        // Reset state to defaults (start of trajectory)
        state.center_x = explorer.start_x;
        state.center_y = explorer.start_y;
        state.center_x_dd = DD(explorer.start_x);
        state.center_y_dd = DD(explorer.start_y);
        state.zoom = explorer.start_zoom;
        state.angle = explorer.start_angle;

        // Clear pan offset to ensure view matches interpolated path
        state.pan_offset_x = DD(0.0);
        state.pan_offset_y = DD(0.0);
    } else if (explorer.enabled) {
        // Regular auto mode: find interesting point and start exploring
        auto [new_x, new_y] = find_interesting_point(explorer);
        state.center_x = new_x;
        state.center_y = new_y;
        state.center_x_dd = DD(new_x);
        state.center_y_dd = DD(new_y);
    }

    // Benchmark mode: compute one frame and print timing
    if (benchmark_mode) {
        // Set up dimensions
        state.width = state.iterm_image_mode ? state.image_width : 80;
        state.height = state.iterm_image_mode ? state.image_height : 48;
        state.iterations.resize(state.width * state.height);
        state.image_buffer.resize(state.width * state.height * 3);

        // Scale max_iter with zoom (same as normal mode)
        int suggested_iter = 256 + (int)(log10(std::max(1.0, state.zoom)) * 50);
        state.max_iter = std::min(2048, std::max(256, suggested_iter));

        // Run 3 iterations for warm-up + averaging
        const int RUNS = 3;
        double times[RUNS];

        fprintf(stderr, "Benchmark: %dx%d, zoom=%.2e, SA=%s, max_iter=%d\n",
                state.width, state.height, state.zoom,
                disable_sa ? "OFF" : "ON", state.max_iter);

        for (int r = 0; r < RUNS; r++) {
            state.needs_redraw = true;
            auto start = std::chrono::high_resolution_clock::now();
            compute_mandelbrot_unified(state);
            auto end = std::chrono::high_resolution_clock::now();
            times[r] = std::chrono::duration<double>(end - start).count();
            fprintf(stderr, "  Run %d: %.3fs\n", r + 1, times[r]);
        }

        double avg = (times[0] + times[1] + times[2]) / 3.0;
        fprintf(stderr, "Average: %.3fs\n", avg);

        // Show SA skip info if perturbation was used
        if (state.use_perturbation && !disable_sa) {
            double aspect = (double)state.width / (state.height * 2.0);
            double scale = 3.0 / state.zoom;
            double dCr = scale * aspect * 0.1;
            double dCi = scale * 0.1;
            double dzr, dzi;
            int skip = sa_find_skip_iteration(state.primary_orbit, dCr, dCi, dzr, dzi, state.max_iter);
            fprintf(stderr, "SA skip: %d of %d iterations (%.1f%%)\n",
                    skip, state.max_iter, 100.0 * skip / state.max_iter);
        }

        // Save output if requested
        if (!output_file.empty()) {
            // Render to image buffer
            compute_mandelbrot_image(state);
            auto ppm = create_ppm(state.image_buffer.data(), state.width, state.height);

            // Check if PNG output requested
            bool want_png = (output_file.size() > 4 &&
                            output_file.substr(output_file.size() - 4) == ".png");

            if (want_png) {
                // Write PPM to temp file, convert with sips
                std::string tmp_ppm = output_file + ".tmp.ppm";
                FILE* f = fopen(tmp_ppm.c_str(), "wb");
                if (f) {
                    fwrite(ppm.data(), 1, ppm.size(), f);
                    fclose(f);
                    std::string cmd = "sips -s format png \"" + tmp_ppm + "\" --out \"" + output_file + "\" >/dev/null 2>&1";
                    int ret = system(cmd.c_str());
                    unlink(tmp_ppm.c_str());
                    if (ret == 0) {
                        fprintf(stderr, "Saved: %s\n", output_file.c_str());
                    } else {
                        fprintf(stderr, "Error: Failed to convert to PNG\n");
                    }
                } else {
                    fprintf(stderr, "Error: Cannot write %s\n", tmp_ppm.c_str());
                }
            } else {
                // Write PPM directly
                FILE* f = fopen(output_file.c_str(), "wb");
                if (f) {
                    fwrite(ppm.data(), 1, ppm.size(), f);
                    fclose(f);
                    fprintf(stderr, "Saved: %s\n", output_file.c_str());
                } else {
                    fprintf(stderr, "Error: Cannot write %s\n", output_file.c_str());
                }
            }
        }
        return 0;
    }

    // Output-only mode (--output without --benchmark)
    if (!output_file.empty()) {
        // Set up dimensions from image mode or defaults
        state.width = state.iterm_image_mode ? state.image_width : 800;
        state.height = state.iterm_image_mode ? state.image_height : 600;
        state.iterations.resize(state.width * state.height);
        state.image_buffer.resize(state.width * state.height * 3);

        // Scale max_iter with zoom
        int suggested_iter = 256 + (int)(log10(std::max(1.0, state.zoom)) * 50);
        state.max_iter = std::min(2048, std::max(256, suggested_iter));

        fprintf(stderr, "Rendering: %dx%d, zoom=%.2e\n", state.width, state.height, state.zoom);

        state.needs_redraw = true;
        compute_mandelbrot_unified(state);
        compute_mandelbrot_image(state);

        auto ppm = create_ppm(state.image_buffer.data(), state.width, state.height);

        bool want_png = (output_file.size() > 4 &&
                        output_file.substr(output_file.size() - 4) == ".png");

        if (want_png) {
            std::string tmp_ppm = output_file + ".tmp.ppm";
            FILE* f = fopen(tmp_ppm.c_str(), "wb");
            if (f) {
                fwrite(ppm.data(), 1, ppm.size(), f);
                fclose(f);
                std::string cmd = "sips -s format png \"" + tmp_ppm + "\" --out \"" + output_file + "\" >/dev/null 2>&1";
                int ret = system(cmd.c_str());
                unlink(tmp_ppm.c_str());
                if (ret == 0) {
                    fprintf(stderr, "Saved: %s\n", output_file.c_str());
                } else {
                    fprintf(stderr, "Error: Failed to convert to PNG\n");
                }
            } else {
                fprintf(stderr, "Error: Cannot write %s\n", tmp_ppm.c_str());
            }
        } else {
            FILE* f = fopen(output_file.c_str(), "wb");
            if (f) {
                fwrite(ppm.data(), 1, ppm.size(), f);
                fclose(f);
                fprintf(stderr, "Saved: %s\n", output_file.c_str());
            } else {
                fprintf(stderr, "Error: Cannot write %s\n", output_file.c_str());
            }
        }
        return 0;
    }

    setup_terminal();

    signal(SIGWINCH, sigwinch_handler);
    signal(SIGINT, sigint_handler);

    handle_resize(state);

    printf(CLEAR_SCREEN);

    // Target 60 FPS
    const int TARGET_FPS = 60;
    const long FRAME_TIME_US = 1000000 / TARGET_FPS;  // ~16667 microseconds

    while (state.running) {
        auto frame_start = std::chrono::high_resolution_clock::now();

        // Handle signals outside handlers for async-signal safety
        if (sigint_pending) {
            sigint_pending = 0;
            state.running = false;
        }
        if (resize_pending) {
            resize_pending = 0;
            handle_resize(state);
        }

        // Update auto exploration
        update_auto_exploration(state, explorer);

        if (state.needs_redraw) {
            state.needs_redraw = false;
            compute_mandelbrot_unified(state);
            render_frame(state);
        }

        handle_input(state);

        // Frame rate limiting
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            frame_end - frame_start).count();

        if (elapsed_us < FRAME_TIME_US) {
            usleep(FRAME_TIME_US - elapsed_us);
        }
    }

    restore_terminal();

    // Commit any accumulated pan offset before printing final position
    state.commit_pan_offset();

    // Debug: Print DD values at exit
    if (debug_dd) {
        fprintf(stderr, "\n=== DD DEBUG: At exit ===\n");
        fprintf(stderr, "Terminal: %d x %d (cols x rows as width x height)\n", state.width, state.height);
        fprintf(stderr, "center_x_dd: hi=%.17g lo=%.17g (0x%llx, 0x%llx)\n",
                state.center_x_dd.hi, state.center_x_dd.lo,
                *(unsigned long long*)&state.center_x_dd.hi, *(unsigned long long*)&state.center_x_dd.lo);
        fprintf(stderr, "center_y_dd: hi=%.17g lo=%.17g (0x%llx, 0x%llx)\n",
                state.center_y_dd.hi, state.center_y_dd.lo,
                *(unsigned long long*)&state.center_y_dd.hi, *(unsigned long long*)&state.center_y_dd.lo);
        fprintf(stderr, "dd_authoritative: %s\n", state.dd_authoritative ? "true" : "false");
        fprintf(stderr, "=========================\n\n");
    }

    // Print CLI command to return to this location
    // Use DD format (hi:lo) for full precision at deep zoom
    double angle_deg = state.angle * 180.0 / M_PI;
    printf("\n");
    printf("To return to this location:\n");
    // Format: --pos <re_hi>:<re_lo><sign><im_hi>:<im_lo>i
    char sign = (state.center_y_dd.hi >= 0) ? '+' : '-';
    double im_hi_abs = fabs(state.center_y_dd.hi);
    double im_lo = (state.center_y_dd.hi >= 0) ? state.center_y_dd.lo : -state.center_y_dd.lo;
    printf("  ./mandelbrot --pos %.17g:%.17g%c%.17g:%.17gi --zoom %.17g --angle %.6f\n",
           state.center_x_dd.hi, state.center_x_dd.lo,
           sign, im_hi_abs, im_lo,
           state.zoom, angle_deg);
    printf("\n");
    printf("Thanks for exploring the Mandelbrot set!\n");
    printf("Compiled with %s optimization\n", USE_AVX2 ? "AVX2 SIMD" : "scalar");

    return 0;
}
