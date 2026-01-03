/*
 * PERTURBATION REGRESSION TEST
 *
 * Tests perturbation rendering against direct DD iteration at deep zoom.
 * Reproduces the "mostly black" bug at zoom ~2.79e+11.
 *
 * Usage: ./test_perturbation [--verbose]
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════
// DOUBLE-DOUBLE ARITHMETIC (copied from mandelbrot.cpp with strict FP)
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math")
#endif

#if defined(__clang__)
#define STRICT_FP __attribute__((optnone))
#else
#define STRICT_FP
#endif

struct DD {
    double hi, lo;
    DD() : hi(0), lo(0) {}
    DD(double h) : hi(h), lo(0) {}
    DD(double h, double l) : hi(h), lo(l) {}
    explicit operator double() const { return hi + lo; }
};

STRICT_FP inline void two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    double v = s - a;
    e = (a - (s - v)) + (b - v);
}

STRICT_FP inline void quick_two_sum(double a, double b, double& s, double& e) {
    s = a + b;
    e = b - (s - a);
}

STRICT_FP inline void two_prod(double a, double b, double& p, double& e) {
    p = a * b;
#ifdef FP_FAST_FMA
    e = std::fma(a, b, -p);
#else
    constexpr double SPLIT = 134217729.0;
    double ca = SPLIT * a;
    double cb = SPLIT * b;
    double a_hi = ca - (ca - a);
    double a_lo = a - a_hi;
    double b_hi = cb - (cb - b);
    double b_lo = b - b_hi;
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
#endif
}

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

inline DD dd_add(DD a, double b) {
    double s1, s2;
    two_sum(a.hi, b, s1, s2);
    s2 += a.lo;
    quick_two_sum(s1, s2, s1, s2);
    return DD(s1, s2);
}

inline DD dd_sub(DD a, DD b) {
    return dd_add(a, DD(-b.hi, -b.lo));
}

inline DD dd_mul(DD a, DD b) {
    double p1, p2;
    two_prod(a.hi, b.hi, p1, p2);
    p2 += a.hi * b.lo + a.lo * b.hi;
    quick_two_sum(p1, p2, p1, p2);
    return DD(p1, p2);
}

inline DD dd_mul(DD a, double b) {
    double p1, p2;
    two_prod(a.hi, b, p1, p2);
    p2 += a.lo * b;
    quick_two_sum(p1, p2, p1, p2);
    return DD(p1, p2);
}

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

struct DDComplex {
    DD re, im;
    DDComplex() : re(), im() {}
    DDComplex(DD r, DD i) : re(r), im(i) {}
    DDComplex(double r, double i) : re(r), im(i) {}

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
// REFERENCE ORBIT (FIXED implementation - stores full DD precision)
// ═══════════════════════════════════════════════════════════════════════════

struct ReferenceOrbit {
    // Store full DD precision: hi and lo parts separately
    std::vector<double> Zr_hi, Zr_lo;
    std::vector<double> Zi_hi, Zi_lo;
    // Precomputed sums for fast access
    std::vector<double> Zr_sum;
    std::vector<double> Zi_sum;
    std::vector<double> Z_norm;
    int length;
    int escape_iter;
    DD center_re, center_im;

    ReferenceOrbit() : length(0), escape_iter(-1) {}

    void clear() {
        Zr_hi.clear();
        Zr_lo.clear();
        Zi_hi.clear();
        Zi_lo.clear();
        Zr_sum.clear();
        Zi_sum.clear();
        Z_norm.clear();
        length = 0;
        escape_iter = -1;
    }
};

void compute_reference_orbit(ReferenceOrbit& orbit, DD center_x, DD center_y, int max_iter) {
    orbit.clear();
    orbit.center_re = center_x;
    orbit.center_im = center_y;

    orbit.Zr_hi.reserve(max_iter + 1);
    orbit.Zr_lo.reserve(max_iter + 1);
    orbit.Zi_hi.reserve(max_iter + 1);
    orbit.Zi_lo.reserve(max_iter + 1);
    orbit.Zr_sum.reserve(max_iter + 1);
    orbit.Zi_sum.reserve(max_iter + 1);
    orbit.Z_norm.reserve(max_iter + 1);

    // Initial Z = 0
    orbit.Zr_hi.push_back(0.0);
    orbit.Zr_lo.push_back(0.0);
    orbit.Zi_hi.push_back(0.0);
    orbit.Zi_lo.push_back(0.0);
    orbit.Zr_sum.push_back(0.0);
    orbit.Zi_sum.push_back(0.0);
    orbit.Z_norm.push_back(0.0);

    DDComplex C(center_x, center_y);
    DDComplex Z(0.0, 0.0);

    for (int n = 0; n < max_iter; n++) {
        Z = Z.square() + C;

        // Store full DD precision
        orbit.Zr_hi.push_back(Z.re.hi);
        orbit.Zr_lo.push_back(Z.re.lo);
        orbit.Zi_hi.push_back(Z.im.hi);
        orbit.Zi_lo.push_back(Z.im.lo);

        // Precompute sums
        double zr_sum = Z.re.hi + Z.re.lo;
        double zi_sum = Z.im.hi + Z.im.lo;
        orbit.Zr_sum.push_back(zr_sum);
        orbit.Zi_sum.push_back(zi_sum);

        // Compute norm from full-precision sums
        double norm = zr_sum * zr_sum + zi_sum * zi_sum;
        orbit.Z_norm.push_back(norm);

        if (norm > 1e6) {
            orbit.escape_iter = n + 1;
            orbit.length = n + 2;
            return;
        }
    }

    orbit.escape_iter = -1;
    orbit.length = max_iter + 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// PERTURBATION COMPUTATION (FIXED with adaptive glitch detection)
// ═══════════════════════════════════════════════════════════════════════════

// Glitch detection constants
// Key insight: δz grows exponentially, so it WILL become larger than Z.
// This is expected behavior, not a glitch. A true glitch is when precision
// is lost in the δz computation itself, not just when |δz| > |Z|.
// We use a very lenient threshold and primarily rely on the escape/iteration check.
constexpr double GLITCH_BASE_RATIO = 1e6;   // Only glitch if δz dominates by 1000x
constexpr double GLITCH_ESCAPE_RATIO = 1e8; // Even more lenient near escape
constexpr double GLITCH_FLOOR = 1e-30;
constexpr int GLITCH_MIN_ITER = 50;         // Skip more early iterations
constexpr double NEAR_ESCAPE_THRESHOLD = 3.5; // Higher threshold
constexpr int ESCAPE_WINDOW = 20;

double smooth_iter(double zr, double zi, int iter, int max_iter) {
    if (iter >= max_iter) return -1.0;
    double log_zn = log(zr * zr + zi * zi) / 2.0;
    double nu = log(log_zn / log(2.0)) / log(2.0);
    return iter + 1 - nu;
}

struct PixelResult {
    double iteration;  // Smooth iteration count, or -1 (bounded), -2 (glitched)
    bool escaped;
    bool glitched;
    int raw_iter;
};

PixelResult compute_perturbation_pixel(
    double dCr, double dCi,
    const ReferenceOrbit& orbit,
    int max_iter
) {
    PixelResult result = {-1.0, false, false, 0};

    double dzr = 0.0, dzi = 0.0;
    int iter = 0;
    int max_ref = orbit.length - 1;

    while (iter < max_iter && iter < max_ref) {
        // Use precomputed sums for full DD precision
        double Zr = orbit.Zr_sum[iter];
        double Zi = orbit.Zi_sum[iter];

        double temp_r = 2.0 * Zr + dzr;
        double temp_i = 2.0 * Zi + dzi;

        double new_dzr = temp_r * dzr - temp_i * dzi + dCr;
        double new_dzi = temp_r * dzi + temp_i * dzr + dCi;

        dzr = new_dzr;
        dzi = new_dzi;
        iter++;

        // Use precomputed sums
        double full_zr = orbit.Zr_sum[iter] + dzr;
        double full_zi = orbit.Zi_sum[iter] + dzi;
        double mag2 = full_zr * full_zr + full_zi * full_zi;

        if (mag2 > 4.0) {
            result.escaped = true;
            result.raw_iter = iter;
            result.iteration = smooth_iter(full_zr, full_zi, iter, max_iter);
            return result;
        }

        // Adaptive glitch detection (skip early iters and near-escape)
        if (iter > GLITCH_MIN_ITER && mag2 < NEAR_ESCAPE_THRESHOLD) {
            double dz_mag2 = dzr * dzr + dzi * dzi;
            double Z_mag2 = orbit.Z_norm[iter] + GLITCH_FLOOR;

            // Adaptive ratio based on escape status
            double ratio_threshold = GLITCH_BASE_RATIO;
            if (orbit.escape_iter >= 0 && iter > orbit.escape_iter - ESCAPE_WINDOW) {
                ratio_threshold = GLITCH_ESCAPE_RATIO;
            }
            double adaptive_ratio = ratio_threshold * (1.0 + 0.1 * sqrt(Z_mag2));

            if (dz_mag2 / Z_mag2 > adaptive_ratio) {
                result.glitched = true;
                result.iteration = -2.0;
                result.raw_iter = iter;
                return result;
            }
        }
    }

    // If we ran out of reference orbit but haven't reached max_iter,
    // continue with direct iteration (this is the fallback)
    if (iter >= max_ref && iter < max_iter) {
        // Current Z = Z_ref + δz
        double zr = orbit.Zr_sum[max_ref] + dzr;
        double zi = orbit.Zi_sum[max_ref] + dzi;

        // Continue standard iteration (using the actual C coordinate)
        // Use dd_add for C to match runtime precision (Codex review fix)
        DD cx_dd = dd_add(orbit.center_re, dCr);
        DD cy_dd = dd_add(orbit.center_im, dCi);
        double cx = cx_dd.hi + cx_dd.lo;
        double cy = cy_dd.hi + cy_dd.lo;

        while (zr * zr + zi * zi < 4.0 && iter < max_iter) {
            double tmp = zr * zr - zi * zi + cx;
            zi = 2 * zr * zi + cy;
            zr = tmp;
            iter++;
        }

        if (iter < max_iter) {
            result.escaped = true;
            result.raw_iter = iter;
            result.iteration = smooth_iter(zr, zi, iter, max_iter);
            return result;
        }
    }

    result.raw_iter = iter;
    result.iteration = -1.0;  // Bounded
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// DIRECT DD ITERATION (ground truth)
// ═══════════════════════════════════════════════════════════════════════════

PixelResult compute_direct_dd(DD cx, DD cy, int max_iter) {
    PixelResult result = {-1.0, false, false, 0};

    DDComplex C(cx, cy);
    DDComplex Z(0.0, 0.0);

    for (int n = 0; n < max_iter; n++) {
        Z = Z.square() + C;
        // Use full DD precision for escape check (matches runtime fix)
        double mag2 = Z.norm_full();

        if (mag2 > 4.0) {
            result.escaped = true;
            result.raw_iter = n + 1;
            result.iteration = smooth_iter(Z.re.hi, Z.im.hi, n + 1, max_iter);
            return result;
        }
    }

    result.raw_iter = max_iter;
    result.iteration = -1.0;  // Bounded
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST HARNESS
// ═══════════════════════════════════════════════════════════════════════════

struct TestConfig {
    DD center_x, center_y;
    double zoom;
    double angle_deg;
    int width, height;
    int max_iter;
};

struct TestResults {
    int total_pixels;
    int direct_escaped;
    int direct_bounded;
    int perturb_escaped;
    int perturb_bounded;
    int perturb_glitched;
    int matches;           // Same escape status and close iteration
    int mismatches;        // Different escape status
    double max_iter_diff;  // Max iteration difference for matching pixels
    double avg_iter_diff;  // Average iteration difference
};

void run_test(const TestConfig& config, TestResults& results, bool verbose) {
    results = {};
    results.total_pixels = config.width * config.height;

    double aspect = (double)config.width / (config.height * 2.0);
    double scale = 3.0 / config.zoom;
    double angle_rad = config.angle_deg * M_PI / 180.0;
    double cos_a = cos(angle_rad);
    double sin_a = sin(angle_rad);

    // Compute reference orbit at screen center
    ReferenceOrbit orbit;
    compute_reference_orbit(orbit, config.center_x, config.center_y, config.max_iter);

    if (verbose) {
        printf("Reference orbit: length=%d, escape_iter=%d\n",
               orbit.length, orbit.escape_iter);
    }

    double sum_iter_diff = 0;
    int diff_count = 0;

    for (int y = 0; y < config.height; y++) {
        double dy = (y - config.height / 2.0) / config.height * scale;

        for (int x = 0; x < config.width; x++) {
            double dx = (x - config.width / 2.0) / config.width * scale * aspect;

            // Rotated offset from center
            double offset_x = dx * cos_a - dy * sin_a;
            double offset_y = dx * sin_a + dy * cos_a;

            // Pixel coordinate (DD precision)
            DD px = dd_add(config.center_x, offset_x);
            DD py = dd_add(config.center_y, offset_y);

            // Delta from reference center
            double dCr = offset_x;
            double dCi = offset_y;

            // Compute both ways
            PixelResult direct = compute_direct_dd(px, py, config.max_iter);
            PixelResult perturb = compute_perturbation_pixel(dCr, dCi, orbit, config.max_iter);

            // Collect stats
            if (direct.escaped) results.direct_escaped++;
            else results.direct_bounded++;

            if (perturb.glitched) {
                results.perturb_glitched++;
            } else if (perturb.escaped) {
                results.perturb_escaped++;
            } else {
                results.perturb_bounded++;
            }

            // Compare
            if (perturb.glitched) {
                // Glitched pixels are mismatches (need recovery)
                results.mismatches++;
            } else if (direct.escaped != perturb.escaped) {
                results.mismatches++;
                if (verbose && results.mismatches <= 10) {
                    printf("MISMATCH at (%d,%d): direct=%s iter=%d, perturb=%s iter=%d\n",
                           x, y,
                           direct.escaped ? "ESC" : "BND", direct.raw_iter,
                           perturb.escaped ? "ESC" : "BND", perturb.raw_iter);
                }
            } else {
                results.matches++;
                if (direct.escaped && perturb.escaped) {
                    double diff = fabs(direct.iteration - perturb.iteration);
                    sum_iter_diff += diff;
                    diff_count++;
                    if (diff > results.max_iter_diff) {
                        results.max_iter_diff = diff;
                    }
                }
            }
        }
    }

    results.avg_iter_diff = diff_count > 0 ? sum_iter_diff / diff_count : 0;
}

void print_results(const char* name, const TestConfig& config, const TestResults& r) {
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("TEST: %s\n", name);
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("Config: zoom=%.2e, angle=%.1f°, size=%dx%d, max_iter=%d\n",
           config.zoom, config.angle_deg, config.width, config.height, config.max_iter);
    printf("Center: %.16g + %.16gi\n", config.center_x.hi, config.center_y.hi);
    printf("\n");

    printf("Direct DD results:\n");
    printf("  Escaped: %d (%.1f%%)\n", r.direct_escaped,
           100.0 * r.direct_escaped / r.total_pixels);
    printf("  Bounded: %d (%.1f%%)\n", r.direct_bounded,
           100.0 * r.direct_bounded / r.total_pixels);
    printf("\n");

    printf("Perturbation results:\n");
    printf("  Escaped:  %d (%.1f%%)\n", r.perturb_escaped,
           100.0 * r.perturb_escaped / r.total_pixels);
    printf("  Bounded:  %d (%.1f%%)\n", r.perturb_bounded,
           100.0 * r.perturb_bounded / r.total_pixels);
    printf("  Glitched: %d (%.1f%%)\n", r.perturb_glitched,
           100.0 * r.perturb_glitched / r.total_pixels);
    printf("\n");

    printf("Comparison:\n");
    printf("  Matches:    %d (%.1f%%)\n", r.matches,
           100.0 * r.matches / r.total_pixels);
    printf("  Mismatches: %d (%.1f%%)\n", r.mismatches,
           100.0 * r.mismatches / r.total_pixels);
    printf("  Max iter diff:  %.3f\n", r.max_iter_diff);
    printf("  Avg iter diff:  %.3f\n", r.avg_iter_diff);
    printf("\n");

    bool pass = r.mismatches == 0 ||
                (100.0 * r.matches / r.total_pixels >= 99.0);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
}

int main(int argc, char* argv[]) {
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }

    printf("PERTURBATION REGRESSION TEST\n");
    printf("============================\n");
    printf("Testing perturbation vs direct DD iteration at deep zoom.\n");
    if (verbose) printf("Verbose mode: ON\n");

    TestResults results;

    // Test 1: The exact failing view (zoom 2.79e+11)
    {
        TestConfig config;
        config.center_x = DD(-0.7115114743);
        config.center_y = DD(-0.3078112463);
        config.zoom = 2.79e+11;
        config.angle_deg = -15.0;
        config.width = 160;
        config.height = 50;
        config.max_iter = 512;  // Higher for deep zoom

        run_test(config, results, verbose);
        print_results("Failing View (zoom 2.79e+11)", config, results);
    }

    // Test 2: Just before the transition (zoom 1.86e+11 - should work)
    {
        TestConfig config;
        config.center_x = DD(-0.7115114743);
        config.center_y = DD(-0.3078112463);
        config.zoom = 1.86e+11;
        config.angle_deg = -15.0;
        config.width = 160;
        config.height = 50;
        config.max_iter = 512;

        run_test(config, results, verbose);
        print_results("Pre-transition (zoom 1.86e+11)", config, results);
    }

    // Test 3: Even deeper zoom (zoom 3.18e+12)
    {
        TestConfig config;
        config.center_x = DD(-0.7115114743);
        config.center_y = DD(-0.3078112463);
        config.zoom = 3.18e+12;
        config.angle_deg = -15.0;
        config.width = 160;
        config.height = 50;
        config.max_iter = 1024;

        run_test(config, results, verbose);
        print_results("Deep Zoom (zoom 3.18e+12)", config, results);
    }

    // Test 4: No rotation (angle=0) at failing zoom
    {
        TestConfig config;
        config.center_x = DD(-0.7115114743);
        config.center_y = DD(-0.3078112463);
        config.zoom = 2.79e+11;
        config.angle_deg = 0.0;
        config.width = 160;
        config.height = 50;
        config.max_iter = 512;

        run_test(config, results, verbose);
        print_results("No Rotation (angle=0°)", config, results);
    }

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("All tests complete.\n");
    printf("════════════════════════════════════════════════════════════════\n");

    return 0;
}
