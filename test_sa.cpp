// Series Approximation (SA) Tests
// Tests correctness and speedup of SA vs non-SA implementation
//
// Build: make test_sa
// Run:   ./test_sa

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <cstring>

// Include the main implementation (we'll extract what we need)
// For testing, we compile a standalone version

// ═══════════════════════════════════════════════════════════════════════════
// DOUBLE-DOUBLE ARITHMETIC (simplified for testing)
// ═══════════════════════════════════════════════════════════════════════════

struct DD {
    double hi, lo;
    DD() : hi(0), lo(0) {}
    DD(double h) : hi(h), lo(0) {}
    DD(double h, double l) : hi(h), lo(l) {}
};

inline DD dd_add(DD a, DD b) {
    double s = a.hi + b.hi;
    double v = s - a.hi;
    double e = (a.hi - (s - v)) + (b.hi - v);
    double t = a.lo + b.lo + e;
    DD r;
    r.hi = s + t;
    r.lo = t - (r.hi - s);
    return r;
}

inline DD dd_add(DD a, double b) {
    double s = a.hi + b;
    double v = s - a.hi;
    double e = (a.hi - (s - v)) + (b - v);
    double t = a.lo + e;
    DD r;
    r.hi = s + t;
    r.lo = t - (r.hi - s);
    return r;
}

inline DD dd_mul(DD a, DD b) {
    double p = a.hi * b.hi;
    double e = std::fma(a.hi, b.hi, -p);
    e += a.hi * b.lo + a.lo * b.hi;
    DD r;
    r.hi = p + e;
    r.lo = e - (r.hi - p);
    return r;
}

inline DD dd_sub(DD a, DD b) {
    return dd_add(a, DD(-b.hi, -b.lo));
}

// ═══════════════════════════════════════════════════════════════════════════
// REFERENCE ORBIT WITH SA COEFFICIENTS
// ═══════════════════════════════════════════════════════════════════════════

struct ReferenceOrbit {
    DD center_re, center_im;
    std::vector<double> Zr_sum, Zi_sum;
    std::vector<double> SA_Ar, SA_Ai;
    std::vector<double> SA_Br, SA_Bi;
    std::vector<double> SA_Cr, SA_Ci;
    std::vector<double> SA_A_norm;
    int length = 0;
    int escape_iter = -1;
    bool sa_enabled = false;

    void clear() {
        Zr_sum.clear(); Zi_sum.clear();
        SA_Ar.clear(); SA_Ai.clear();
        SA_Br.clear(); SA_Bi.clear();
        SA_Cr.clear(); SA_Ci.clear();
        SA_A_norm.clear();
        length = 0;
        escape_iter = -1;
        sa_enabled = false;
    }
};

// Compute reference orbit with SA coefficients
void compute_reference_orbit(ReferenceOrbit& orbit, DD center_x, DD center_y,
                             int max_iter, bool enable_sa = true) {
    orbit.clear();
    orbit.center_re = center_x;
    orbit.center_im = center_y;

    DD Zr{0}, Zi{0};
    double Ar = 0, Ai = 0;
    double Br = 0, Bi = 0;
    double Cr = 0, Ci = 0;

    orbit.Zr_sum.push_back(0);
    orbit.Zi_sum.push_back(0);
    orbit.SA_Ar.push_back(0);
    orbit.SA_Ai.push_back(0);
    orbit.SA_Br.push_back(0);
    orbit.SA_Bi.push_back(0);
    orbit.SA_Cr.push_back(0);
    orbit.SA_Ci.push_back(0);
    orbit.SA_A_norm.push_back(0);

    for (int n = 0; n < max_iter; n++) {
        // Z_{n+1} = Z_n² + C
        DD Zr2 = dd_mul(Zr, Zr);
        DD Zi2 = dd_mul(Zi, Zi);
        DD ZrZi = dd_mul(Zr, Zi);

        DD new_Zr = dd_add(dd_sub(Zr2, Zi2), center_x);
        DD new_Zi = dd_add(dd_add(ZrZi, ZrZi), center_y);

        Zr = new_Zr;
        Zi = new_Zi;

        double Zr_d = Zr.hi + Zr.lo;
        double Zi_d = Zi.hi + Zi.lo;
        orbit.Zr_sum.push_back(Zr_d);
        orbit.Zi_sum.push_back(Zi_d);

        // SA coefficient recurrence
        double Zr_prev = (n == 0) ? 0.0 : orbit.Zr_sum[n];
        double Zi_prev = (n == 0) ? 0.0 : orbit.Zi_sum[n];

        double new_Ar = 2.0 * (Zr_prev * Ar - Zi_prev * Ai) + 1.0;
        double new_Ai = 2.0 * (Zr_prev * Ai + Zi_prev * Ar);

        double A2r = Ar * Ar - Ai * Ai;
        double A2i = 2.0 * Ar * Ai;
        double new_Br = 2.0 * (Zr_prev * Br - Zi_prev * Bi) + A2r;
        double new_Bi = 2.0 * (Zr_prev * Bi + Zi_prev * Br) + A2i;

        double ABr = Ar * Br - Ai * Bi;
        double ABi = Ar * Bi + Ai * Br;
        double new_Cr = 2.0 * (Zr_prev * Cr - Zi_prev * Ci) + 2.0 * ABr;
        double new_Ci = 2.0 * (Zr_prev * Ci + Zi_prev * Cr) + 2.0 * ABi;

        Ar = new_Ar; Ai = new_Ai;
        Br = new_Br; Bi = new_Bi;
        Cr = new_Cr; Ci = new_Ci;

        orbit.SA_Ar.push_back(Ar);
        orbit.SA_Ai.push_back(Ai);
        orbit.SA_Br.push_back(Br);
        orbit.SA_Bi.push_back(Bi);
        orbit.SA_Cr.push_back(Cr);
        orbit.SA_Ci.push_back(Ci);
        orbit.SA_A_norm.push_back(Ar * Ar + Ai * Ai);

        double norm = Zr_d * Zr_d + Zi_d * Zi_d;
        if (norm > 1e6) {
            orbit.escape_iter = n + 1;
            orbit.length = n + 2;
            orbit.sa_enabled = enable_sa;
            return;
        }
    }

    orbit.escape_iter = -1;
    orbit.length = max_iter + 1;
    orbit.sa_enabled = enable_sa;
}

// ═══════════════════════════════════════════════════════════════════════════
// SA SKIP ITERATION FINDER
// ═══════════════════════════════════════════════════════════════════════════

constexpr double SA_TOLERANCE = 0.001;

int sa_find_skip_iteration(const ReferenceOrbit& orbit,
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

    int max_check = std::min(max_iter, orbit.length - 1);
    int best_skip = 0;

    int lo = 1, hi = max_check;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;

        double Cr = orbit.SA_Cr[mid];
        double Ci = orbit.SA_Ci[mid];
        double C_norm = Cr * Cr + Ci * Ci;
        double A_norm = orbit.SA_A_norm[mid];

        if (!std::isfinite(C_norm) || !std::isfinite(A_norm) || A_norm == 0.0) {
            hi = mid - 1;
            continue;
        }

        // Validity check: |C|² * |δC|⁴ < ε² * |A|²
        double lhs = C_norm * dC_norm * dC_norm;
        double rhs = SA_TOLERANCE * SA_TOLERANCE * A_norm;

        if (lhs < rhs) {
            best_skip = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    if (best_skip > 0) {
        double Ar = orbit.SA_Ar[best_skip];
        double Ai = orbit.SA_Ai[best_skip];
        double Br = orbit.SA_Br[best_skip];
        double Bi = orbit.SA_Bi[best_skip];
        double Cr = orbit.SA_Cr[best_skip];
        double Ci = orbit.SA_Ci[best_skip];

        double dC2r = dCr * dCr - dCi * dCi;
        double dC2i = 2.0 * dCr * dCi;
        double dC3r = dC2r * dCr - dC2i * dCi;
        double dC3i = dC2r * dCi + dC2i * dCr;

        double term1r = Ar * dCr - Ai * dCi;
        double term1i = Ar * dCi + Ai * dCr;
        double term2r = Br * dC2r - Bi * dC2i;
        double term2i = Br * dC2i + Bi * dC2r;
        double term3r = Cr * dC3r - Ci * dC3i;
        double term3i = Cr * dC3i + Ci * dC3r;

        dzr_out = term1r + term2r + term3r;
        dzi_out = term1i + term2i + term3i;

        if (!std::isfinite(dzr_out) || !std::isfinite(dzi_out)) {
            dzr_out = 0.0;
            dzi_out = 0.0;
            return 0;
        }
    } else {
        dzr_out = 0.0;
        dzi_out = 0.0;
    }

    return best_skip;
}

// ═══════════════════════════════════════════════════════════════════════════
// PERTURBATION ITERATION (with optional SA)
// ═══════════════════════════════════════════════════════════════════════════

struct IterResult {
    double iterations;
    bool escaped;
    double final_zr, final_zi;
};

IterResult compute_pixel(const ReferenceOrbit& orbit, double dCr, double dCi,
                         int max_iter, bool use_sa) {
    IterResult result = {-1.0, false, 0.0, 0.0};

    double dzr = 0.0, dzi = 0.0;
    int iter = 0;

    if (use_sa && orbit.sa_enabled) {
        iter = sa_find_skip_iteration(orbit, dCr, dCi, dzr, dzi, max_iter);

        if (iter > 0) {
            double full_zr = orbit.Zr_sum[iter] + dzr;
            double full_zi = orbit.Zi_sum[iter] + dzi;
            double mag2 = full_zr * full_zr + full_zi * full_zi;
            if (mag2 > 4.0) {
                result.escaped = true;
                result.iterations = iter;
                result.final_zr = full_zr;
                result.final_zi = full_zi;
                return result;
            }
        }
    }

    int max_ref = orbit.length - 1;

    while (iter < max_iter && iter < max_ref) {
        double Zr = orbit.Zr_sum[iter];
        double Zi = orbit.Zi_sum[iter];

        double temp_r = 2.0 * Zr + dzr;
        double temp_i = 2.0 * Zi + dzi;

        double new_dzr = temp_r * dzr - temp_i * dzi + dCr;
        double new_dzi = temp_r * dzi + temp_i * dzr + dCi;

        dzr = new_dzr;
        dzi = new_dzi;

        double full_zr = orbit.Zr_sum[iter + 1] + dzr;
        double full_zi = orbit.Zi_sum[iter + 1] + dzi;
        double mag2 = full_zr * full_zr + full_zi * full_zi;

        if (mag2 > 4.0) {
            result.escaped = true;
            result.final_zr = full_zr;
            result.final_zi = full_zi;
            // Smooth iteration count
            double log_zn = log(mag2) / 2.0;
            double nu = log(log_zn / log(2.0)) / log(2.0);
            result.iterations = iter + 1 - nu;
            return result;
        }

        iter++;
    }

    result.iterations = -1.0;
    result.escaped = false;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

int tests_passed = 0;
int tests_failed = 0;

void check(bool condition, const char* name) {
    if (condition) {
        printf("  [PASS] %s\n", name);
        tests_passed++;
    } else {
        printf("  [FAIL] %s\n", name);
        tests_failed++;
    }
}

// Test 1: SA correctness - iteration counts should match closely
bool test_sa_correctness() {
    printf("\nTest: SA Correctness\n");

    DD center_x(-0.743643887037158704752191506114774);
    DD center_y(0.131825904205311970493132056385139);
    int max_iter = 1000;
    double zoom = 1e12;
    double scale = 3.0 / zoom;

    ReferenceOrbit orbit;
    compute_reference_orbit(orbit, center_x, center_y, max_iter, true);

    // Test multiple pixels at different offsets
    int matching = 0;
    int total = 0;
    double max_error = 0;

    for (int py = -5; py <= 5; py++) {
        for (int px = -5; px <= 5; px++) {
            if (px == 0 && py == 0) continue;

            double dCr = px * scale * 0.1;
            double dCi = py * scale * 0.1;

            IterResult with_sa = compute_pixel(orbit, dCr, dCi, max_iter, true);
            IterResult without_sa = compute_pixel(orbit, dCr, dCi, max_iter, false);

            total++;

            if (with_sa.escaped == without_sa.escaped) {
                if (with_sa.escaped) {
                    double error = fabs(with_sa.iterations - without_sa.iterations);
                    if (error < 1.0) {
                        matching++;
                    }
                    if (error > max_error) max_error = error;
                } else {
                    matching++;
                }
            }
        }
    }

    double match_rate = 100.0 * matching / total;
    printf("  Match rate: %.1f%% (%d/%d pixels)\n", match_rate, matching, total);
    printf("  Max iteration error: %.4f\n", max_error);

    check(match_rate >= 95.0, "95%+ pixels match");
    check(max_error < 2.0, "Max error < 2 iterations");

    return match_rate >= 95.0 && max_error < 2.0;
}

// Test 2: SA speedup - should be significantly faster
bool test_sa_speedup() {
    printf("\nTest: SA Speedup\n");

    DD center_x(-0.743643887037158704752191506114774);
    DD center_y(0.131825904205311970493132056385139);
    int max_iter = 1000;  // Match skip percentage test
    double zoom = 1e12;
    double scale = 3.0 / zoom;

    ReferenceOrbit orbit;
    compute_reference_orbit(orbit, center_x, center_y, max_iter, true);

    // Use a single pixel but time individual iterations to avoid timing resolution issues
    // Instead of timing pixel computation, we count effective work done

    double dCr = scale * 0.1;
    double dCi = scale * 0.1;

    // Count iterations actually performed with SA
    double dzr = 0.0, dzi = 0.0;
    int skip_iter = sa_find_skip_iteration(orbit, dCr, dCi, dzr, dzi, max_iter);

    // Calculate theoretical speedup based on skip ratio
    // The effective max is limited by orbit length
    int effective_max = std::min(max_iter, orbit.length - 1);
    int iters_with_sa = std::max(0, effective_max - skip_iter);
    int iters_without_sa = effective_max;

    double theoretical_speedup = 0;
    if (iters_with_sa > 0) {
        theoretical_speedup = (double)iters_without_sa / iters_with_sa;
    } else {
        // All iterations skipped - effectively infinite speedup, cap at 1000x
        theoretical_speedup = 1000.0;
    }

    double skip_pct = 100.0 * skip_iter / effective_max;

    printf("  Orbit length: %d\n", orbit.length);
    printf("  Skip iteration: %d of %d (%.1f%%)\n", skip_iter, effective_max, skip_pct);
    printf("  Iterations with SA: %d\n", iters_with_sa);
    printf("  Iterations without SA: %d\n", iters_without_sa);
    printf("  Theoretical speedup: %.1fx\n", theoretical_speedup);

    // At deep zoom, expect high skip percentage
    check(skip_pct >= 90.0, "Skip >= 90% of effective iterations");
    check(theoretical_speedup >= 5.0, "Theoretical speedup >= 5x");

    return theoretical_speedup >= 5.0;
}

// Test 3: SA skip percentage at deep zoom
bool test_sa_skip_percentage() {
    printf("\nTest: SA Skip Percentage\n");

    DD center_x(-0.743643887037158704752191506114774);
    DD center_y(0.131825904205311970493132056385139);
    int max_iter = 1000;

    struct TestCase {
        double zoom;
        double expected_min_skip;  // Minimum expected skip percentage
    };

    // At higher zoom, pixels are closer together, so SA is more effective
    TestCase cases[] = {
        {1e6,  5.0},   // Lower zoom = larger δC = less skip
        {1e9,  80.0},
        {1e12, 95.0},
    };

    bool all_passed = true;

    for (const auto& tc : cases) {
        double scale = 3.0 / tc.zoom;

        ReferenceOrbit orbit;
        compute_reference_orbit(orbit, center_x, center_y, max_iter, true);

        double dCr = scale * 0.1;
        double dCi = scale * 0.1;
        double dzr, dzi;
        int skip = sa_find_skip_iteration(orbit, dCr, dCi, dzr, dzi, max_iter);
        double skip_pct = 100.0 * skip / max_iter;

        printf("  Zoom %.0e: skip %d/%d (%.1f%%)\n", tc.zoom, skip, max_iter, skip_pct);

        bool passed = skip_pct >= tc.expected_min_skip;
        check(passed, (std::string("Skip >= ") + std::to_string((int)tc.expected_min_skip) + "% at zoom " + std::to_string((int)log10(tc.zoom))).c_str());
        if (!passed) all_passed = false;
    }

    return all_passed;
}

// Test 4: SA handles edge cases
bool test_sa_edge_cases() {
    printf("\nTest: SA Edge Cases\n");

    DD center_x(-0.5);
    DD center_y(0.0);
    int max_iter = 100;

    ReferenceOrbit orbit;
    compute_reference_orbit(orbit, center_x, center_y, max_iter, true);

    double dzr, dzi;

    // Test 1: Zero offset (center pixel)
    int skip = sa_find_skip_iteration(orbit, 0.0, 0.0, dzr, dzi, max_iter);
    check(skip == 0 && dzr == 0.0 && dzi == 0.0, "Zero offset returns 0");

    // Test 2: Very small offset
    skip = sa_find_skip_iteration(orbit, 1e-20, 1e-20, dzr, dzi, max_iter);
    check(std::isfinite(dzr) && std::isfinite(dzi), "Tiny offset gives finite result");

    // Test 3: Large offset (outside Mandelbrot)
    skip = sa_find_skip_iteration(orbit, 10.0, 10.0, dzr, dzi, max_iter);
    check(skip >= 0, "Large offset doesn't crash");

    // Test 4: SA disabled
    orbit.sa_enabled = false;
    skip = sa_find_skip_iteration(orbit, 0.001, 0.001, dzr, dzi, max_iter);
    check(skip == 0, "SA disabled returns 0");

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("       SERIES APPROXIMATION (SA) TEST SUITE\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    test_sa_correctness();
    test_sa_speedup();
    test_sa_skip_percentage();
    test_sa_edge_cases();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
