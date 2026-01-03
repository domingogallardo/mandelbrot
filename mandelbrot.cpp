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
 *   R                   - Reset view
 *   Q/ESC               - Quit
 *
 * CLI Arguments:
 *   --pos <re+imi>      - Starting position (e.g., -0.5+0.0i)
 *   --zoom <value>      - Starting zoom level
 *   --angle <degrees>   - Starting view angle
 *   --auto              - Automatic exploration mode
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
inline void compute_reference_orbit(ReferenceOrbit& orbit,
                                    DD center_x, DD center_y,
                                    int max_iter) {
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

        // Precompute sums for fast access in perturbation loop
        double zr_sum = Z.re.hi + Z.re.lo;
        double zi_sum = Z.im.hi + Z.im.lo;
        orbit.Zr_sum.push_back(zr_sum);
        orbit.Zi_sum.push_back(zi_sum);

        // Compute norm from full-precision sums
        double norm = zr_sum * zr_sum + zi_sum * zi_sum;
        orbit.Z_norm.push_back(norm);

        // Use larger escape radius for reference (extends orbit length)
        if (norm > 1e6) {
            orbit.escape_iter = n + 1;
            orbit.length = n + 2;
            return;
        }
    }

    orbit.escape_iter = -1;  // Didn't escape
    orbit.length = max_iter + 1;
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
    ReferenceOrbit primary_orbit;  // Primary reference orbit
    std::vector<std::pair<int,int>> glitch_pixels;  // Pixels needing recompute

    // Sync DD and double centers
    void sync_centers_to_dd() {
        center_x = center_x_dd.hi;
        center_y = center_y_dd.hi;
    }
    void sync_centers_from_double() {
        center_x_dd = DD(center_x);
        center_y_dd = DD(center_y);
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

// Scalar perturbation computation
void compute_perturbation_scalar(MandelbrotState& state,
                                 const ReferenceOrbit& orbit,
                                 int start_row, int end_row,
                                 bool glitch_only = false) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);

    // Use full DD precision for reference center (hi + lo)
    double ref_cr = orbit.center_re.hi + orbit.center_re.lo;
    double ref_ci = orbit.center_im.hi + orbit.center_im.lo;

    int max_ref_iter = orbit.length - 1;

    for (int y = start_row; y < end_row; y++) {
        double dy = (y - state.height / 2.0) / state.height * scale;

        for (int x = 0; x < state.width; x++) {
            int idx = y * state.width + x;

            // Skip if doing glitch-only pass and this isn't a glitch
            if (glitch_only && state.iterations[idx] != -2.0) continue;

            // Compute rotated pixel offset from center
            double dx = (x - state.width / 2.0) / state.width * scale * aspect;

            // δC = offset from reference center (computed directly to preserve precision)
            // At deep zoom, adding ref_cr + offset then subtracting ref_cr loses precision!
            double dCr = dx * cos_a - dy * sin_a;
            double dCi = dx * sin_a + dy * cos_a;

            // Perturbation iteration
            double dzr = 0.0, dzi = 0.0;
            int iter = 0;
            bool escaped = false;
            bool glitched = false;
            double final_zr = 0, final_zi = 0;

            while (iter < state.max_iter && iter < max_ref_iter) {
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
                DD cx_dd = dd_add(state.center_x_dd, dCr);
                DD cy_dd = dd_add(state.center_y_dd, dCi);
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
        }
    }
}

#if USE_AVX2
// AVX2 perturbation computation
void compute_perturbation_avx2(MandelbrotState& state,
                               const ReferenceOrbit& orbit,
                               int start_row, int end_row) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);

    // Use full DD precision for reference center (hi + lo)
    double ref_cr = orbit.center_re.hi + orbit.center_re.lo;
    double ref_ci = orbit.center_im.hi + orbit.center_im.lo;

    __m256d four = _mm256_set1_pd(4.0);
    __m256d two = _mm256_set1_pd(2.0);
    __m256d one = _mm256_set1_pd(1.0);
    __m256d near_escape_thresh = _mm256_set1_pd(NEAR_ESCAPE_THRESHOLD);
    __m256d floor_val = _mm256_set1_pd(GLITCH_FLOOR);

    int max_ref_iter = orbit.length - 1;

    for (int y = start_row; y < end_row; y++) {
        double dy = (y - state.height / 2.0) / state.height * scale;

        for (int x = 0; x < state.width; x += 4) {
            // Compute rotated δC for 4 pixels
            double dx0 = (x - state.width / 2.0) / state.width * scale * aspect;
            double dx1 = ((x + 1) - state.width / 2.0) / state.width * scale * aspect;
            double dx2 = ((x + 2) - state.width / 2.0) / state.width * scale * aspect;
            double dx3 = ((x + 3) - state.width / 2.0) / state.width * scale * aspect;

            __m256d dC_r = _mm256_set_pd(
                dx3 * cos_a - dy * sin_a,
                dx2 * cos_a - dy * sin_a,
                dx1 * cos_a - dy * sin_a,
                dx0 * cos_a - dy * sin_a
            );
            __m256d dC_i = _mm256_set_pd(
                dx3 * sin_a + dy * cos_a,
                dx2 * sin_a + dy * cos_a,
                dx1 * sin_a + dy * cos_a,
                dx0 * sin_a + dy * cos_a
            );

            __m256d dzr = _mm256_setzero_pd();
            __m256d dzi = _mm256_setzero_pd();
            __m256d iter = _mm256_setzero_pd();
            __m256d active = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));  // All true

            // For smooth coloring
            __m256d escaped_zr = _mm256_setzero_pd();
            __m256d escaped_zi = _mm256_setzero_pd();
            __m256d has_escaped = _mm256_setzero_pd();
            __m256d is_glitched = _mm256_setzero_pd();

            int max_n = std::min(state.max_iter, max_ref_iter);

            for (int n = 0; n < max_n; n++) {
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
            double dzr_arr[4], dzi_arr[4];
            _mm256_storeu_pd(iter_arr, iter);
            _mm256_storeu_pd(zr_arr, escaped_zr);
            _mm256_storeu_pd(zi_arr, escaped_zi);
            _mm256_storeu_pd(escaped_arr, has_escaped);
            _mm256_storeu_pd(glitched_arr, is_glitched);
            _mm256_storeu_pd(dzr_arr, dzr);
            _mm256_storeu_pd(dzi_arr, dzi);

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
                        double zr = orbit.Zr_sum[max_ref_iter] + dzr_arr[i];
                        double zi = orbit.Zi_sum[max_ref_iter] + dzi_arr[i];

                        // Compute δC for this pixel (offset from center)
                        double dxi = ((x + i) - state.width / 2.0) / state.width * scale * aspect;
                        double dCr = dxi * cos_a - dy * sin_a;
                        double dCi = dxi * sin_a + dy * cos_a;

                        // Get C using DD arithmetic to preserve precision at deep zoom
                        DD cx_dd = dd_add(state.center_x_dd, dCr);
                        DD cy_dd = dd_add(state.center_y_dd, dCi);
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

// Convert pixel to DD coordinates (with rotation)
DD pixel_to_dd_x(int px, int py, const MandelbrotState& state) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);
    double dx = (px - state.width / 2.0) / state.width * scale * aspect;
    double dy = (py - state.height / 2.0) / state.height * scale;
    double offset = dx * cos_a - dy * sin_a;
    return dd_add(state.center_x_dd, offset);
}

DD pixel_to_dd_y(int px, int py, const MandelbrotState& state) {
    double aspect = (double)state.width / (state.height * 2.0);
    double scale = 3.0 / state.zoom;
    double cos_a = cos(state.angle);
    double sin_a = sin(state.angle);
    double dx = (px - state.width / 2.0) / state.width * scale * aspect;
    double dy = (py - state.height / 2.0) / state.height * scale;
    double offset = dx * sin_a + dy * cos_a;
    return dd_add(state.center_y_dd, offset);
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
        compute_reference_orbit(new_orbit, new_center_x, new_center_y, state.max_iter);

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

            // Use DD centers to preserve low bits at deep zoom.
            DD ppx = dd_add(state.center_x_dd, offset_x);
            DD ppy = dd_add(state.center_y_dd, offset_y);
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

                // Compute C using DD arithmetic for precision
                DD ppx = dd_add(state.center_x_dd, offset_x);
                DD ppy = dd_add(state.center_y_dd, offset_y);
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
        state.use_perturbation = false;
        compute_mandelbrot_threaded(state);
    } else {
        // Perturbation path
        bool was_perturbation = state.use_perturbation;
        state.use_perturbation = true;

        // Sync DD centers from double on mode transition
        // This ensures DD centers match double centers if user panned/zoomed
        // before entering perturbation mode
        if (!was_perturbation) {
            state.sync_centers_from_double();
        }

        // Auto-scale iterations for deep zoom (deep areas need more iterations)
        // This applies even in manual mode, not just auto-explore
        int min_iter_for_zoom = 256 + (int)(log10(std::max(1.0, state.zoom)) * 64);
        int effective_max_iter = std::max(state.max_iter, std::min(4096, min_iter_for_zoom));

        // Compute reference orbit at screen center
        compute_reference_orbit(state.primary_orbit,
                               state.center_x_dd, state.center_y_dd,
                               effective_max_iter);

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

void render_frame(MandelbrotState& state) {
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

    // Status bar
    char status[640];
    const char* mode_str = state.use_perturbation ? "PERTURB" : "DOUBLE";
    double angle_deg = state.angle * 180.0 / M_PI;
    snprintf(status, sizeof(status),
        BOLD " ═══ MANDELBROT EXPLORER ═══ " RESET
        " │ Pos: %.10g%+.10gi │ Zoom: %.2e │ Angle: %.1f° │ Iter: %d │ [%s] │ %s",
        state.center_x, state.center_y, state.zoom, angle_deg, state.max_iter,
        scheme_names[state.color_scheme], mode_str);
    out += status;

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
    KEY_C, KEY_V
};

Key read_key() {
    char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return KEY_NONE;

    if (c == 'q' || c == 'Q') return KEY_Q;
    if (c == 'r' || c == 'R') return KEY_R;
    if (c == 'c' || c == 'C') return KEY_C;
    if (c == 'v' || c == 'V') return KEY_V;
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

    // Use DD arithmetic for movement when in perturbation mode
    bool use_dd = needs_perturbation(state);

    switch (key) {
        case KEY_UP:
            if (use_dd) {
                DD move = dd_div(DD(0.1), state.zoom);
                state.center_y_dd = dd_sub(state.center_y_dd, move);
                state.sync_centers_to_dd();
            } else {
                state.center_y -= 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_DOWN:
            if (use_dd) {
                DD move = dd_div(DD(0.1), state.zoom);
                state.center_y_dd = dd_add(state.center_y_dd, move);
                state.sync_centers_to_dd();
            } else {
                state.center_y += 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_LEFT:
            if (use_dd) {
                DD move = dd_div(DD(0.1), state.zoom);
                state.center_x_dd = dd_sub(state.center_x_dd, move);
                state.sync_centers_to_dd();
            } else {
                state.center_x -= 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;
        case KEY_RIGHT:
            if (use_dd) {
                DD move = dd_div(DD(0.1), state.zoom);
                state.center_x_dd = dd_add(state.center_x_dd, move);
                state.sync_centers_to_dd();
            } else {
                state.center_x += 0.1 / state.zoom;
                state.sync_centers_from_double();
            }
            state.needs_redraw = true;
            break;

        case KEY_SHIFT_UP:
            state.zoom *= zoom_factor;
            state.needs_redraw = true;
            break;
        case KEY_SHIFT_DOWN:
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
            state.zoom = 1.0;
            state.angle = 0.0;
            state.max_iter = 256;
            state.color_rotation = 0.0;
            state.use_perturbation = false;
            state.needs_redraw = true;
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

struct AutoExplorer {
    bool enabled = false;
    double target_zoom = 1e14;           // How deep to zoom
    double zoom_factor = 1.008;          // Per-frame zoom multiplier (~60 fps)
    double rotation_speed = 0.3;         // Degrees per frame
    double current_rotation_dir = 1.0;   // 1 or -1
    std::mt19937 rng;

    AutoExplorer() : rng(std::random_device{}()) {}
};

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

// Update auto exploration state
void update_auto_exploration(MandelbrotState& state, AutoExplorer& explorer) {
    if (!explorer.enabled) return;

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

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --pos <re+imi>     Starting position (e.g., -0.5+0.3i)\n");
    printf("  --zoom <value>     Starting zoom level (e.g., 1e6)\n");
    printf("  --angle <degrees>  Starting view angle (e.g., 45)\n");
    printf("  --auto             Enable automatic exploration mode\n");
    printf("  --help             Show this help message\n");
    printf("\nControls:\n");
    printf("  Arrow Keys          - Pan view\n");
    printf("  SHIFT + Up/Down     - Zoom in/out\n");
    printf("  SHIFT + Left/Right  - Rotate view\n");
    printf("  C/V                 - Rotate color palette\n");
    printf("  1-9                 - Switch color schemes\n");
    printf("  +/-                 - Adjust max iterations\n");
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
    static struct option long_options[] = {
        {"pos",   required_argument, 0, 'p'},
        {"zoom",  required_argument, 0, 'z'},
        {"angle", required_argument, 0, 'a'},
        {"auto",  no_argument,       0, 'A'},
        {"help",  no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:z:a:Ah", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'p': {
                double re, im;
                if (parse_complex(optarg, re, im)) {
                    state.center_x = re;
                    state.center_y = im;
                    state.center_x_dd = DD(re);
                    state.center_y_dd = DD(im);
                } else {
                    fprintf(stderr, "Error: Invalid position format '%s'\n", optarg);
                    fprintf(stderr, "Expected format: re+imi (e.g., -0.5+0.3i)\n");
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
                state.angle = angle_deg * M_PI / 180.0;
                break;
            }
            case 'A':
                explorer.enabled = true;
                // In auto mode, start with finding an interesting point
                {
                    auto [new_x, new_y] = find_interesting_point(explorer);
                    state.center_x = new_x;
                    state.center_y = new_y;
                    state.center_x_dd = DD(new_x);
                    state.center_y_dd = DD(new_y);
                }
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
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

    // Print CLI command to return to this location
    double angle_deg = state.angle * 180.0 / M_PI;
    printf("\n");
    printf("To return to this location:\n");
    printf("  ./mandelbrot --pos %.17g%+.17gi --zoom %.17g --angle %.6f\n",
           state.center_x, state.center_y, state.zoom, angle_deg);
    printf("\n");
    printf("Thanks for exploring the Mandelbrot set!\n");
    printf("Compiled with %s optimization\n", USE_AVX2 ? "AVX2 SIMD" : "scalar");

    return 0;
}
