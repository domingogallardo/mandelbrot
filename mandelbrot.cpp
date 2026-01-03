/*
 * MANDELBROT EXPLORER
 * Ultra-fast console-based fractal explorer
 *
 * Controls:
 *   Arrow Keys     - Pan view
 *   SHIFT + Up/Down    - Zoom in/out
 *   SHIFT + Left/Right - Rotate color palette
 *   1-9            - Switch color schemes
 *   +/-            - Increase/decrease max iterations
 *   R              - Reset view
 *   Q/ESC          - Quit
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

#ifdef __AVX2__
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

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
    int max_iter = 256;
    int color_scheme = 0;
    double color_rotation = 0.0;

    int width = 80;
    int height = 24;

    std::vector<double> iterations;
    std::string output_buffer;

    std::atomic<bool> needs_redraw{true};
    std::atomic<bool> running{true};
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

    __m256d four = _mm256_set1_pd(4.0);
    __m256d one = _mm256_set1_pd(1.0);

    for (int y = start_row; y < end_row; y++) {
        double cy = state.center_y + (y - state.height / 2.0) / state.height * scale;

        for (int x = 0; x < state.width; x += 4) {
            __m256d cx = _mm256_set_pd(
                state.center_x + ((x + 3) - state.width / 2.0) / state.width * scale * aspect,
                state.center_x + ((x + 2) - state.width / 2.0) / state.width * scale * aspect,
                state.center_x + ((x + 1) - state.width / 2.0) / state.width * scale * aspect,
                state.center_x + (x - state.width / 2.0) / state.width * scale * aspect
            );
            __m256d cy_v = _mm256_set1_pd(cy);

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

    for (int y = start_row; y < end_row; y++) {
        double cy = state.center_y + (y - state.height / 2.0) / state.height * scale;

        for (int x = 0; x < state.width; x++) {
            double cx = state.center_x + (x - state.width / 2.0) / state.width * scale * aspect;

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
    char status[256];
    snprintf(status, sizeof(status),
        BOLD " ═══ MANDELBROT EXPLORER ═══ " RESET
        " │ Pos: %.6f%+.6fi │ Zoom: %.2e │ Iter: %d │ [%s] │ ←↑↓→:Move SHIFT+↑↓:Zoom SHIFT+←→:Rotate 1-9:Color R:Reset Q:Quit",
        state.center_x, state.center_y, state.zoom, state.max_iter,
        scheme_names[state.color_scheme]);
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
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(restore_terminal);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 1;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    terminal_raw = true;

    printf(ALT_BUFFER_ON CURSOR_HIDE CLEAR_SCREEN);
    fflush(stdout);
}

void handle_resize(MandelbrotState& state) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        state.width = ws.ws_col;
        state.height = (ws.ws_row - 1) * 2;  // -1 for status bar, *2 for half-blocks
        state.iterations.resize(state.width * state.height);
        state.needs_redraw = true;
    }
}

static MandelbrotState* g_state = nullptr;
static volatile sig_atomic_t resize_pending = 0;

void sigwinch_handler(int) {
    resize_pending = 1;
}

void sigint_handler(int) {
    if (g_state) {
        g_state->running = false;
    }
}

enum Key {
    KEY_NONE = 0,
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_SHIFT_UP, KEY_SHIFT_DOWN, KEY_SHIFT_LEFT, KEY_SHIFT_RIGHT,
    KEY_Q, KEY_R, KEY_ESC,
    KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9,
    KEY_PLUS, KEY_MINUS
};

Key read_key() {
    char c;
    if (read(STDIN_FILENO, &c, 1) != 1) return KEY_NONE;

    if (c == 'q' || c == 'Q') return KEY_Q;
    if (c == 'r' || c == 'R') return KEY_R;
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

    double move_amount = 0.1 / state.zoom;
    double zoom_factor = 1.5;
    double rotation_step = 0.05;

    switch (key) {
        case KEY_UP:    state.center_y -= move_amount; state.needs_redraw = true; break;
        case KEY_DOWN:  state.center_y += move_amount; state.needs_redraw = true; break;
        case KEY_LEFT:  state.center_x -= move_amount; state.needs_redraw = true; break;
        case KEY_RIGHT: state.center_x += move_amount; state.needs_redraw = true; break;

        case KEY_SHIFT_UP:    state.zoom *= zoom_factor; state.needs_redraw = true; break;
        case KEY_SHIFT_DOWN:  state.zoom /= zoom_factor; state.needs_redraw = true; break;
        case KEY_SHIFT_LEFT:  state.color_rotation -= rotation_step; state.needs_redraw = true; break;
        case KEY_SHIFT_RIGHT: state.color_rotation += rotation_step; state.needs_redraw = true; break;

        case KEY_PLUS:  state.max_iter = std::min(4096, state.max_iter + 64); state.needs_redraw = true; break;
        case KEY_MINUS: state.max_iter = std::max(64, state.max_iter - 64); state.needs_redraw = true; break;

        case KEY_1: case KEY_2: case KEY_3: case KEY_4: case KEY_5:
        case KEY_6: case KEY_7: case KEY_8: case KEY_9:
            state.color_scheme = key - KEY_1;
            state.needs_redraw = true;
            break;

        case KEY_R:
            state.center_x = -0.5;
            state.center_y = 0.0;
            state.zoom = 1.0;
            state.max_iter = 256;
            state.color_rotation = 0.0;
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
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    MandelbrotState state;
    g_state = &state;

    setup_terminal();

    signal(SIGWINCH, sigwinch_handler);
    signal(SIGINT, sigint_handler);

    handle_resize(state);

    printf(CLEAR_SCREEN);

    while (state.running) {
        // Handle resize outside signal handler for async-signal safety
        if (resize_pending) {
            resize_pending = 0;
            handle_resize(state);
        }

        if (state.needs_redraw) {
            state.needs_redraw = false;

            auto start = std::chrono::high_resolution_clock::now();
            compute_mandelbrot_threaded(state);
            auto end = std::chrono::high_resolution_clock::now();

            render_frame(state);
        }

        handle_input(state);

        if (!state.needs_redraw) {
            usleep(10000);  // 10ms sleep to reduce CPU usage when idle
        }
    }

    restore_terminal();

    printf("\n✨ Thanks for exploring the Mandelbrot set!\n");
    printf("   Compiled with %s optimization\n", USE_AVX2 ? "AVX2 SIMD" : "scalar");

    return 0;
}
