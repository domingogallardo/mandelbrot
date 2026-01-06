# Mandelbrot Explorer

Ultra-fast console-based Mandelbrot fractal explorer with deep zoom capabilities using perturbation theory and double-double arithmetic.

![Mandelbrot Animation](assets/mandelbrot.gif)

## Features

- **Deep Zoom**: Explore to zoom levels beyond 10^30 using perturbation theory
- **Series Approximation**: Significant speedup at deep zoom via 4-term polynomial approximation
- **Double-Double Precision**: ~31 decimal digits of precision for reference orbit computation
- **AVX2 SIMD**: Optional 4x parallel pixel computation on supported CPUs
- **Smooth Animation**: Trajectory mode with ease-in-out cubic easing
- **Multiple Color Schemes**: 9 built-in palettes with rotation support
- **Real-time Navigation**: Pan, zoom, and rotate interactively
- **iTerm2 Image Mode**: High-resolution rendering using iTerm2 inline images

## Building

```bash
# Standard build (portable)
make

# With AVX2 optimization (faster on supported CPUs)
make avx2

# Native optimizations for current CPU
make native
```

## Usage

```bash
# Interactive mode
./mandelbrot

# Start at specific position and zoom
./mandelbrot --pos -0.7+0.3i --zoom 1e6

# Automatic exploration mode
./mandelbrot --auto

# Animated trajectory (zoom from default to target over 60 seconds)
./mandelbrot --pos -0.7115114743-0.3078112463i --zoom 1.86e+11 --auto=60
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--pos <position>` | Target position: standard (`-0.5+0.3i`) or DD format (`re_hi:re_lo+im_hi:im_loi`) for deep zoom |
| `--zoom <value>` | Target zoom level (e.g., `1e6`) |
| `--angle <degrees>` | Target view angle (e.g., `45`) |
| `--auto [N]` | Auto exploration, or trajectory over N seconds |
| `--image [WxH]` | iTerm2 image mode (e.g., `--image=800x600`) |
| `--output <file>` | Save rendered image to file (PPM or PNG on macOS) |
| `--benchmark` | Compute one frame and print timing (no interactive mode) |
| `--no-sa` | Disable Series Approximation (for comparison/debugging) |
| `--debug` | Print DD precision values and exit |
| `--help` | Show help message |

### Interactive Controls

| Key | Action |
|-----|--------|
| Arrow Keys | Pan view |
| SHIFT + Up/Down | Zoom in/out |
| SHIFT + Left/Right | Rotate view |
| C/V | Rotate color palette |
| 1-9 | Switch color schemes |
| +/- | Adjust max iterations |
| I | Toggle iTerm2 image mode |
| R | Reset view |
| Q/ESC | Quit |

## Technical Details

### Series Approximation (SA)

At deep zoom, most pixels follow nearly identical iteration paths. Series Approximation exploits this by computing polynomial coefficients that approximate the perturbation:

```
δZ_n ≈ A_n * δC + B_n * δC² + C_n * δC³ + D_n * δC⁴
```

The coefficients follow recurrence relations derived from the Mandelbrot iteration:
- `A_{n+1} = 2*Z_n*A_n + 1`
- `B_{n+1} = 2*Z_n*B_n + A_n²`
- `C_{n+1} = 2*Z_n*C_n + 2*A_n*B_n`
- `D_{n+1} = 2*Z_n*D_n + 2*A_n*C_n + B_n²`

Instead of iterating each pixel individually, SA evaluates this polynomial to skip directly to a later iteration. The number of skipped iterations varies by position and zoom level; in favorable cases SA can skip a significant portion of iterations, providing substantial speedup.

The validity check ensures the approximation error stays below tolerance:
```
|D_n|² * |δC|⁶ < ε² * |A_n|²
```

### Perturbation Theory

At deep zoom levels, standard double-precision floating point loses accuracy. This explorer uses perturbation theory:

1. Compute a high-precision reference orbit using double-double arithmetic
2. For each pixel, compute only the small perturbation from the reference
3. Detect and recover from "glitched" pixels where perturbation becomes inaccurate

### Double-Double Arithmetic

Double-double uses two `double` values to achieve ~31 decimal digits of precision (vs ~16 for standard double). This is sufficient for zoom levels up to approximately 10^30.

### AVX2 Optimization

When built with `-mavx2`, the explorer processes 4 pixels simultaneously using SIMD instructions, providing significant speedup on modern CPUs.

### iTerm2 Image Mode

When running in iTerm2 (detected via `LC_TERMINAL` or `ITERM_SESSION_ID`), the explorer can render using iTerm2's inline image protocol. This provides much higher resolution output (default 640x400 pixels) compared to terminal character cells. The image is encoded as PPM and transmitted via base64.

## Requirements

- C++17 compiler (clang++ or g++)
- POSIX terminal with ANSI escape code support
- Optional: CPU with AVX2 for SIMD acceleration
- Optional: iTerm2 for high-resolution image mode
- Optional: macOS for PNG output (uses `sips`); PPM output works on all platforms

## License

MIT License
