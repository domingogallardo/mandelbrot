# Mandelbrot Explorer

Ultra-fast console-based Mandelbrot fractal explorer with deep zoom capabilities using perturbation theory and double-double arithmetic.

![Mandelbrot Animation](assets/mandelbrot.gif)

## Features

- **Deep Zoom**: Explore to zoom levels beyond 10^30 using perturbation theory
- **Double-Double Precision**: ~31 decimal digits of precision for reference orbit computation
- **AVX2 SIMD**: Optional 4x parallel pixel computation on supported CPUs
- **Smooth Animation**: Trajectory mode with ease-in-out cubic easing
- **Multiple Color Schemes**: 9 built-in palettes with rotation support
- **Real-time Navigation**: Pan, zoom, and rotate interactively

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
| `--pos <re+imi>` | Target position (e.g., `-0.5+0.3i`) |
| `--zoom <value>` | Target zoom level (e.g., `1e6`) |
| `--angle <degrees>` | Target view angle (e.g., `45`) |
| `--auto [N]` | Auto exploration, or trajectory over N seconds |
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
| R | Reset view |
| Q/ESC | Quit |

## Technical Details

### Perturbation Theory

At deep zoom levels, standard double-precision floating point loses accuracy. This explorer uses perturbation theory:

1. Compute a high-precision reference orbit using double-double arithmetic
2. For each pixel, compute only the small perturbation from the reference
3. Detect and recover from "glitched" pixels where perturbation becomes inaccurate

### Double-Double Arithmetic

Double-double uses two `double` values to achieve ~31 decimal digits of precision (vs ~16 for standard double). This is sufficient for zoom levels up to approximately 10^30.

### AVX2 Optimization

When built with `-mavx2`, the explorer processes 4 pixels simultaneously using SIMD instructions, providing significant speedup on modern CPUs.

## Requirements

- C++17 compiler (clang++ or g++)
- POSIX terminal with ANSI escape code support
- Optional: CPU with AVX2 for SIMD acceleration

## License

MIT License
