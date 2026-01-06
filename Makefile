# Mandelbrot Explorer Makefile
# Builds with maximum optimizations

CXX ?= clang++
# Default: portable build without AVX2 (works on all x86_64 CPUs)
CXXFLAGS = -std=c++17 -O3 -flto -fno-fast-math
LDFLAGS = -pthread

TARGET = mandelbrot
SRC = mandelbrot.cpp

TEST_TARGET = test_perturbation
TEST_SRC = test_perturbation.cpp

TEST_TRAJ_TARGET = test_trajectory
TEST_TRAJ_SRC = test_trajectory.cpp

TEST_SA_TARGET = test_sa
TEST_SA_SRC = test_sa.cpp

.PHONY: all clean run avx2 native test test-trajectory test-sa test-all

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║           MANDELBROT EXPLORER - BUILD COMPLETE             ║"
	@echo "╠════════════════════════════════════════════════════════════╣"
	@echo "║  Run with: ./mandelbrot                                    ║"
	@echo "║                                                            ║"
	@echo "║  Controls:                                                 ║"
	@echo "║    Arrow Keys        - Pan view                            ║"
	@echo "║    SHIFT + Up/Down   - Zoom in/out                         ║"
	@echo "║    SHIFT + Left/Right- Rotate colors                       ║"
	@if [ "$$(uname -s)" = "Darwin" ]; then echo "║    Z                 - Toggle arrows: Pan<->Zoom/Rotate    ║"; fi
	@echo "║    1-9               - Switch color schemes                ║"
	@echo "║    +/-               - Adjust iterations                   ║"
	@echo "║    R                 - Reset view                          ║"
	@echo "║    Q / ESC           - Quit                                ║"
	@echo "╚════════════════════════════════════════════════════════════╝"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET) $(TEST_TRAJ_TARGET) $(TEST_SA_TARGET)

# Debug build
debug: CXXFLAGS = -std=c++17 -g -fsanitize=address
debug: $(TARGET)

# AVX2 optimized build (only for CPUs with AVX2 support)
avx2: CXXFLAGS += -mavx2
avx2: $(TARGET)
	@echo "Built with AVX2 SIMD optimization"

# Native build (uses all features of the current CPU)
native: CXXFLAGS += -march=native
native: $(TARGET)
	@echo "Built with native CPU optimizations"

# Test build and run
$(TEST_TARGET): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Trajectory test build and run
$(TEST_TRAJ_TARGET): $(TEST_TRAJ_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

test-trajectory: $(TEST_TRAJ_TARGET)
	./$(TEST_TRAJ_TARGET)

# Series Approximation test build and run
$(TEST_SA_TARGET): $(TEST_SA_SRC)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

test-sa: $(TEST_SA_TARGET)
	./$(TEST_SA_TARGET)

# Run all tests
test-all: test test-trajectory test-sa
