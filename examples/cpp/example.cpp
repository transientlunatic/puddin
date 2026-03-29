/*
 * examples/cpp/example.cpp
 *
 * Demonstrates calling the Puddin C API from C++.
 *
 * Build (from repo root):
 *   cargo build --release -p puddin-julia
 *   g++ examples/cpp/example.cpp \
 *       -I bindings/julia/include \
 *       -L target/release -lpuddin_julia \
 *       -Wl,-rpath,$(pwd)/target/release \
 *       -o examples/cpp/example
 *   ./examples/cpp/example
 *
 * puddin.h includes an `extern "C"` guard so the header is directly usable
 * from C++ without any modifications.
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "puddin.h"

int main() {
    constexpr double MSUN = PUDDIN_MSUN;

    // ── Single event ──────────────────────────────────────────────────────────
    double m1 = 30.0 * MSUN;
    double m2 = 30.0 * MSUN;

    std::cout << "=== GW150914-like binary (30+30 Msun) ===\n";
    std::cout << "Total mass      : " << puddin_total_mass(m1, m2) / MSUN   << " Msun\n";
    std::cout << "Mass ratio      : " << puddin_mass_ratio(m1, m2)           << "\n";
    std::cout << "Sym. mass ratio : " << puddin_symmetric_mass_ratio(m1, m2) << "\n";
    std::cout << "Chirp mass      : " << puddin_chirp_mass(m1, m2) / MSUN   << " Msun\n";

    // ── Batch processing over a vector of masses ──────────────────────────────
    std::vector<double> masses = {10.0, 20.0, 30.0, 40.0, 50.0};  // Msun
    std::cout << "\n=== Chirp masses for equal-mass binaries ===\n";
    for (double m_sun : masses) {
        double m = m_sun * MSUN;
        std::cout << "  m1=m2=" << m_sun << " Msun  ->  Mc="
                  << puddin_chirp_mass(m, m) / MSUN << " Msun\n";
    }

    return 0;
}
