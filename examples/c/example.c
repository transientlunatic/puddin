/*
 * examples/c/example.c
 *
 * Demonstrates calling the Puddin C API from C.
 *
 * Build (from repo root):
 *   cargo build --release -p puddin-julia
 *   gcc examples/c/example.c \
 *       -I bindings/julia/include \
 *       -L target/release -lpuddin_julia \
 *       -Wl,-rpath,$(pwd)/target/release \
 *       -o examples/c/example -lm
 *   ./examples/c/example
 */

#include <stdio.h>
#include "puddin.h"

int main(void) {
    double m1 = 30.0 * PUDDIN_MSUN;
    double m2 = 30.0 * PUDDIN_MSUN;

    printf("=== GW150914-like binary (30+30 Msun) ===\n");
    printf("Total mass         : %.2f Msun\n", puddin_total_mass(m1, m2) / PUDDIN_MSUN);
    printf("Mass ratio         : %.4f\n",       puddin_mass_ratio(m1, m2));
    printf("Sym. mass ratio    : %.4f\n",       puddin_symmetric_mass_ratio(m1, m2));
    printf("Chirp mass         : %.4f Msun\n",  puddin_chirp_mass(m1, m2) / PUDDIN_MSUN);

    /* Inverse: recover component masses from chirp mass + mass ratio */
    double mc = puddin_chirp_mass(m1, m2);
    double q  = puddin_mass_ratio(m1, m2);
    printf("m1 from (Mc, q)    : %.4f Msun\n", puddin_m1_from_mc_q(mc, q)  / PUDDIN_MSUN);
    printf("m2 from (Mc, q)    : %.4f Msun\n", puddin_m2_from_mc_q(mc, q)  / PUDDIN_MSUN);

    /* Inverse: recover component masses from chirp mass + sym. mass ratio */
    double eta = puddin_symmetric_mass_ratio(m1, m2);
    printf("m1 from (Mc, eta)  : %.4f Msun\n", puddin_m1_from_mc_eta(mc, eta) / PUDDIN_MSUN);
    printf("m2 from (Mc, eta)  : %.4f Msun\n", puddin_m2_from_mc_eta(mc, eta) / PUDDIN_MSUN);

    /* Mild spin, 30 degrees off axis */
    double a1 = 0.3, a2 = 0.2;
    double tilt1 = 0.5236, tilt2 = 1.0472;   /* 30 deg, 60 deg in radians */
    printf("chi_eff            : %.4f\n",
           puddin_chi_eff(m1, m2, a1, a2, tilt1, tilt2));
    printf("chi_p              : %.4f\n",
           puddin_chi_p(m1, m2, a1, a2, tilt1, tilt2));

    return 0;
}
