/*
 * puddin_r.c — thin C shims adapting the Puddin C ABI (value-returning) to
 * the output-pointer convention expected by R's .C() interface.
 *
 * These functions are compiled as part of the Puddin R package and linked
 * against libpuddin_julia (the cdylib produced by the bindings/julia crate).
 *
 * Routine registration at the bottom follows the modern R convention
 * (R_registerRoutines / R_useDynamicSymbols) so symbols can be resolved
 * reliably with PACKAGE = "Puddin" in .C() calls.
 */

#include <R.h>
#include <R_ext/Rdynload.h>
#include "puddin.h"

/* ── binary parameters ─────────────────────────────────────────────────────── */

void r_puddin_total_mass(double *m1_kg, double *m2_kg, double *result) {
    *result = puddin_total_mass(*m1_kg, *m2_kg);
}

void r_puddin_mass_ratio(double *m1_kg, double *m2_kg, double *result) {
    *result = puddin_mass_ratio(*m1_kg, *m2_kg);
}

void r_puddin_symmetric_mass_ratio(double *m1_kg, double *m2_kg, double *result) {
    *result = puddin_symmetric_mass_ratio(*m1_kg, *m2_kg);
}

void r_puddin_chirp_mass(double *m1_kg, double *m2_kg, double *result) {
    *result = puddin_chirp_mass(*m1_kg, *m2_kg);
}

void r_puddin_m1_from_mc_q(double *mc_kg, double *q, double *result) {
    *result = puddin_m1_from_mc_q(*mc_kg, *q);
}

void r_puddin_m2_from_mc_q(double *mc_kg, double *q, double *result) {
    *result = puddin_m2_from_mc_q(*mc_kg, *q);
}

void r_puddin_m1_from_mc_eta(double *mc_kg, double *eta, double *result) {
    *result = puddin_m1_from_mc_eta(*mc_kg, *eta);
}

void r_puddin_m2_from_mc_eta(double *mc_kg, double *eta, double *result) {
    *result = puddin_m2_from_mc_eta(*mc_kg, *eta);
}

void r_puddin_chi_eff(
    double *m1_kg, double *m2_kg,
    double *a1, double *a2,
    double *tilt1, double *tilt2,
    double *result)
{
    *result = puddin_chi_eff(*m1_kg, *m2_kg, *a1, *a2, *tilt1, *tilt2);
}

void r_puddin_chi_p(
    double *m1_kg, double *m2_kg,
    double *a1, double *a2,
    double *tilt1, double *tilt2,
    double *result)
{
    *result = puddin_chi_p(*m1_kg, *m2_kg, *a1, *a2, *tilt1, *tilt2);
}

/* ── routine registration ───────────────────────────────────────────────────── */

static const R_CMethodDef cMethods[] = {
    {"r_puddin_total_mass",             (DL_FUNC) &r_puddin_total_mass,             3},
    {"r_puddin_mass_ratio",             (DL_FUNC) &r_puddin_mass_ratio,             3},
    {"r_puddin_symmetric_mass_ratio",   (DL_FUNC) &r_puddin_symmetric_mass_ratio,   3},
    {"r_puddin_chirp_mass",             (DL_FUNC) &r_puddin_chirp_mass,             3},
    {"r_puddin_m1_from_mc_q",           (DL_FUNC) &r_puddin_m1_from_mc_q,           3},
    {"r_puddin_m2_from_mc_q",           (DL_FUNC) &r_puddin_m2_from_mc_q,           3},
    {"r_puddin_m1_from_mc_eta",         (DL_FUNC) &r_puddin_m1_from_mc_eta,         3},
    {"r_puddin_m2_from_mc_eta",         (DL_FUNC) &r_puddin_m2_from_mc_eta,         3},
    {"r_puddin_chi_eff",                (DL_FUNC) &r_puddin_chi_eff,                7},
    {"r_puddin_chi_p",                  (DL_FUNC) &r_puddin_chi_p,                  7},
    {NULL, NULL, 0}
};

void R_init_Puddin(DllInfo *dll) {
    R_registerRoutines(dll, cMethods, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
