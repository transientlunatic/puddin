# ── R/puddin.R ────────────────────────────────────────────────────────────────
#
# R interface to the Puddin shared library.
#
# All functions are vectorised via base R `Vectorize()`.  For large arrays,
# consider using the scalar forms directly inside `vapply()` or `mapply()` for
# maximum control.
#
# All masses are in kilograms; use MSUN to convert from solar masses.
# All angles are in radians; spin magnitudes are dimensionless (0–1).

#' Solar mass in kilograms.
#' @export
MSUN <- 1.988416e30

# ── internal scalar wrappers calling the C ABI ────────────────────────────────

.total_mass_scalar <- function(m1_kg, m2_kg) {
  .C("r_puddin_total_mass",
     m1_kg  = as.double(m1_kg),
     m2_kg  = as.double(m2_kg),
     result = double(1L),
     PACKAGE = "Puddin")$result
}

.mass_ratio_scalar <- function(m1_kg, m2_kg) {
  .C("r_puddin_mass_ratio",
     m1_kg  = as.double(m1_kg),
     m2_kg  = as.double(m2_kg),
     result = double(1L),
     PACKAGE = "Puddin")$result
}

.symmetric_mass_ratio_scalar <- function(m1_kg, m2_kg) {
  .C("r_puddin_symmetric_mass_ratio",
     m1_kg  = as.double(m1_kg),
     m2_kg  = as.double(m2_kg),
     result = double(1L),
     PACKAGE = "Puddin")$result
}

.chirp_mass_scalar <- function(m1_kg, m2_kg) {
  .C("r_puddin_chirp_mass",
     m1_kg  = as.double(m1_kg),
     m2_kg  = as.double(m2_kg),
     result = double(1L),
     PACKAGE = "Puddin")$result
}

.chi_eff_scalar <- function(m1_kg, m2_kg, a1, a2, tilt1, tilt2) {
  .C("r_puddin_chi_eff",
     m1_kg  = as.double(m1_kg),
     m2_kg  = as.double(m2_kg),
     a1     = as.double(a1),
     a2     = as.double(a2),
     tilt1  = as.double(tilt1),
     tilt2  = as.double(tilt2),
     result = double(1L),
     PACKAGE = "Puddin")$result
}

.chi_p_scalar <- function(m1_kg, m2_kg, a1, a2, tilt1, tilt2) {
  .C("r_puddin_chi_p",
     m1_kg  = as.double(m1_kg),
     m2_kg  = as.double(m2_kg),
     a1     = as.double(a1),
     a2     = as.double(a2),
     tilt1  = as.double(tilt1),
     tilt2  = as.double(tilt2),
     result = double(1L),
     PACKAGE = "Puddin")$result
}

# ── public vectorised API ─────────────────────────────────────────────────────

#' Total mass M = m1 + m2
#'
#' @param m1_kg Component mass 1 (kg, numeric vector)
#' @param m2_kg Component mass 2 (kg, numeric vector)
#' @return Total mass in kilograms (numeric vector)
#' @export
#' @examples
#' total_mass(30 * MSUN, 30 * MSUN) / MSUN  # 60
total_mass <- Vectorize(.total_mass_scalar)

#' Mass ratio q = m2 / m1
#'
#' @param m1_kg Primary mass (kg, m1 >= m2, numeric vector)
#' @param m2_kg Secondary mass (kg, numeric vector)
#' @return Dimensionless mass ratio (numeric vector)
#' @export
#' @examples
#' mass_ratio(30 * MSUN, 15 * MSUN)  # 0.5
mass_ratio <- Vectorize(.mass_ratio_scalar)

#' Symmetric mass ratio eta = m1*m2 / M^2
#'
#' @param m1_kg Component mass 1 (kg, numeric vector)
#' @param m2_kg Component mass 2 (kg, numeric vector)
#' @return Dimensionless symmetric mass ratio in (0, 0.25] (numeric vector)
#' @export
#' @examples
#' symmetric_mass_ratio(30 * MSUN, 30 * MSUN)  # 0.25
symmetric_mass_ratio <- Vectorize(.symmetric_mass_ratio_scalar)

#' Chirp mass Mc = (m1*m2)^(3/5) / M^(1/5)
#'
#' @param m1_kg Component mass 1 (kg, numeric vector)
#' @param m2_kg Component mass 2 (kg, numeric vector)
#' @return Chirp mass in kilograms (numeric vector)
#' @export
#' @examples
#' chirp_mass(30 * MSUN, 30 * MSUN) / MSUN  # ~26.1
chirp_mass <- Vectorize(.chirp_mass_scalar)

#' Effective inspiral spin chi_eff
#'
#' @param m1_kg  Component mass 1 (kg, numeric vector)
#' @param m2_kg  Component mass 2 (kg, numeric vector)
#' @param a1     Spin magnitude of body 1, 0-1 (numeric vector)
#' @param a2     Spin magnitude of body 2, 0-1 (numeric vector)
#' @param tilt1  Spin tilt of body 1 in radians, 0-pi (numeric vector)
#' @param tilt2  Spin tilt of body 2 in radians, 0-pi (numeric vector)
#' @return Effective inspiral spin in [-1, 1] (numeric vector)
#' @export
#' @examples
#' chi_eff(30 * MSUN, 30 * MSUN, 0, 0, 0, 0)  # 0
chi_eff <- Vectorize(.chi_eff_scalar)

#' Effective precession spin chi_p
#'
#' Requires m1 >= m2.
#'
#' @param m1_kg  Primary mass (kg, m1 >= m2, numeric vector)
#' @param m2_kg  Secondary mass (kg, numeric vector)
#' @param a1     Spin magnitude of body 1, 0-1 (numeric vector)
#' @param a2     Spin magnitude of body 2, 0-1 (numeric vector)
#' @param tilt1  Spin tilt of body 1 in radians, 0-pi (numeric vector)
#' @param tilt2  Spin tilt of body 2 in radians, 0-pi (numeric vector)
#' @return Effective precession spin in [0, 1] (numeric vector)
#' @export
#' @examples
#' chi_p(30 * MSUN, 30 * MSUN, 0, 0, 0, 0)  # 0
chi_p <- Vectorize(.chi_p_scalar)
