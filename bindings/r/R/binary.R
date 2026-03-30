# ── R/binary.R ────────────────────────────────────────────────────────────────
#
# Provides a `binary` environment that groups all compact binary parameter
# functions under a single domain-organised namespace, mirroring the
# `puddin.binary` submodule in Python and the `Puddin.Binary` submodule in
# Julia.
#
# Usage:
#   library(Puddin)
#   Puddin::binary$chirp_mass(30 * MSUN, 30 * MSUN)
#
# All top-level function names remain available for backward compatibility.

#' Compact binary parameter functions for gravitational-wave astronomy.
#'
#' An environment containing all binary system parameter functions, grouped
#' under a single domain namespace.  Mirrors the \code{puddin.binary} module
#' in Python and the \code{Puddin.Binary} module in Julia.
#'
#' Usage:
#' \preformatted{
#' library(Puddin)
#' Puddin::binary$chirp_mass(30 * MSUN, 30 * MSUN) / MSUN  # ~26.1
#' }
#'
#' @export
binary <- new.env(parent = emptyenv())

binary$MSUN                    <- MSUN
binary$total_mass               <- total_mass
binary$mass_ratio               <- mass_ratio
binary$symmetric_mass_ratio     <- symmetric_mass_ratio
binary$chirp_mass               <- chirp_mass
binary$masses_from_chirp_mass_q <- masses_from_chirp_mass_q
binary$masses_from_chirp_mass_eta <- masses_from_chirp_mass_eta
binary$chi_eff                  <- chi_eff
binary$chi_p                    <- chi_p

lockEnvironment(binary, bindings = TRUE)
