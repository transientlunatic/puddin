"""
    Puddin

Julia interface to the Puddin Rust library for gravitational-wave binary
parameter computations.

All functions accept and return SI values (`Float64`, kilograms for masses,
radians for angles, dimensionless otherwise).

Functions are organised by physics domain into submodules:

- [`Puddin.Binary`](@ref) — compact binary parameter conversions (masses, spins)

All functions are also exported from the top-level `Puddin` module for
convenience, so existing code using `using Puddin; chirp_mass(...)` continues
to work without modification.

Use Julia broadcasting to apply scalar functions over arrays:

```julia
using Puddin

m1 = fill(30.0 * MSUN, 1000)
m2 = fill(30.0 * MSUN, 1000)

mc  = Binary.chirp_mass.(m1, m2)   # via submodule
mc  = chirp_mass.(m1, m2)          # top-level shortcut (backward compat)
eta = symmetric_mass_ratio.(m1, m2)
```
"""
module Puddin

# ── shared library location ───────────────────────────────────────────────────
#
# Development layout (monorepo):
#   cargo build --release -p puddin-julia
#   → <repo-root>/target/release/libpuddin_julia.{so,dylib,dll}
#
# For a registered Julia package the library should instead be supplied by a
# companion JLL package created with BinaryBuilder.jl.

const _REPO_ROOT = joinpath(@__DIR__, "..", "..", "..")
const _LIB = joinpath(_REPO_ROOT, "target", "release", "libpuddin_julia")

# ── Binary submodule ──────────────────────────────────────────────────────────

"""
    Binary

Compact binary system parameter functions for gravitational-wave astronomy.

All functions accept scalar `Float64` values in SI units (kg for mass, radians
for angles).  Vectorisation is handled by Julia broadcasting:

```julia
using Puddin

mc = Binary.chirp_mass.(m1_array, m2_array)
```
"""
module Binary

export MSUN,
       total_mass, mass_ratio, symmetric_mass_ratio, chirp_mass,
       masses_from_chirp_mass_q, masses_from_chirp_mass_eta,
       chi_eff, chi_p

const _REPO_ROOT = joinpath(@__DIR__, "..", "..", "..")
const _LIB = joinpath(_REPO_ROOT, "target", "release", "libpuddin_julia")

"Solar mass in kilograms."
const MSUN::Float64 = 1.988_416e30

"""
    total_mass(m1_kg, m2_kg) -> Float64

Total mass ``M = m_1 + m_2`` in kilograms.
"""
function total_mass(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:puddin_total_mass, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    mass_ratio(m1_kg, m2_kg) -> Float64

Mass ratio ``q = m_2 / m_1`` (dimensionless, requires ``m_1 \\geq m_2``).
"""
function mass_ratio(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:puddin_mass_ratio, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    symmetric_mass_ratio(m1_kg, m2_kg) -> Float64

Symmetric mass ratio ``\\eta = m_1 m_2 / M^2 \\in (0, 1/4]`` (dimensionless).
"""
function symmetric_mass_ratio(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:puddin_symmetric_mass_ratio, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    chirp_mass(m1_kg, m2_kg) -> Float64

Chirp mass ``\\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}`` in kilograms.
"""
function chirp_mass(m1_kg::Float64, m2_kg::Float64)::Float64
    ccall((:puddin_chirp_mass, _LIB), Float64, (Float64, Float64), m1_kg, m2_kg)
end

"""
    masses_from_chirp_mass_q(mc_kg, q) -> Tuple{Float64, Float64}

Component masses ``(m_1, m_2)`` in kilograms from chirp mass ``\\mathcal{M}``
(kg) and mass ratio ``q = m_2/m_1 \\in (0, 1]``.

Returns `(m1_kg, m2_kg)` with `m1 ≥ m2`.
"""
function masses_from_chirp_mass_q(mc_kg::Float64, q::Float64)::Tuple{Float64,Float64}
    m1 = ccall((:puddin_m1_from_mc_q, _LIB), Float64, (Float64, Float64), mc_kg, q)
    m2 = ccall((:puddin_m2_from_mc_q, _LIB), Float64, (Float64, Float64), mc_kg, q)
    (m1, m2)
end

"""
    masses_from_chirp_mass_eta(mc_kg, eta) -> Tuple{Float64, Float64}

Component masses ``(m_1, m_2)`` in kilograms from chirp mass ``\\mathcal{M}``
(kg) and symmetric mass ratio ``\\eta \\in (0, 0.25]``.

Returns `(m1_kg, m2_kg)` with `m1 ≥ m2`.
"""
function masses_from_chirp_mass_eta(mc_kg::Float64, eta::Float64)::Tuple{Float64,Float64}
    m1 = ccall((:puddin_m1_from_mc_eta, _LIB), Float64, (Float64, Float64), mc_kg, eta)
    m2 = ccall((:puddin_m2_from_mc_eta, _LIB), Float64, (Float64, Float64), mc_kg, eta)
    (m1, m2)
end

"""
    chi_eff(m1_kg, m2_kg, a1, a2, tilt1, tilt2) -> Float64

Effective inspiral spin ``\\chi_\\mathrm{eff} \\in [-1, 1]``.

- `a1`, `a2`: dimensionless spin magnitudes (0–1)
- `tilt1`, `tilt2`: spin tilt angles in radians (0–π)
"""
function chi_eff(
    m1_kg::Float64, m2_kg::Float64,
    a1::Float64, a2::Float64,
    tilt1::Float64, tilt2::Float64,
)::Float64
    ccall(
        (:puddin_chi_eff, _LIB), Float64,
        (Float64, Float64, Float64, Float64, Float64, Float64),
        m1_kg, m2_kg, a1, a2, tilt1, tilt2,
    )
end

"""
    chi_p(m1_kg, m2_kg, a1, a2, tilt1, tilt2) -> Float64

Effective precession spin ``\\chi_p \\in [0, 1]``.

Requires ``m_1 \\geq m_2``.

- `a1`, `a2`: dimensionless spin magnitudes (0–1)
- `tilt1`, `tilt2`: spin tilt angles in radians (0–π)
"""
function chi_p(
    m1_kg::Float64, m2_kg::Float64,
    a1::Float64, a2::Float64,
    tilt1::Float64, tilt2::Float64,
)::Float64
    ccall(
        (:puddin_chi_p, _LIB), Float64,
        (Float64, Float64, Float64, Float64, Float64, Float64),
        m1_kg, m2_kg, a1, a2, tilt1, tilt2,
    )
end

end # module Binary

# ── top-level re-exports (backward-compatible shortcuts) ──────────────────────

export Binary

export MSUN,
       total_mass, mass_ratio, symmetric_mass_ratio, chirp_mass,
       masses_from_chirp_mass_q, masses_from_chirp_mass_eta,
       chi_eff, chi_p

const MSUN = Binary.MSUN
const total_mass = Binary.total_mass
const mass_ratio = Binary.mass_ratio
const symmetric_mass_ratio = Binary.symmetric_mass_ratio
const chirp_mass = Binary.chirp_mass
const masses_from_chirp_mass_q = Binary.masses_from_chirp_mass_q
const masses_from_chirp_mass_eta = Binary.masses_from_chirp_mass_eta
const chi_eff = Binary.chi_eff
const chi_p = Binary.chi_p

end # module Puddin
