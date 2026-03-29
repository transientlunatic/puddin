"""JAX wrappers for puddin Rust functions.

Each Rust function is wrapped with :func:`jax.pure_callback` so that it can
participate in JIT-compiled JAX programs.  Analytical VJP (reverse-mode
autodiff) rules are registered via :func:`jax.custom_vjp` for every function,
since the Rust kernel is opaque to JAX's automatic differentiation.

If JAX is not installed this module fails silently; the public ``__init__``
will fall back to the raw Rust wrappers.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import pure_callback
from jax.lib import xla_client

from puddin import _puddin  # the compiled Rust extension

# ── helpers ───────────────────────────────────────────────────────────────────


def _result_shape(x: jax.Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


# ── total_mass ────────────────────────────────────────────────────────────────

@jax.custom_vjp
def total_mass(m1: jax.Array, m2: jax.Array) -> jax.Array:
    """Total mass M = m1 + m2 (kg)."""
    return pure_callback(
        _puddin.total_mass, _result_shape(m1), m1, m2, vectorized=True
    )


def _total_mass_fwd(m1, m2):
    return total_mass(m1, m2), (m1, m2)


def _total_mass_bwd(_, g):
    # dM/dm1 = 1, dM/dm2 = 1
    return g, g


total_mass.defvjp(_total_mass_fwd, _total_mass_bwd)

# ── mass_ratio ────────────────────────────────────────────────────────────────

@jax.custom_vjp
def mass_ratio(m1: jax.Array, m2: jax.Array) -> jax.Array:
    """Mass ratio q = m2 / m1."""
    return pure_callback(
        _puddin.mass_ratio, _result_shape(m1), m1, m2, vectorized=True
    )


def _mass_ratio_fwd(m1, m2):
    return mass_ratio(m1, m2), (m1, m2)


def _mass_ratio_bwd(res, g):
    m1, m2 = res
    # dq/dm1 = -m2/m1^2,  dq/dm2 = 1/m1
    return -g * m2 / (m1 ** 2), g / m1


mass_ratio.defvjp(_mass_ratio_fwd, _mass_ratio_bwd)

# ── symmetric_mass_ratio ──────────────────────────────────────────────────────

@jax.custom_vjp
def symmetric_mass_ratio(m1: jax.Array, m2: jax.Array) -> jax.Array:
    """Symmetric mass ratio η = m1*m2 / (m1+m2)^2."""
    return pure_callback(
        _puddin.symmetric_mass_ratio, _result_shape(m1), m1, m2, vectorized=True
    )


def _eta_fwd(m1, m2):
    return symmetric_mass_ratio(m1, m2), (m1, m2)


def _eta_bwd(res, g):
    m1, m2 = res
    M = m1 + m2
    # dη/dm1 = m2*(m2 - m1) / M^3
    # dη/dm2 = m1*(m1 - m2) / M^3
    dm1 = m2 * (m2 - m1) / (M ** 3)
    dm2 = m1 * (m1 - m2) / (M ** 3)
    return g * dm1, g * dm2


symmetric_mass_ratio.defvjp(_eta_fwd, _eta_bwd)

# ── chirp_mass ────────────────────────────────────────────────────────────────

@jax.custom_vjp
def chirp_mass(m1: jax.Array, m2: jax.Array) -> jax.Array:
    """Chirp mass Mc = (m1*m2)^(3/5) / M^(1/5) (kg)."""
    return pure_callback(
        _puddin.chirp_mass, _result_shape(m1), m1, m2, vectorized=True
    )


def _mc_fwd(m1, m2):
    return chirp_mass(m1, m2), (m1, m2)


def _mc_bwd(res, g):
    m1, m2 = res
    M = m1 + m2
    eta = m1 * m2 / (M ** 2)
    Mc = (m1 * m2) ** (3.0 / 5.0) / M ** (1.0 / 5.0)
    # dMc/dm1 = Mc * (3/(5*m1) - 1/(5*M))
    dMc_dm1 = Mc * (3.0 / (5.0 * m1) - 1.0 / (5.0 * M))
    dMc_dm2 = Mc * (3.0 / (5.0 * m2) - 1.0 / (5.0 * M))
    return g * dMc_dm1, g * dMc_dm2


chirp_mass.defvjp(_mc_fwd, _mc_bwd)

# ── chi_eff ───────────────────────────────────────────────────────────────────

@jax.custom_vjp
def chi_eff(
    m1: jax.Array,
    m2: jax.Array,
    a1: jax.Array,
    a2: jax.Array,
    tilt1: jax.Array,
    tilt2: jax.Array,
) -> jax.Array:
    """Effective inspiral spin χ_eff."""
    return pure_callback(
        _puddin.chi_eff,
        _result_shape(m1),
        m1, m2, a1, a2, tilt1, tilt2,
        vectorized=True,
    )


def _chi_eff_fwd(m1, m2, a1, a2, tilt1, tilt2):
    return chi_eff(m1, m2, a1, a2, tilt1, tilt2), (m1, m2, a1, a2, tilt1, tilt2)


def _chi_eff_bwd(res, g):
    m1, m2, a1, a2, tilt1, tilt2 = res
    M = m1 + m2
    c1, c2 = jnp.cos(tilt1), jnp.cos(tilt2)
    # χ_eff = (m1*a1*cos(t1) + m2*a2*cos(t2)) / M
    dx_dm1 = (a1 * c1 * M - (m1 * a1 * c1 + m2 * a2 * c2)) / (M ** 2)
    dx_dm2 = (a2 * c2 * M - (m1 * a1 * c1 + m2 * a2 * c2)) / (M ** 2)
    dx_da1 = m1 * c1 / M
    dx_da2 = m2 * c2 / M
    dx_dt1 = -m1 * a1 * jnp.sin(tilt1) / M
    dx_dt2 = -m2 * a2 * jnp.sin(tilt2) / M
    return g * dx_dm1, g * dx_dm2, g * dx_da1, g * dx_da2, g * dx_dt1, g * dx_dt2


chi_eff.defvjp(_chi_eff_fwd, _chi_eff_bwd)

# ── chi_p ─────────────────────────────────────────────────────────────────────

@jax.custom_vjp
def chi_p(
    m1: jax.Array,
    m2: jax.Array,
    a1: jax.Array,
    a2: jax.Array,
    tilt1: jax.Array,
    tilt2: jax.Array,
) -> jax.Array:
    """Effective precession spin χ_p."""
    return pure_callback(
        _puddin.chi_p,
        _result_shape(m1),
        m1, m2, a1, a2, tilt1, tilt2,
        vectorized=True,
    )


def _chi_p_fwd(m1, m2, a1, a2, tilt1, tilt2):
    return chi_p(m1, m2, a1, a2, tilt1, tilt2), (m1, m2, a1, a2, tilt1, tilt2)


def _chi_p_bwd(res, g):
    m1, m2, a1, a2, tilt1, tilt2 = res
    # χ_p = max(term1, term2) where
    # term1 = a1 * sin(t1)
    # term2 = (3+4q)/(4(1+q)) * q * a2 * sin(t2),  q = m2/m1
    q = m2 / m1
    coeff = (3.0 + 4.0 * q) / (4.0 * (1.0 + q))
    term1 = a1 * jnp.sin(tilt1)
    term2 = coeff * q * a2 * jnp.sin(tilt2)
    sel = term1 >= term2  # where term1 wins
    # Gradients for the winning branch; the other branch gets zero
    # (subgradient at tie; JAX convention: left branch wins)
    dt1_da1 = jnp.sin(tilt1)
    dt1_dt1 = a1 * jnp.cos(tilt1)
    # d(coeff*q)/dm1 and dm1 for term2 is complex; use JAX to compute
    dcoeff_dq = (4.0 * (1.0 + q) * 4.0 - (3.0 + 4.0 * q) * 4.0) / (4.0 * (1.0 + q)) ** 2
    d_term2_dq = (dcoeff_dq * q + coeff) * a2 * jnp.sin(tilt2)
    dq_dm1 = -m2 / (m1 ** 2)
    dq_dm2 = 1.0 / m1
    d_term2_dm1 = d_term2_dq * dq_dm1
    d_term2_dm2 = d_term2_dq * dq_dm2
    d_term2_da2 = coeff * q * jnp.sin(tilt2)
    d_term2_dt2 = coeff * q * a2 * jnp.cos(tilt2)

    gm1 = g * jnp.where(sel, 0.0, d_term2_dm1)
    gm2 = g * jnp.where(sel, 0.0, d_term2_dm2)
    ga1 = g * jnp.where(sel, dt1_da1, 0.0)
    ga2 = g * jnp.where(sel, 0.0, d_term2_da2)
    gt1 = g * jnp.where(sel, dt1_dt1, 0.0)
    gt2 = g * jnp.where(sel, 0.0, d_term2_dt2)
    return gm1, gm2, ga1, ga2, gt1, gt2


chi_p.defvjp(_chi_p_fwd, _chi_p_bwd)

# ── masses_from_chirp_mass_q ──────────────────────────────────────────────────

def _mc_q_result_shapes(mc, q):
    return (_result_shape(mc), _result_shape(mc))


@jax.custom_vjp
def masses_from_chirp_mass_q(mc: jax.Array, q: jax.Array):
    """Component masses (m1, m2) from chirp mass Mc and mass ratio q."""
    return pure_callback(
        _puddin.masses_from_chirp_mass_q,
        _mc_q_result_shapes(mc, q),
        mc, q,
        vectorized=True,
    )


def _mc_q_fwd(mc, q):
    return masses_from_chirp_mass_q(mc, q), (mc, q)


def _mc_q_bwd(res, g):
    mc, q = res
    gm1, gm2 = g
    eta = q / (1.0 + q) ** 2
    M = mc / eta ** (3.0 / 5.0)
    # m1 = M / (1 + q),  m2 = q * m1
    # dM/dMc = 1/eta^(3/5)
    dM_dmc = 1.0 / eta ** (3.0 / 5.0)
    dM_dq = mc * (-3.0 / 5.0) * eta ** (-8.0 / 5.0) * (1.0 - q) / (1.0 + q) ** 3
    dm1_dmc = dM_dmc / (1.0 + q)
    dm1_dq = dM_dq / (1.0 + q) - M / (1.0 + q) ** 2
    dm2_dmc = dm1_dmc * q + M / (1.0 + q)  # chain rule: m2 = q * m1
    dm2_dq = dm1_dq * q + M / (1.0 + q)   # dm2/dq = dm1/dq * q + m1
    g_mc = gm1 * dm1_dmc + gm2 * dm2_dmc
    g_q = gm1 * dm1_dq + gm2 * dm2_dq
    return g_mc, g_q


masses_from_chirp_mass_q.defvjp(_mc_q_fwd, _mc_q_bwd)

# ── masses_from_chirp_mass_eta ────────────────────────────────────────────────

def _mc_eta_result_shapes(mc, eta):
    return (_result_shape(mc), _result_shape(mc))


@jax.custom_vjp
def masses_from_chirp_mass_eta(mc: jax.Array, eta: jax.Array):
    """Component masses (m1, m2) from chirp mass Mc and symmetric mass ratio eta."""
    return pure_callback(
        _puddin.masses_from_chirp_mass_eta,
        _mc_eta_result_shapes(mc, eta),
        mc, eta,
        vectorized=True,
    )


def _mc_eta_fwd(mc, eta):
    return masses_from_chirp_mass_eta(mc, eta), (mc, eta)


def _mc_eta_bwd(res, g):
    mc, eta = res
    gm1, gm2 = g
    eta_safe = jnp.minimum(eta, 0.25)
    disc = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta_safe, 0.0))
    M = mc / eta_safe ** (3.0 / 5.0)
    # m1 = M(1+disc)/2,  m2 = M(1-disc)/2
    dM_dmc = 1.0 / eta_safe ** (3.0 / 5.0)
    dM_deta = mc * (-3.0 / 5.0) * eta_safe ** (-8.0 / 5.0)
    ddisc_deta = jnp.where(disc > 0.0, -2.0 / disc, 0.0)
    dm1_dmc = dM_dmc * (1.0 + disc) / 2.0
    dm1_deta = dM_deta * (1.0 + disc) / 2.0 + M * ddisc_deta / 2.0
    dm2_dmc = dM_dmc * (1.0 - disc) / 2.0
    dm2_deta = dM_deta * (1.0 - disc) / 2.0 - M * ddisc_deta / 2.0
    g_mc = gm1 * dm1_dmc + gm2 * dm2_dmc
    g_eta = gm1 * dm1_deta + gm2 * dm2_deta
    return g_mc, g_eta


masses_from_chirp_mass_eta.defvjp(_mc_eta_fwd, _mc_eta_bwd)
