"""Puddin — mathematical and physical primitives for gravitational-wave astronomy.

Public API
----------
All functions accept:

* plain ``float`` or ``numpy.ndarray`` (assumed SI: kg, rad, dimensionless)
* ``astropy.units.Quantity``
* ``pint.Quantity``
* ``jax.Array`` (routes automatically to the JAX-compatible backend)

Scalar inputs are broadcast to 1-D arrays.  All functions return
``numpy.ndarray`` (Rust backend) or ``jax.Array`` (JAX backend).

Mass functions return values in **kilograms**; use ``astropy.constants`` or
``pint`` to convert to solar masses.

Examples
--------
>>> import numpy as np
>>> from puddin import chirp_mass
>>> chirp_mass(30 * 1.989e30, 30 * 1.989e30)   # plain SI
array([...])

>>> from astropy import units as u
>>> chirp_mass(30 * u.Msun, 30 * u.Msun)
array([...])
"""

from __future__ import annotations

import numpy as np

from puddin.units import to_kg, to_rad, to_dimensionless
from puddin import _puddin as _rust

# ── JAX availability detection ────────────────────────────────────────────────

try:
    import jax
    import jax.numpy as jnp
    import puddin.jax_wrapper as _jax

    def _is_jax(x) -> bool:
        return isinstance(x, jax.Array)

    _JAX_AVAILABLE = True
except ModuleNotFoundError:
    _JAX_AVAILABLE = False

    def _is_jax(x) -> bool:  # type: ignore[misc]
        return False


def _any_jax(*args) -> bool:
    return any(_is_jax(a) for a in args)


# ── public API ────────────────────────────────────────────────────────────────

def total_mass(m1, m2):
    """Total mass $M = m_1 + m_2$ (kg).

    Parameters
    ----------
    m1, m2:
        Component masses.  Accepts plain SI floats/arrays, astropy Quantities,
        pint Quantities, or JAX arrays.
    """
    if _any_jax(m1, m2):
        return _jax.total_mass(m1, m2)
    return _rust.total_mass(to_kg(m1), to_kg(m2))


def mass_ratio(m1, m2):
    """Mass ratio $q = m_2 / m_1$ (dimensionless)."""
    if _any_jax(m1, m2):
        return _jax.mass_ratio(m1, m2)
    return _rust.mass_ratio(to_kg(m1), to_kg(m2))


def symmetric_mass_ratio(m1, m2):
    r"""Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$."""
    if _any_jax(m1, m2):
        return _jax.symmetric_mass_ratio(m1, m2)
    return _rust.symmetric_mass_ratio(to_kg(m1), to_kg(m2))


def chirp_mass(m1, m2):
    r"""Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$ (kg)."""
    if _any_jax(m1, m2):
        return _jax.chirp_mass(m1, m2)
    return _rust.chirp_mass(to_kg(m1), to_kg(m2))


def masses_from_chirp_mass_q(mc, q):
    r"""Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and mass ratio $q$.

    Parameters
    ----------
    mc:
        Chirp mass.  Accepts plain SI floats/arrays (kg), astropy Quantities,
        or pint Quantities.
    q:
        Mass ratio $q = m_2 / m_1 \in (0, 1]$ (dimensionless).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(m1, m2)`` in kilograms, with $m_1 \geq m_2$.
    """
    if _any_jax(mc, q):
        return _jax.masses_from_chirp_mass_q(mc, q)
    return _rust.masses_from_chirp_mass_q(to_kg(mc), to_dimensionless(q))


def masses_from_chirp_mass_eta(mc, eta):
    r"""Component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and symmetric mass ratio $\eta$.

    Parameters
    ----------
    mc:
        Chirp mass.  Accepts plain SI floats/arrays (kg), astropy Quantities,
        or pint Quantities.
    eta:
        Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$ (dimensionless).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(m1, m2)`` in kilograms, with $m_1 \geq m_2$.
    """
    if _any_jax(mc, eta):
        return _jax.masses_from_chirp_mass_eta(mc, eta)
    return _rust.masses_from_chirp_mass_eta(to_kg(mc), to_dimensionless(eta))


def chi_eff(m1, m2, a1, a2, tilt1, tilt2):
    r"""Effective inspiral spin $\chi_\mathrm{eff}$.

    Parameters
    ----------
    m1, m2:
        Component masses (SI or unit-carrying).
    a1, a2:
        Dimensionless spin magnitudes $\in [0, 1]$.
    tilt1, tilt2:
        Spin tilt angles (radians) w.r.t. the orbital angular momentum axis.
    """
    if _any_jax(m1, m2, a1, a2, tilt1, tilt2):
        return _jax.chi_eff(m1, m2, a1, a2, tilt1, tilt2)
    return _rust.chi_eff(
        to_kg(m1), to_kg(m2),
        to_dimensionless(a1), to_dimensionless(a2),
        to_rad(tilt1), to_rad(tilt2),
    )


def chi_p(m1, m2, a1, a2, tilt1, tilt2):
    r"""Effective precession spin $\chi_p$.

    Parameters
    ----------
    m1, m2:
        Component masses (SI or unit-carrying).
    a1, a2:
        Dimensionless spin magnitudes $\in [0, 1]$.
    tilt1, tilt2:
        Spin tilt angles (radians) w.r.t. the orbital angular momentum axis.
    """
    if _any_jax(m1, m2, a1, a2, tilt1, tilt2):
        return _jax.chi_p(m1, m2, a1, a2, tilt1, tilt2)
    return _rust.chi_p(
        to_kg(m1), to_kg(m2),
        to_dimensionless(a1), to_dimensionless(a2),
        to_rad(tilt1), to_rad(tilt2),
    )


__all__ = [
    "total_mass",
    "mass_ratio",
    "symmetric_mass_ratio",
    "chirp_mass",
    "masses_from_chirp_mass_q",
    "masses_from_chirp_mass_eta",
    "chi_eff",
    "chi_p",
]
