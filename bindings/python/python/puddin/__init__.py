"""Puddin — mathematical and physical primitives for gravitational-wave astronomy.

Public API
----------
Functions are organised into submodules by physics domain:

* :mod:`puddin.binary` – compact binary parameter conversions (masses, spins)

All functions are also importable directly from the top-level ``puddin``
namespace for convenience.  They accept:

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
>>> import puddin.binary
>>> puddin.binary.chirp_mass(30 * 1.989e30, 30 * 1.989e30)   # plain SI
array([...])

>>> from astropy import units as u
>>> puddin.binary.chirp_mass(30 * u.Msun, 30 * u.Msun)
array([...])

>>> from puddin import chirp_mass          # top-level shortcut still works
>>> chirp_mass(30 * 1.989e30, 30 * 1.989e30)
array([...])
"""

from __future__ import annotations

# ── submodule ─────────────────────────────────────────────────────────────────

from puddin import binary  # noqa: F401 — expose as puddin.binary

# ── top-level re-exports (backward-compatible shortcuts) ─────────────────────

from puddin.binary import (  # noqa: F401
    total_mass,
    mass_ratio,
    symmetric_mass_ratio,
    chirp_mass,
    masses_from_chirp_mass_q,
    masses_from_chirp_mass_eta,
    chi_eff,
    chi_p,
)

__all__ = [
    "binary",
    "total_mass",
    "mass_ratio",
    "symmetric_mass_ratio",
    "chirp_mass",
    "masses_from_chirp_mass_q",
    "masses_from_chirp_mass_eta",
    "chi_eff",
    "chi_p",
]
