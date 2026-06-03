"""LALSimulation interface for puddin spin-frame transformations.

This submodule wraps the LALSimulation function
``SimInspiralTransformPrecessingNewInitialConditions`` to provide a
vectorised, unit-aware conversion from the bilby/LALInference spin
parameterisation to the Cartesian spin components expected by
LALSimulation waveform generators.

The *underlying geometry* (spherical decomposition in the L-frame) lives
in :func:`puddin.spin_components`; the additional step performed here is
the **J-frame → L-frame rotation**, which requires the orbital angular
momentum magnitude at the reference frequency and is computed internally
by LALSimulation using a post-Newtonian expansion.

Requires
--------
lalsimulation, lal
    Both must be installed (e.g. via a LIGO-style conda environment).

See also
--------
puddin.spin_components : Pure-Python L-frame decomposition (no lalsim).
puddin.orbital_angular_momentum : Newtonian |L_N| at a reference frequency.
bilby.gw.conversion.bilby_to_lalsimulation_spins : Reference implementation
    used to validate this module.
"""

from __future__ import annotations

import numpy as np

try:
    import lalsimulation as _lalsim
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "puddin.lalsim requires lalsimulation to be installed.  "
        "Install it via 'conda install -c conda-forge lalsimulation'."
    ) from exc

from puddin.units import to_dimensionless, to_kg, to_rad

__all__ = ["spins_to_lalsim"]


# ---------------------------------------------------------------------------
# Internal scalar worker (vectorised below)
# ---------------------------------------------------------------------------


def _scalar_transform(
    theta_jn: float,
    phi_jl: float,
    tilt1: float,
    tilt2: float,
    phi12: float,
    a1: float,
    a2: float,
    m1: float,
    m2: float,
    f_ref: float,
    phase: float,
) -> tuple[float, float, float, float, float, float, float]:
    """Call SimInspiralTransformPrecessingNewInitialConditions for a single event.

    Aligned-spin fast path
    ----------------------
    When both spins are zero or aligned/anti-aligned with the orbital angular
    momentum (tilt ∈ {0, π}), the total and orbital angular momenta are
    parallel and the transformation reduces to:

    * ``iota = theta_jn``
    * ``S1x = S1y = 0``, ``S1z = a1 * cos(tilt1)``
    * ``S2x = S2y = 0``, ``S2z = a2 * cos(tilt2)``

    This avoids a LALSim call for the majority of production injections,
    which use quasi-aligned-spin populations.
    """
    spin1_aligned = (a1 == 0.0) or (abs(tilt1) < 1e-15) or (abs(tilt1 - np.pi) < 1e-15)
    spin2_aligned = (a2 == 0.0) or (abs(tilt2) < 1e-15) or (abs(tilt2 - np.pi) < 1e-15)

    if spin1_aligned and spin2_aligned:
        return (
            theta_jn,
            0.0,
            0.0,
            a1 * np.cos(tilt1),
            0.0,
            0.0,
            a2 * np.cos(tilt2),
        )

    result = _lalsim.SimInspiralTransformPrecessingNewInitialConditions(
        theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase
    )
    # LALSim returns a list: [iota, S1x, S1y, S1z, S2x, S2y, S2z]
    return tuple(result)  # type: ignore[return-value]


# Vectorised version: applies _scalar_transform element-wise over arrays.
# otypes specifies that all 7 outputs are float64.
_vec_transform = np.vectorize(
    _scalar_transform,
    otypes=[np.float64] * 7,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def spins_to_lalsim(
    theta_jn,
    phi_jl,
    tilt1,
    tilt2,
    phi12,
    a1,
    a2,
    m1,
    m2,
    f_ref,
    phase=0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    r"""Convert bilby-style spin parameters to LALSimulation Cartesian components.

    Wraps ``lalsimulation.SimInspiralTransformPrecessingNewInitialConditions``
    with puddin's unit-handling layer and vectorisation over arrays of events.

    The transformation converts from the **J-frame** parameterisation used by
    bilby and LALInference — where angles are defined relative to the total
    angular momentum :math:`\mathbf{J}` and the orbital angular momentum
    :math:`\mathbf{L}` — to the **L-frame** Cartesian spin components used
    directly by LALSimulation waveform generators.

    The conversion requires the orbital angular momentum magnitude
    :math:`|L|(f_\mathrm{ref})`, which LALSimulation computes internally
    via a post-Newtonian expansion at the reference gravitational-wave
    frequency :math:`f_\mathrm{ref}`.

    Aligned-spin fast path
    ----------------------
    When both spins are aligned or anti-aligned with :math:`\mathbf{L}`, or
    both spin magnitudes are zero, the LALSim call is skipped and the result
    is computed analytically:

    .. math::

        \iota = \theta_{JN}, \quad
        S_{1x} = S_{1y} = 0, \quad S_{1z} = a_1 \cos\theta_1, \quad
        S_{2x} = S_{2y} = 0, \quad S_{2z} = a_2 \cos\theta_2.

    Parameters
    ----------
    theta_jn : array-like
        Inclination of total angular momentum :math:`\mathbf{J}` relative to
        the line of sight, in radians.  Accepts astropy/pint angle Quantities.
    phi_jl : array-like
        Azimuthal angle of the orbital angular momentum :math:`\mathbf{L}`
        around :math:`\mathbf{J}`, in radians.
    tilt1, tilt2 : array-like
        Spin tilt angles :math:`\theta_1, \theta_2` of each body measured
        from :math:`\mathbf{L}`, in radians.  Values in :math:`[0, \pi]`.
    phi12 : array-like
        Azimuthal angle of spin 2 relative to spin 1 in the orbital plane,
        in radians.  Values in :math:`[0, 2\pi)`.
    a1, a2 : array-like
        Dimensionless spin magnitudes :math:`\chi_1, \chi_2 \in [0, 1]`.
    m1, m2 : array-like
        Component masses in SI (kg), or unit-carrying mass Quantities.
    f_ref : array-like
        Gravitational-wave reference frequency in Hz at which the spin
        components are defined.
    phase : array-like, optional
        Reference orbital phase in radians.  Default 0.

    Returns
    -------
    iota : numpy.ndarray
        Inclination of :math:`\mathbf{L}` relative to the line of sight in
        radians, after the J→L rotation.  Values in :math:`[0, \pi]`.
    S1x, S1y, S1z, S2x, S2y, S2z : numpy.ndarray
        Cartesian dimensionless spin components in the L-frame, each of the
        same shape as the broadcast inputs.

    Notes
    -----
    This function calls LALSimulation's
    ``SimInspiralTransformPrecessingNewInitialConditions`` for the
    precessing case, which uses leading-order post-Newtonian spin–orbit
    coupling to determine the :math:`|\mathbf{L}|` magnitude at
    :math:`f_\mathrm{ref}`.  The pure-math Newtonian approximation is
    available via :func:`puddin.orbital_angular_momentum`.

    For the bilby reference implementation see
    ``bilby.gw.conversion.bilby_to_lalsimulation_spins``.

    Examples
    --------
    >>> import numpy as np
    >>> from puddin import lalsim
    >>> import lal
    >>> iota, s1x, s1y, s1z, s2x, s2y, s2z = lalsim.spins_to_lalsim(
    ...     theta_jn=np.array([0.4]), phi_jl=np.array([0.3]),
    ...     tilt1=np.array([0.5]), tilt2=np.array([0.3]),
    ...     phi12=np.array([1.2]),
    ...     a1=np.array([0.6]), a2=np.array([0.4]),
    ...     m1=np.array([30 * lal.MSUN_SI]),
    ...     m2=np.array([20 * lal.MSUN_SI]),
    ...     f_ref=np.array([20.0]),
    ... )
    """
    theta_jn = to_rad(theta_jn)
    phi_jl   = to_rad(phi_jl)
    tilt1    = to_rad(tilt1)
    tilt2    = to_rad(tilt2)
    phi12    = to_rad(phi12)
    a1       = to_dimensionless(a1)
    a2       = to_dimensionless(a2)
    m1       = to_kg(m1)
    m2       = to_kg(m2)
    f_ref    = np.atleast_1d(np.asarray(f_ref, dtype=np.float64))
    phase    = to_rad(phase)

    iota, s1x, s1y, s1z, s2x, s2y, s2z = _vec_transform(
        theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase
    )
    return iota, s1x, s1y, s1z, s2x, s2y, s2z
