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

from puddin import spin_components
from puddin.units import to_dimensionless, to_hz, to_kg, to_rad

__all__ = ["spins_to_lalsim"]


# ---------------------------------------------------------------------------
# Internal scalar worker (vectorised below) — precessing events only
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
    """Call SimInspiralTransformPrecessingNewInitialConditions for a single event."""
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


def _is_aligned(a: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """True where a spin is zero, or aligned/anti-aligned with L (tilt in {0, pi})."""
    return (a == 0.0) | (np.abs(tilt) < 1e-15) | (np.abs(tilt - np.pi) < 1e-15)


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
    both spin magnitudes are zero, the LALSim call is skipped for those
    samples: :math:`\iota = \theta_{JN}` and the Cartesian components are
    computed directly via :func:`puddin.spin_components` (which reduces to
    :math:`S_{1x}=S_{1y}=0,\ S_{1z}=a_1\cos\theta_1` and similarly for spin 2
    when :math:`\sin\theta_i = 0`). Only genuinely precessing samples pay for
    a LALSim call, batched over the whole input array.

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
    theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref, phase = np.broadcast_arrays(
        to_rad(theta_jn),
        to_rad(phi_jl),
        to_rad(tilt1),
        to_rad(tilt2),
        to_rad(phi12),
        to_dimensionless(a1),
        to_dimensionless(a2),
        to_kg(m1),
        to_kg(m2),
        to_hz(f_ref),
        to_rad(phase),
    )

    aligned = _is_aligned(a1, tilt1) & _is_aligned(a2, tilt2)

    iota = np.zeros_like(theta_jn)
    s1x, s1y, s1z = (np.zeros_like(theta_jn) for _ in range(3))
    s2x, s2y, s2z = (np.zeros_like(theta_jn) for _ in range(3))

    if np.any(aligned):
        iota[aligned] = theta_jn[aligned]
        (
            s1x[aligned], s1y[aligned], s1z[aligned],
            s2x[aligned], s2y[aligned], s2z[aligned],
        ) = spin_components(a1[aligned], a2[aligned], tilt1[aligned], tilt2[aligned], phi12[aligned])

    precessing = ~aligned
    if np.any(precessing):
        (
            iota[precessing], s1x[precessing], s1y[precessing], s1z[precessing],
            s2x[precessing], s2y[precessing], s2z[precessing],
        ) = _vec_transform(
            theta_jn[precessing], phi_jl[precessing], tilt1[precessing], tilt2[precessing],
            phi12[precessing], a1[precessing], a2[precessing], m1[precessing], m2[precessing],
            f_ref[precessing], phase[precessing],
        )

    return iota, s1x, s1y, s1z, s2x, s2y, s2z
