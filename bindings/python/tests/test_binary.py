"""Python-level tests for the puddin binary parameter functions.

Run with: pytest bindings/python/tests/

Exercises:
- plain numpy inputs (SI)
- astropy.units.Quantity inputs
- pint.Quantity inputs (if pint is installed)
- JAX inputs (if jax is installed)
- unit mismatch raises an error
- numerical correctness matches Rust reference values
"""

from __future__ import annotations

import math
import pytest
import numpy as np

import puddin

MSUN_KG = 1.989e30  # approximate solar mass in kg


# ── helpers ───────────────────────────────────────────────────────────────────

def sol(m):
    """Return mass m (solar masses) as a plain SI float (kg)."""
    return np.atleast_1d(np.float64(m * MSUN_KG))


# ── plain numpy tests ─────────────────────────────────────────────────────────

class TestPlainSI:
    def test_total_mass(self):
        result = puddin.total_mass(sol(30), sol(20))
        assert math.isclose(result[0] / MSUN_KG, 50.0, rel_tol=1e-8)

    def test_mass_ratio(self):
        result = puddin.mass_ratio(sol(30), sol(15))
        assert math.isclose(result[0], 0.5, rel_tol=1e-8)

    def test_symmetric_mass_ratio_equal(self):
        result = puddin.symmetric_mass_ratio(sol(30), sol(30))
        assert math.isclose(result[0], 0.25, rel_tol=1e-8)

    def test_symmetric_mass_ratio_upper_bound(self):
        for m1, m2 in [(10, 5), (100, 1), (50, 50)]:
            eta = puddin.symmetric_mass_ratio(sol(m1), sol(m2))[0]
            assert eta <= 0.25 + 1e-10

    def test_chirp_mass_equal_masses(self):
        mc = puddin.chirp_mass(sol(30), sol(30))[0] / MSUN_KG
        expected = 30.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(mc, expected, rel_tol=1e-8)

    def test_chirp_mass_le_total(self):
        for m1, m2 in [(30, 30), (30, 10), (100, 1)]:
            mc = puddin.chirp_mass(sol(m1), sol(m2))[0]
            mt = puddin.total_mass(sol(m1), sol(m2))[0]
            assert mc <= mt + 1e-6

    def test_chi_eff_aligned(self):
        x = puddin.chi_eff(sol(30), sol(30), np.array([0.5]), np.array([0.5]),
                           np.array([0.0]), np.array([0.0]))[0]
        assert math.isclose(x, 0.5, rel_tol=1e-8)

    def test_chi_p_in_plane(self):
        x = puddin.chi_p(sol(30), sol(30), np.array([1.0]), np.array([0.0]),
                         np.array([math.pi / 2]), np.array([0.0]))[0]
        assert math.isclose(x, 1.0, rel_tol=1e-8)


# ── astropy tests ─────────────────────────────────────────────────────────────

class TestAstropy:
    pytest.importorskip("astropy")

    def test_chirp_mass_with_astropy_units(self):
        from astropy import units as u
        mc = puddin.chirp_mass(30 * u.Msun, 30 * u.Msun)[0] / MSUN_KG
        expected = 30.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(mc, expected, rel_tol=1e-6)

    def test_wrong_unit_raises(self):
        from astropy import units as u
        with pytest.raises(Exception):  # astropy raises UnitConversionError
            puddin.chirp_mass(30 * u.meter, 30 * u.meter)


# ── pint tests ────────────────────────────────────────────────────────────────

class TestPint:
    pytest.importorskip("pint")

    def test_chirp_mass_with_pint_units(self):
        import pint
        ureg = pint.UnitRegistry()
        m1 = 30 * ureg.solar_mass
        m2 = 30 * ureg.solar_mass
        mc = puddin.chirp_mass(m1, m2)[0] / MSUN_KG
        expected = 30.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(mc, expected, rel_tol=1e-4)

    def test_wrong_unit_raises(self):
        import pint
        ureg = pint.UnitRegistry()
        with pytest.raises(Exception):
            puddin.chirp_mass(30 * ureg.meter, 30 * ureg.meter)


# ── JAX tests ─────────────────────────────────────────────────────────────────

class TestJAX:
    jax = pytest.importorskip("jax")

    def test_chirp_mass_jit(self):
        import jax
        import jax.numpy as jnp

        m1 = jnp.array([30.0 * MSUN_KG])
        m2 = jnp.array([30.0 * MSUN_KG])
        mc_fn = jax.jit(puddin.chirp_mass)
        mc = mc_fn(m1, m2)[0] / MSUN_KG
        expected = 30.0 * 0.25 ** (3.0 / 5.0)
        assert math.isclose(float(mc), expected, rel_tol=1e-6)

    def test_chirp_mass_grad(self):
        import jax
        import jax.numpy as jnp

        m1 = jnp.array([30.0 * MSUN_KG])
        m2 = jnp.array([30.0 * MSUN_KG])

        def scalar_mc(m1, m2):
            return puddin.chirp_mass(m1, m2)[0]

        grad = jax.grad(scalar_mc)(m1, m2)
        # dMc/dm1 at equal masses = Mc * (3/(5*m1) - 1/(5*M))
        # = Mc * (3/5 - 1/10) / m1 = Mc * 1/2 / m1
        mc = puddin.chirp_mass(m1, m2)[0]
        expected_grad = mc * (3.0 / (5.0 * m1[0]) - 1.0 / (5.0 * 2.0 * m1[0]))
        assert math.isclose(float(grad[0]), float(expected_grad), rel_tol=1e-5)
