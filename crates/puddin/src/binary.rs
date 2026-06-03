//! Binary system parameters.
//!
//! This module provides functions for converting between the parameterisations
//! commonly used to describe compact binary systems in gravitational-wave
//! astronomy.  All quantities are expressed using SI units via the [`uom`]
//! crate, which enforces dimensional correctness at compile time.
//!
//! # Conventions
//!
//! - Component masses are labelled $m_1 \geq m_2 > 0$.
//! - Spin magnitudes are $a_i \in [0, 1]$ (dimensionless, normalised to the
//!   Kerr maximum).
//! - Spin tilt angles $\theta_i$ are measured from the orbital angular momentum
//!   axis, $\theta_i \in [0, \pi]$.
//! - Angles are in radians.

use uom::si::f64::*;
use uom::si::mass::kilogram;

// ── Total mass ───────────────────────────────────────────────────────────────

/// Total mass $M = m_1 + m_2$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::total_mass;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(20.0 * MSUN);
/// let m = total_mass(m1, m2);
/// assert!((m.get::<kilogram>() - 50.0 * MSUN).abs() < 1e6);
/// ```
pub fn total_mass(m1: Mass, m2: Mass) -> Mass {
    m1 + m2
}

// ── Mass ratio ───────────────────────────────────────────────────────────────

/// Mass ratio $q = m_2 / m_1$, where $m_1 \geq m_2$ so $q \in (0, 1]$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::mass_ratio;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(15.0 * MSUN);
/// let q = mass_ratio(m1, m2);
/// assert!((q - 0.5).abs() < 1e-10);
/// ```
pub fn mass_ratio(m1: Mass, m2: Mass) -> f64 {
    m2.get::<kilogram>() / m1.get::<kilogram>()
}

// ── Symmetric mass ratio ─────────────────────────────────────────────────────

/// Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$.
///
/// Equal-mass systems have $\eta = 1/4$; highly asymmetric systems have
/// $\eta \to 0$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::symmetric_mass_ratio;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// let eta = symmetric_mass_ratio(m1, m2);
/// assert!((eta - 0.25).abs() < 1e-10);
/// ```
pub fn symmetric_mass_ratio(m1: Mass, m2: Mass) -> f64 {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    (m1_kg * m2_kg) / (m_kg * m_kg)
}

// ── Chirp mass ───────────────────────────────────────────────────────────────

/// Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$.
///
/// The chirp mass is the combination of masses that governs the leading-order
/// gravitational-wave frequency evolution during inspiral.  It is always less
/// than or equal to the total mass.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::chirp_mass;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// let mc = chirp_mass(m1, m2);
/// // For equal masses: Mc = M * (1/4)^(3/5) = 2m * (1/4)^(3/5)
/// let expected_kg = 2.0 * 30.0 * MSUN * 0.25_f64.powf(3.0 / 5.0);
/// assert!((mc.get::<kilogram>() - expected_kg).abs() / expected_kg < 1e-10);
/// ```
pub fn chirp_mass(m1: Mass, m2: Mass) -> Mass {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    let mc_kg = (m1_kg * m2_kg).powf(3.0 / 5.0) / m_kg.powf(1.0 / 5.0);
    Mass::new::<kilogram>(mc_kg)
}

// ── Inverse mass transforms ──────────────────────────────────────────────────

/// Recover component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and
/// mass ratio $q = m_2/m_1$.
///
/// This is the inverse of `chirp_mass` + `mass_ratio`.  The returned masses
/// satisfy $m_1 \geq m_2$ whenever $q \in (0, 1]$.
///
/// # Arguments
///
/// * `mc` — chirp mass $\mathcal{M}$ (SI kg).
/// * `q`  — mass ratio $q = m_2/m_1 \in (0, 1]$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::{chirp_mass, mass_ratio, masses_from_chirp_mass_q};
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1_in = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2_in = Mass::new::<kilogram>(20.0 * MSUN);
/// let mc = chirp_mass(m1_in, m2_in);
/// let q  = mass_ratio(m1_in, m2_in);
/// let (m1_out, m2_out) = masses_from_chirp_mass_q(mc, q);
/// assert!((m1_out.get::<kilogram>() - m1_in.get::<kilogram>()).abs() / m1_in.get::<kilogram>() < 1e-10);
/// assert!((m2_out.get::<kilogram>() - m2_in.get::<kilogram>()).abs() / m2_in.get::<kilogram>() < 1e-10);
/// ```
pub fn masses_from_chirp_mass_q(mc: Mass, q: f64) -> (Mass, Mass) {
    debug_assert!(
        q > 0.0 && q <= 1.0,
        "mass ratio q must be in (0, 1], got q={q}"
    );
    let eta = q / (1.0 + q).powi(2);
    let m_kg = mc.get::<kilogram>() / eta.powf(3.0 / 5.0);
    let m1_kg = m_kg / (1.0 + q);
    let m2_kg = m1_kg * q;
    (Mass::new::<kilogram>(m1_kg), Mass::new::<kilogram>(m2_kg))
}

/// Recover component masses $(m_1, m_2)$ from chirp mass $\mathcal{M}$ and
/// symmetric mass ratio $\eta$.
///
/// This is the inverse of `chirp_mass` + `symmetric_mass_ratio`.  The
/// quadratic $x^2 - x + \eta = 0$ gives $m_1/M$ and $m_2/M$; the larger
/// root is assigned to $m_1$ so that $m_1 \geq m_2$.
///
/// For equal-mass systems ($\eta = 1/4$) the two roots coincide and
/// $m_1 = m_2$.
///
/// # Arguments
///
/// * `mc`  — chirp mass $\mathcal{M}$ (SI kg).
/// * `eta` — symmetric mass ratio $\eta \in (0, 1/4]$.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::{chirp_mass, symmetric_mass_ratio, masses_from_chirp_mass_eta};
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1_in = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2_in = Mass::new::<kilogram>(20.0 * MSUN);
/// let mc  = chirp_mass(m1_in, m2_in);
/// let eta = symmetric_mass_ratio(m1_in, m2_in);
/// let (m1_out, m2_out) = masses_from_chirp_mass_eta(mc, eta);
/// assert!((m1_out.get::<kilogram>() - m1_in.get::<kilogram>()).abs() / m1_in.get::<kilogram>() < 1e-10);
/// assert!((m2_out.get::<kilogram>() - m2_in.get::<kilogram>()).abs() / m2_in.get::<kilogram>() < 1e-10);
/// ```
pub fn masses_from_chirp_mass_eta(mc: Mass, eta: f64) -> (Mass, Mass) {
    debug_assert!(
        eta > 0.0 && eta <= 0.25 + 1e-12,
        "symmetric mass ratio eta must be in (0, 0.25], got eta={eta}"
    );
    let eta = eta.min(0.25); // clamp floating-point noise at equal mass
    let m_kg = mc.get::<kilogram>() / eta.powf(3.0 / 5.0);
    let disc = (1.0 - 4.0 * eta).max(0.0).sqrt();
    let m1_kg = m_kg * (1.0 + disc) / 2.0;
    let m2_kg = m_kg * (1.0 - disc) / 2.0;
    (Mass::new::<kilogram>(m1_kg), Mass::new::<kilogram>(m2_kg))
}

// ── Effective inspiral spin ──────────────────────────────────────────────────

/// Effective inspiral spin parameter
/// $\chi_\mathrm{eff} = (m_1 a_1 \cos\theta_1 + m_2 a_2 \cos\theta_2) / M$.
///
/// $\chi_\mathrm{eff} \in [-1, 1]$ and is approximately conserved through
/// inspiral at 1.5 post-Newtonian order.
///
/// # Arguments
///
/// * `m1`, `m2` — component masses ($m_1 \geq m_2$).
/// * `a1`, `a2` — dimensionless spin magnitudes $\in [0, 1]$.
/// * `tilt1`, `tilt2` — spin tilt angles (radians) with respect to the orbital
///   angular momentum axis.
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::chi_eff;
/// use std::f64::consts::PI;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// // Both spins aligned, magnitude 0.5 -> chi_eff = 0.5
/// let x = chi_eff(m1, m2, 0.5, 0.5, 0.0, 0.0);
/// assert!((x - 0.5).abs() < 1e-10);
///
/// // Anti-aligned -> chi_eff = -0.5
/// let x = chi_eff(m1, m2, 0.5, 0.5, PI, PI);
/// assert!((x - (-0.5)).abs() < 1e-10);
/// ```
pub fn chi_eff(m1: Mass, m2: Mass, a1: f64, a2: f64, tilt1: f64, tilt2: f64) -> f64 {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    (m1_kg * a1 * tilt1.cos() + m2_kg * a2 * tilt2.cos()) / m_kg
}

// ── Effective precession spin ────────────────────────────────────────────────

/// Effective precession spin parameter
/// $\chi_p = \max\!\bigl(a_1 \sin\theta_1,\; \tfrac{3+4q}{4(1+q)} q \, a_2 \sin\theta_2\bigr)$
///
/// as defined in [Hannam et al. (2014)](https://doi.org/10.1103/PhysRevLett.113.151101).
/// $\chi_p \in [0, 1]$.
///
/// # Arguments
///
/// * `m1`, `m2` — component masses.  **Must satisfy $m_1 \geq m_2$**; the
///   formula is undefined (and not bounded) for $m_2 > m_1$.
/// * `a1`, `a2` — dimensionless spin magnitudes $\in [0, 1]$.
/// * `tilt1`, `tilt2` — spin tilt angles (radians).
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::chi_p;
/// use std::f64::consts::FRAC_PI_2;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(30.0 * MSUN);
/// // Primary spin fully in-plane, secondary aligned -> chi_p = a1 sin(pi/2) = 1.0
/// let x = chi_p(m1, m2, 1.0, 0.0, FRAC_PI_2, 0.0);
/// assert!((x - 1.0).abs() < 1e-10);
/// ```
pub fn chi_p(m1: Mass, m2: Mass, a1: f64, a2: f64, tilt1: f64, tilt2: f64) -> f64 {
    debug_assert!(
        m1.get::<kilogram>() >= m2.get::<kilogram>(),
        "chi_p requires m1 >= m2 (got m1={}, m2={})",
        m1.get::<kilogram>(),
        m2.get::<kilogram>()
    );
    let q = mass_ratio(m1, m2); // m2/m1 <= 1
    let term1 = a1 * tilt1.sin();
    let term2 = (3.0 + 4.0 * q) / (4.0 * (1.0 + q)) * q * a2 * tilt2.sin();
    term1.max(term2)
}

// ── Spin components ──────────────────────────────────────────────────────────

/// Decompose spin tilts and azimuths into Cartesian components in the L-frame.
///
/// Converts from the bilby / LALInference spin parameterisation
/// (dimensionless magnitude + tilt angle + relative azimuth) to the
/// Cartesian spin components used by LALSimulation.
///
/// The **L-frame** has its z-axis aligned with the Newtonian orbital angular
/// momentum **L̂**.  By convention spin 1 is placed in the x-z plane
/// (φ₁ = 0), so S₁ᵧ = 0 identically.  Spin 2 is rotated by φ₁₂ around the
/// z-axis relative to spin 1:
///
/// ```text
/// S₁ = a₁ (sin θ₁,  0,              cos θ₁)
/// S₂ = a₂ (sin θ₂ cos φ₁₂,  sin θ₂ sin φ₁₂,  cos θ₂)
/// ```
///
/// # Arguments
///
/// * `a1`, `a2`    — dimensionless spin magnitudes χ₁, χ₂ ∈ [0, 1].
/// * `tilt1`, `tilt2` — spin tilt angles θ₁, θ₂ ∈ [0, π] (radians).
/// * `phi12`       — azimuthal angle of spin 2 relative to spin 1 ∈ [0, 2π) (radians).
///
/// # Returns
///
/// `(S1x, S1y, S1z, S2x, S2y, S2z)` — dimensionless Cartesian components.
///
/// # Examples
///
/// ```
/// use puddin::binary::spin_components;
/// use std::f64::consts::FRAC_PI_2;
///
/// // Aligned spins: both along z-axis
/// let (s1x, s1y, s1z, s2x, s2y, s2z) = spin_components(0.5, 0.3, 0.0, 0.0, 0.0);
/// assert!(s1x.abs() < 1e-14 && s1y.abs() < 1e-14);
/// assert!((s1z - 0.5).abs() < 1e-14);
/// assert!((s2z - 0.3).abs() < 1e-14);
///
/// // In-plane spin 1: tilt = π/2 → S1x = a1
/// let (s1x, s1y, s1z, _, _, _) = spin_components(0.8, 0.0, FRAC_PI_2, 0.0, 0.0);
/// assert!((s1x - 0.8).abs() < 1e-14);
/// assert!(s1z.abs() < 1e-14);
/// ```
pub fn spin_components(
    a1: f64,
    a2: f64,
    tilt1: f64,
    tilt2: f64,
    phi12: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let s1x = a1 * tilt1.sin();
    let s1y = 0.0_f64;
    let s1z = a1 * tilt1.cos();
    let s2x = a2 * tilt2.sin() * phi12.cos();
    let s2y = a2 * tilt2.sin() * phi12.sin();
    let s2z = a2 * tilt2.cos();
    (s1x, s1y, s1z, s2x, s2y, s2z)
}

// ── Orbital angular momentum ─────────────────────────────────────────────────

/// Gravitational constant in SI units (CODATA 2014, consistent with LALSuite).
const G_SI: f64 = 6.674_30e-11; // m³ kg⁻¹ s⁻²

/// Newtonian orbital angular momentum magnitude at a reference frequency.
///
/// Computes the leading-order (Newtonian) orbital angular momentum
///
/// ```text
/// |L_N| = μ (G M)^{2/3} / (π f_ref)^{1/3}
/// ```
///
/// where μ = m₁ m₂ / M is the reduced mass and M = m₁ + m₂ is the total mass.
///
/// # Arguments
///
/// * `m1`, `m2` — component masses (SI: kg).
/// * `f_ref`    — gravitational-wave reference frequency (Hz).
///
/// # Returns
///
/// `|L_N|` in SI units (kg m² s⁻¹).
///
/// # Examples
///
/// ```
/// use uom::si::f64::Mass;
/// use uom::si::mass::kilogram;
/// use puddin::binary::orbital_angular_momentum;
///
/// const MSUN: f64 = 1.988_416e30;
/// let m1 = Mass::new::<kilogram>(30.0 * MSUN);
/// let m2 = Mass::new::<kilogram>(20.0 * MSUN);
/// let l = orbital_angular_momentum(m1, m2, 20.0);
/// assert!(l > 0.0);
/// ```
pub fn orbital_angular_momentum(m1: Mass, m2: Mass, f_ref: f64) -> f64 {
    let m1_kg = m1.get::<kilogram>();
    let m2_kg = m2.get::<kilogram>();
    let m_kg = m1_kg + m2_kg;
    let mu_kg = m1_kg * m2_kg / m_kg;
    mu_kg * (G_SI * m_kg).powf(2.0 / 3.0) / (std::f64::consts::PI * f_ref).powf(1.0 / 3.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Approximate solar mass in kilograms (IAU 2015 nominal).
    const MSUN_KG: f64 = 1.988_416e30;

    fn solar(m: f64) -> Mass {
        Mass::new::<kilogram>(m * MSUN_KG)
    }

    // ── total_mass ────────────────────────────────────────────────────────────

    #[test]
    fn total_mass_basic() {
        let m = total_mass(solar(30.0), solar(20.0));
        assert!((m.get::<kilogram>() / MSUN_KG - 50.0).abs() < 1e-10);
    }

    #[test]
    fn total_mass_equal() {
        let m = total_mass(solar(15.0), solar(15.0));
        assert!((m.get::<kilogram>() / MSUN_KG - 30.0).abs() < 1e-10);
    }

    // ── mass_ratio ────────────────────────────────────────────────────────────

    #[test]
    fn mass_ratio_half() {
        let q = mass_ratio(solar(30.0), solar(15.0));
        assert!((q - 0.5).abs() < 1e-10);
    }

    #[test]
    fn mass_ratio_equal_is_one() {
        let q = mass_ratio(solar(20.0), solar(20.0));
        assert!((q - 1.0).abs() < 1e-10);
    }

    // ── symmetric_mass_ratio ──────────────────────────────────────────────────

    #[test]
    fn eta_equal_mass_is_quarter() {
        let eta = symmetric_mass_ratio(solar(30.0), solar(30.0));
        assert!((eta - 0.25).abs() < 1e-10);
    }

    #[test]
    fn eta_never_exceeds_quarter() {
        for (m1, m2) in [(10.0, 5.0), (100.0, 1.0), (50.0, 50.0), (3.0, 1.0)] {
            let eta = symmetric_mass_ratio(solar(m1), solar(m2));
            assert!(eta <= 0.25 + 1e-12, "eta={eta} for m1={m1} m2={m2}");
            assert!(eta > 0.0, "eta must be positive for m1={m1} m2={m2}");
        }
    }

    // ── chirp_mass ────────────────────────────────────────────────────────────

    #[test]
    fn chirp_mass_equal_masses() {
        let m = 30.0_f64;
        let mc = chirp_mass(solar(m), solar(m));
        // Mc = M * eta^(3/5) = 2m * (1/4)^(3/5)
        let expected_msun = 2.0 * m * 0.25_f64.powf(3.0 / 5.0);
        let mc_msun = mc.get::<kilogram>() / MSUN_KG;
        // Use relative tolerance: floating-point errors scale with the magnitude
        assert!(
            (mc_msun - expected_msun).abs() / expected_msun < 1e-10,
            "mc_msun={mc_msun} expected={expected_msun}"
        );
    }

    #[test]
    fn chirp_mass_never_exceeds_total() {
        for (m1, m2) in [(30.0, 30.0), (30.0, 10.0), (100.0, 1.0)] {
            let mc = chirp_mass(solar(m1), solar(m2));
            let mt = total_mass(solar(m1), solar(m2));
            assert!(
                mc.get::<kilogram>() <= mt.get::<kilogram>() + 1e-6,
                "Mc > M for m1={m1} m2={m2}"
            );
        }
    }

    // ── chi_eff ───────────────────────────────────────────────────────────────

    #[test]
    fn chi_eff_aligned() {
        // Both spins fully aligned: chi_eff = a (for equal masses)
        let x = chi_eff(solar(30.0), solar(30.0), 0.5, 0.5, 0.0, 0.0);
        assert!((x - 0.5).abs() < 1e-10);
    }

    #[test]
    fn chi_eff_anti_aligned() {
        let x = chi_eff(
            solar(30.0),
            solar(30.0),
            0.5,
            0.5,
            std::f64::consts::PI,
            std::f64::consts::PI,
        );
        assert!((x - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn chi_eff_zero_spins() {
        let x = chi_eff(solar(30.0), solar(10.0), 0.0, 0.0, 0.0, 0.0);
        assert!(x.abs() < 1e-10);
    }

    // ── chi_p ─────────────────────────────────────────────────────────────────

    #[test]
    fn chi_p_in_plane_primary() {
        let x = chi_p(
            solar(30.0),
            solar(30.0),
            1.0,
            0.0,
            std::f64::consts::FRAC_PI_2,
            0.0,
        );
        assert!((x - 1.0).abs() < 1e-10);
    }

    #[test]
    fn chi_p_aligned_spins_is_zero() {
        let x = chi_p(solar(30.0), solar(30.0), 1.0, 1.0, 0.0, 0.0);
        assert!(x.abs() < 1e-10);
    }

    // ── masses_from_chirp_mass_q ──────────────────────────────────────────────

    #[test]
    fn masses_from_mc_q_roundtrip_equal() {
        let m1 = solar(30.0);
        let m2 = solar(30.0);
        let (r1, r2) = masses_from_chirp_mass_q(chirp_mass(m1, m2), mass_ratio(m1, m2));
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_q_roundtrip_asymmetric() {
        let m1 = solar(36.0);
        let m2 = solar(12.0);
        let (r1, r2) = masses_from_chirp_mass_q(chirp_mass(m1, m2), mass_ratio(m1, m2));
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_q_ordering() {
        // m1 >= m2 must hold for any q in (0, 1]
        let (m1, m2) = masses_from_chirp_mass_q(solar(26.0), 0.3);
        assert!(m1.get::<kilogram>() >= m2.get::<kilogram>());
    }

    // ── masses_from_chirp_mass_eta ────────────────────────────────────────────

    #[test]
    fn masses_from_mc_eta_roundtrip_equal() {
        let m1 = solar(30.0);
        let m2 = solar(30.0);
        let mc = chirp_mass(m1, m2);
        let eta = symmetric_mass_ratio(m1, m2);
        let (r1, r2) = masses_from_chirp_mass_eta(mc, eta);
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_eta_roundtrip_asymmetric() {
        let m1 = solar(40.0);
        let m2 = solar(10.0);
        let mc = chirp_mass(m1, m2);
        let eta = symmetric_mass_ratio(m1, m2);
        let (r1, r2) = masses_from_chirp_mass_eta(mc, eta);
        assert!((r1.get::<kilogram>() - m1.get::<kilogram>()).abs() / m1.get::<kilogram>() < 1e-10);
        assert!((r2.get::<kilogram>() - m2.get::<kilogram>()).abs() / m2.get::<kilogram>() < 1e-10);
    }

    #[test]
    fn masses_from_mc_eta_ordering() {
        let (m1, m2) = masses_from_chirp_mass_eta(solar(20.0), 0.18);
        assert!(m1.get::<kilogram>() >= m2.get::<kilogram>());
    }

    // ── spin_components ───────────────────────────────────────────────────────

    #[test]
    fn spin_components_aligned() {
        let (s1x, s1y, s1z, s2x, s2y, s2z) = spin_components(0.5, 0.3, 0.0, 0.0, 0.0);
        assert!(s1x.abs() < 1e-14, "S1x={s1x}");
        assert!(s1y.abs() < 1e-14, "S1y={s1y}");
        assert!((s1z - 0.5).abs() < 1e-14, "S1z={s1z}");
        assert!(s2x.abs() < 1e-14, "S2x={s2x}");
        assert!(s2y.abs() < 1e-14, "S2y={s2y}");
        assert!((s2z - 0.3).abs() < 1e-14, "S2z={s2z}");
    }

    #[test]
    fn spin_components_antialigned_s1() {
        let (_, _, s1z, _, _, _) = spin_components(0.6, 0.0, std::f64::consts::PI, 0.0, 0.0);
        assert!((s1z - (-0.6)).abs() < 1e-14, "S1z={s1z}");
    }

    #[test]
    fn spin_components_in_plane_s1() {
        let (s1x, s1y, s1z, _, _, _) =
            spin_components(0.8, 0.0, std::f64::consts::FRAC_PI_2, 0.0, 0.0);
        assert!((s1x - 0.8).abs() < 1e-14, "S1x={s1x}");
        assert!(s1y.abs() < 1e-14, "S1y={s1y}");
        assert!(s1z.abs() < 1e-14, "S1z={s1z}");
    }

    #[test]
    fn spin_components_s1y_always_zero() {
        for (a1, t1, phi) in [(0.5, 0.3, 1.2), (0.9, 2.1, 0.0), (0.0, 1.0, 3.0)] {
            let (_, s1y, _, _, _, _) = spin_components(a1, 0.0, t1, 0.0, phi);
            assert!(s1y.abs() < 1e-14, "S1y={s1y} for a1={a1} t1={t1} phi={phi}");
        }
    }

    #[test]
    fn spin_components_s2_phi12_quarter_turn() {
        // tilt2 = π/2, phi12 = π/2 → S2 along y
        let (_, _, _, s2x, s2y, s2z) =
            spin_components(0.0, 0.5, 0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        assert!(s2x.abs() < 1e-14, "S2x={s2x}");
        assert!((s2y - 0.5).abs() < 1e-14, "S2y={s2y}");
        assert!(s2z.abs() < 1e-14, "S2z={s2z}");
    }

    // ── orbital_angular_momentum ─────────────────────────────────────────────

    #[test]
    fn oam_positive() {
        let l = orbital_angular_momentum(solar(30.0), solar(20.0), 20.0);
        assert!(l > 0.0);
    }

    #[test]
    fn oam_newtonian_formula() {
        // |L_N| = μ (G M)^{2/3} / (π f)^{1/3}
        let m1 = 30.0 * MSUN_KG;
        let m2 = 20.0 * MSUN_KG;
        let f = 20.0_f64;
        let m = m1 + m2;
        let mu = m1 * m2 / m;
        let expected = mu * (G_SI * m).powf(2.0 / 3.0) / (std::f64::consts::PI * f).powf(1.0 / 3.0);
        let got = orbital_angular_momentum(solar(30.0), solar(20.0), f);
        assert!((got / expected - 1.0).abs() < 1e-10, "got={got} expected={expected}");
    }

    #[test]
    fn oam_scales_as_f_minus_one_third() {
        // L(f) / L(8f) = 2
        let m1 = solar(30.0);
        let m2 = solar(30.0);
        let l_lo = orbital_angular_momentum(m1, m2, 20.0);
        let l_hi = orbital_angular_momentum(m1, m2, 160.0);
        assert!((l_lo / l_hi - 2.0).abs() < 1e-10);
    }

    #[test]
    fn oam_symmetric_in_masses() {
        let l_ab = orbital_angular_momentum(solar(30.0), solar(20.0), 20.0);
        let l_ba = orbital_angular_momentum(solar(20.0), solar(30.0), 20.0);
        assert!((l_ab / l_ba - 1.0).abs() < 1e-12);
    }

    // ── Property tests ────────────────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_eta_in_range(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0
        ) {
            let eta = symmetric_mass_ratio(solar(m1), solar(m2));
            prop_assert!(eta > 0.0 && eta <= 0.25 + 1e-12);
        }

        #[test]
        fn prop_chirp_mass_le_total(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0
        ) {
            let mc = chirp_mass(solar(m1), solar(m2)).get::<kilogram>();
            let mt = total_mass(solar(m1), solar(m2)).get::<kilogram>();
            prop_assert!(mc <= mt + 1e-6);
        }

        #[test]
        fn prop_mass_ratio_in_range(
            m1 in 1.0_f64..200.0,
            m2 in 0.01_f64..200.0
        ) {
            // q is m2/m1, so no ordering constraint; just check it's positive
            let q = mass_ratio(solar(m1), solar(m2));
            prop_assert!(q > 0.0);
        }

        #[test]
        fn prop_chi_eff_in_range(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            a1 in 0.0_f64..=1.0,
            a2 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            tilt2 in 0.0_f64..=std::f64::consts::PI,
        ) {
            let x = chi_eff(solar(m1), solar(m2), a1, a2, tilt1, tilt2);
            prop_assert!(x >= -1.0 - 1e-10 && x <= 1.0 + 1e-10);
        }

        #[test]
        fn prop_chi_p_in_range(
            // Enforce m1 >= m2 as required by the chi_p definition
            m2 in 1.0_f64..200.0,
            dm in 0.0_f64..200.0,
            a1 in 0.0_f64..=1.0,
            a2 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            tilt2 in 0.0_f64..=std::f64::consts::PI,
        ) {
            let m1 = m2 + dm; // guarantees m1 >= m2
            let x = chi_p(solar(m1), solar(m2), a1, a2, tilt1, tilt2);
            prop_assert!(x >= 0.0 - 1e-10 && x <= 1.0 + 1e-10,
                "chi_p={x} out of [0,1] for m1={m1} m2={m2} a1={a1} a2={a2} tilt1={tilt1} tilt2={tilt2}");
        }

        #[test]
        fn prop_masses_from_mc_q_roundtrip(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
        ) {
            prop_assume!(m1 >= m2);
            let mc = chirp_mass(solar(m1), solar(m2));
            let q  = mass_ratio(solar(m1), solar(m2));
            let (r1, r2) = masses_from_chirp_mass_q(mc, q);
            prop_assert!((r1.get::<kilogram>() / (m1 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m1 roundtrip failed: got {} expected {}", r1.get::<kilogram>() / MSUN_KG, m1);
            prop_assert!((r2.get::<kilogram>() / (m2 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m2 roundtrip failed: got {} expected {}", r2.get::<kilogram>() / MSUN_KG, m2);
        }

        #[test]
        fn prop_spin_components_s1_magnitude(
            a1 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            phi12 in 0.0_f64..=(2.0 * std::f64::consts::PI),
        ) {
            let (s1x, s1y, s1z, _, _, _) = spin_components(a1, 0.0, tilt1, 0.0, phi12);
            let mag2 = s1x*s1x + s1y*s1y + s1z*s1z;
            prop_assert!((mag2 - a1*a1).abs() < 1e-12,
                "|S1|²={mag2} ≠ a1²={} for a1={a1} tilt1={tilt1}", a1*a1);
        }

        #[test]
        fn prop_spin_components_s2_magnitude(
            a2 in 0.0_f64..=1.0,
            tilt2 in 0.0_f64..=std::f64::consts::PI,
            phi12 in 0.0_f64..=(2.0 * std::f64::consts::PI),
        ) {
            let (_, _, _, s2x, s2y, s2z) = spin_components(0.0, a2, 0.0, tilt2, phi12);
            let mag2 = s2x*s2x + s2y*s2y + s2z*s2z;
            prop_assert!((mag2 - a2*a2).abs() < 1e-12,
                "|S2|²={mag2} ≠ a2²={} for a2={a2} tilt2={tilt2} phi12={phi12}", a2*a2);
        }

        #[test]
        fn prop_spin_components_s1y_zero(
            a1 in 0.0_f64..=1.0,
            tilt1 in 0.0_f64..=std::f64::consts::PI,
            phi12 in 0.0_f64..=(2.0 * std::f64::consts::PI),
        ) {
            let (_, s1y, _, _, _, _) = spin_components(a1, 0.0, tilt1, 0.0, phi12);
            prop_assert!(s1y.abs() < 1e-14, "S1y={s1y} ≠ 0 for a1={a1} tilt1={tilt1}");
        }

        #[test]
        fn prop_oam_positive(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            f_ref in 1.0_f64..200.0,
        ) {
            let l = orbital_angular_momentum(solar(m1), solar(m2), f_ref);
            prop_assert!(l > 0.0, "L={l} not positive for m1={m1} m2={m2} f={f_ref}");
        }

        #[test]
        fn prop_oam_symmetric(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
            f_ref in 1.0_f64..200.0,
        ) {
            let l_ab = orbital_angular_momentum(solar(m1), solar(m2), f_ref);
            let l_ba = orbital_angular_momentum(solar(m2), solar(m1), f_ref);
            prop_assert!((l_ab / l_ba - 1.0).abs() < 1e-10,
                "L not symmetric: L(m1,m2)={l_ab} L(m2,m1)={l_ba}");
        }

        #[test]
        fn prop_masses_from_mc_eta_roundtrip(
            m1 in 1.0_f64..200.0,
            m2 in 1.0_f64..200.0,
        ) {
            prop_assume!(m1 >= m2);
            let mc  = chirp_mass(solar(m1), solar(m2));
            let eta = symmetric_mass_ratio(solar(m1), solar(m2));
            let (r1, r2) = masses_from_chirp_mass_eta(mc, eta);
            prop_assert!((r1.get::<kilogram>() / (m1 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m1 roundtrip failed: got {} expected {}", r1.get::<kilogram>() / MSUN_KG, m1);
            prop_assert!((r2.get::<kilogram>() / (m2 * MSUN_KG) - 1.0).abs() < 1e-9,
                "m2 roundtrip failed: got {} expected {}", r2.get::<kilogram>() / MSUN_KG, m2);
        }
    }
}
