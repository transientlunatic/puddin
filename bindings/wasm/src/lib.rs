//! WebAssembly bindings for Puddin.
//!
//! This module exposes the core Puddin functions to JavaScript and TypeScript
//! via `wasm-bindgen`.  All functions accept and return `Float64Array`s in SI
//! units (kg for mass, radians for angles, dimensionless otherwise).
//!
//! Build with:
//! ```sh
//! wasm-pack build --target bundler bindings/wasm
//! ```
//!
//! The output package in `bindings/wasm/pkg/` includes TypeScript declaration
//! files (`.d.ts`) generated automatically by `wasm-bindgen`.

use puddin::binary;
use uom::si::f64::Mass;
use uom::si::mass::kilogram;
use wasm_bindgen::prelude::*;

// ── helpers ───────────────────────────────────────────────────────────────────

fn to_masses(arr: &[f64]) -> Vec<Mass> {
    arr.iter().map(|&v| Mass::new::<kilogram>(v)).collect()
}

// ── binary parameters ─────────────────────────────────────────────────────────

/// Total mass $M = m_1 + m_2$ (kg).
///
/// @param m1 - Component mass 1 in kilograms.
/// @param m2 - Component mass 2 in kilograms.
/// @returns Total mass in kilograms.
#[wasm_bindgen]
pub fn total_mass(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::total_mass(a, b).get::<kilogram>())
        .collect()
}

/// Mass ratio $q = m_2 / m_1$ (dimensionless).
///
/// @param m1 - Primary mass in kilograms ($m_1 \geq m_2$).
/// @param m2 - Secondary mass in kilograms.
/// @returns Dimensionless mass ratio.
#[wasm_bindgen]
pub fn mass_ratio(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::mass_ratio(a, b))
        .collect()
}

/// Symmetric mass ratio $\eta = m_1 m_2 / M^2 \in (0, 1/4]$ (dimensionless).
///
/// @param m1 - Component mass 1 in kilograms.
/// @param m2 - Component mass 2 in kilograms.
/// @returns Dimensionless symmetric mass ratio.
#[wasm_bindgen]
pub fn symmetric_mass_ratio(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::symmetric_mass_ratio(a, b))
        .collect()
}

/// Chirp mass $\mathcal{M} = (m_1 m_2)^{3/5} / M^{1/5}$ (kg).
///
/// @param m1 - Component mass 1 in kilograms.
/// @param m2 - Component mass 2 in kilograms.
/// @returns Chirp mass in kilograms.
#[wasm_bindgen]
pub fn chirp_mass(m1: Vec<f64>, m2: Vec<f64>) -> Vec<f64> {
    to_masses(&m1)
        .into_iter()
        .zip(to_masses(&m2))
        .map(|(a, b)| binary::chirp_mass(a, b).get::<kilogram>())
        .collect()
}

/// Effective inspiral spin $\chi_\mathrm{eff} \in [-1, 1]$.
///
/// @param m1    - Component mass 1 in kilograms.
/// @param m2    - Component mass 2 in kilograms.
/// @param a1    - Dimensionless spin magnitude of body 1 (0–1).
/// @param a2    - Dimensionless spin magnitude of body 2 (0–1).
/// @param tilt1 - Spin tilt angle of body 1 (radians, 0–π).
/// @param tilt2 - Spin tilt angle of body 2 (radians, 0–π).
/// @returns Dimensionless effective inspiral spin.
#[wasm_bindgen]
pub fn chi_eff(
    m1: Vec<f64>,
    m2: Vec<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
    tilt1: Vec<f64>,
    tilt2: Vec<f64>,
) -> Vec<f64> {
    let n = m1.len();
    let masses1 = to_masses(&m1);
    let masses2 = to_masses(&m2);
    (0..n)
        .map(|i| binary::chi_eff(masses1[i], masses2[i], a1[i], a2[i], tilt1[i], tilt2[i]))
        .collect()
}

/// Effective precession spin $\chi_p \in [0, 1]$.
///
/// **Note:** requires $m_1 \geq m_2$.
///
/// @param m1    - Primary mass in kilograms ($m_1 \geq m_2$).
/// @param m2    - Secondary mass in kilograms.
/// @param a1    - Dimensionless spin magnitude of body 1 (0–1).
/// @param a2    - Dimensionless spin magnitude of body 2 (0–1).
/// @param tilt1 - Spin tilt angle of body 1 (radians, 0–π).
/// @param tilt2 - Spin tilt angle of body 2 (radians, 0–π).
/// @returns Dimensionless effective precession spin.
#[wasm_bindgen]
pub fn chi_p(
    m1: Vec<f64>,
    m2: Vec<f64>,
    a1: Vec<f64>,
    a2: Vec<f64>,
    tilt1: Vec<f64>,
    tilt2: Vec<f64>,
) -> Vec<f64> {
    let n = m1.len();
    let masses1 = to_masses(&m1);
    let masses2 = to_masses(&m2);
    (0..n)
        .map(|i| binary::chi_p(masses1[i], masses2[i], a1[i], a2[i], tilt1[i], tilt2[i]))
        .collect()
}
