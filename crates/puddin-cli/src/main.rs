use clap::{ArgAction, Parser, Subcommand};
use puddin::binary::{chi_eff, chi_p, chirp_mass, mass_ratio, symmetric_mass_ratio, total_mass};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

const MSUN: f64 = 1.988_416e30;

/// Gravitational-wave binary parameter calculator.
///
/// Masses are in solar masses (M☉) unless --si is given, in which case they
/// are in kilograms.  Spin magnitudes are dimensionless (0–1).  Tilt angles
/// are in radians.
///
/// Output is a single number on stdout, suitable for shell pipelines.
/// Use --verbose to include a label and units.
#[derive(Parser)]
#[command(name = "puddin", version, author)]
struct Cli {
    /// Accept and emit mass values in kilograms instead of solar masses
    #[arg(long, global = true, action = ArgAction::SetTrue)]
    si: bool,

    /// Print a label and units alongside the result
    #[arg(long, global = true, short = 'v', action = ArgAction::SetTrue)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Total mass M = m1 + m2
    #[command(alias = "m")]
    TotalMass {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Mass ratio q = m2/m1  (0 < q ≤ 1, requires m1 ≥ m2)
    #[command(alias = "q")]
    MassRatio {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Symmetric mass ratio η = m1·m2/(m1+m2)²  (0 < η ≤ 0.25)
    #[command(alias = "eta")]
    SymMassRatio {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Chirp mass ℳ = (m1·m2)^(3/5) / M^(1/5)
    #[command(alias = "mc")]
    ChirpMass {
        /// Primary mass (heavier component)
        m1: f64,
        /// Secondary mass
        m2: f64,
    },

    /// Effective aligned spin χ_eff = (m1·a1·cos θ1 + m2·a2·cos θ2) / M
    #[command(alias = "chieff")]
    ChiEff {
        /// Primary mass
        m1: f64,
        /// Secondary mass
        m2: f64,
        /// Primary spin magnitude a1 ∈ [0, 1]
        a1: f64,
        /// Secondary spin magnitude a2 ∈ [0, 1]
        a2: f64,
        /// Primary tilt angle θ1 (radians, 0 = aligned)
        tilt1: f64,
        /// Secondary tilt angle θ2 (radians, 0 = aligned)
        tilt2: f64,
    },

    /// Precession spin parameter χ_p  (requires m1 ≥ m2)
    #[command(alias = "chip")]
    ChiP {
        /// Primary mass (heavier component, m1 ≥ m2)
        m1: f64,
        /// Secondary mass
        m2: f64,
        /// Primary spin magnitude a1 ∈ [0, 1]
        a1: f64,
        /// Secondary spin magnitude a2 ∈ [0, 1]
        a2: f64,
        /// Primary tilt angle θ1 (radians)
        tilt1: f64,
        /// Secondary tilt angle θ2 (radians)
        tilt2: f64,
    },
}

fn to_kg(val: f64, si: bool) -> f64 {
    if si { val } else { val * MSUN }
}

fn from_kg(val: f64, si: bool) -> f64 {
    if si { val } else { val / MSUN }
}

fn mass(val: f64, si: bool) -> Mass {
    Mass::new::<kilogram>(to_kg(val, si))
}

fn main() {
    let cli = Cli::parse();
    let si = cli.si;
    let verbose = cli.verbose;

    let (result, label, unit) = match cli.command {
        Command::TotalMass { m1, m2 } => {
            let r = total_mass(mass(m1, si), mass(m2, si));
            let unit = if si { "kg" } else { "Msun" };
            (from_kg(r.get::<kilogram>(), si), "total_mass", unit)
        }
        Command::MassRatio { m1, m2 } => {
            let r = mass_ratio(mass(m1, si), mass(m2, si));
            (r, "mass_ratio", "")
        }
        Command::SymMassRatio { m1, m2 } => {
            let r = symmetric_mass_ratio(mass(m1, si), mass(m2, si));
            (r, "eta", "")
        }
        Command::ChirpMass { m1, m2 } => {
            let r = chirp_mass(mass(m1, si), mass(m2, si));
            let unit = if si { "kg" } else { "Msun" };
            (from_kg(r.get::<kilogram>(), si), "chirp_mass", unit)
        }
        Command::ChiEff { m1, m2, a1, a2, tilt1, tilt2 } => {
            let r = chi_eff(mass(m1, si), mass(m2, si), a1, a2, tilt1, tilt2);
            (r, "chi_eff", "")
        }
        Command::ChiP { m1, m2, a1, a2, tilt1, tilt2 } => {
            let r = chi_p(mass(m1, si), mass(m2, si), a1, a2, tilt1, tilt2);
            (r, "chi_p", "")
        }
    };

    if verbose {
        if unit.is_empty() {
            println!("{label} = {result}");
        } else {
            println!("{label} = {result} {unit}");
        }
    } else {
        println!("{result}");
    }
}
