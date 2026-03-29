# Language interfaces

Puddin's core library is written in Rust and exposed to other languages through
thin binding layers. All interfaces accept and return values in **SI units**
(kilograms for mass, radians for angles). Unit conversion helpers are provided
for the Python interface.

---

## Python

Install from PyPI:

```bash
pip install puddin
```

Optional extras:

```bash
pip install "puddin[pint]"   # pint unit support
pip install "puddin[jax]"    # JAX / JIT support
```

### numpy arrays (SI)

```python
import numpy as np
import puddin

MSUN = 1.988_416e30  # kg

m1 = np.full(1000, 30.0 * MSUN)
m2 = np.full(1000, 30.0 * MSUN)

mc  = puddin.chirp_mass(m1, m2)
eta = puddin.symmetric_mass_ratio(m1, m2)
```

### astropy quantities

```python
import astropy.units as u
import puddin

m1 = 30 * u.Msun
m2 = 30 * u.Msun

mc = puddin.chirp_mass(m1, m2)   # returns astropy Quantity in kg
```

### pint quantities

```python
import pint
ureg = pint.UnitRegistry()

m1 = 30 * ureg.solar_mass
mc = puddin.chirp_mass(m1, m1)
```

### JAX

Puddin functions are usable inside `jax.jit`-compiled code via
`jax.pure_callback`. Analytical `custom_vjp` rules are registered so that
gradients flow through correctly.

```python
import jax
import jax.numpy as jnp
import puddin

MSUN = 1.988_416e30

@jax.jit
def log_chirp_mass(m1, m2):
    return jnp.log(puddin.chirp_mass(m1, m2))

m = jnp.array(30.0 * MSUN)
val, grad = jax.value_and_grad(log_chirp_mass)(m, m)
```

---

## JavaScript / TypeScript

Install from npm:

```bash
npm install puddin-wasm
```

The package is built with [wasm-pack](https://rustwasm.github.io/wasm-pack/) and
ships TypeScript declaration files. It targets modern ES module bundlers
(Vite, esbuild, webpack 5, Rollup).

### Vectorised API (`Float64Array`)

All core functions accept and return `Float64Array` for batch processing:

```ts
import { chirp_mass, symmetric_mass_ratio, MSUN } from 'puddin-wasm/puddin';

const m1 = new Float64Array([30 * MSUN, 10 * MSUN]);
const m2 = new Float64Array([30 * MSUN,  5 * MSUN]);

const mc  = chirp_mass(m1, m2);
const eta = symmetric_mass_ratio(m1, m2);
```

### Scalar convenience functions

The TypeScript wrapper layer provides `*_scalar()` helpers that accept and
return plain `number`, useful for single-event interactive work:

```ts
import { chirp_mass_scalar, chi_eff_scalar, MSUN } from 'puddin-wasm/puddin';

const mc = chirp_mass_scalar(30 * MSUN, 30 * MSUN);

const xe = chi_eff_scalar(
  30 * MSUN, 30 * MSUN,  // m1, m2
  0.5, 0.3,              // a1, a2
  0.2, 1.1               // tilt1, tilt2 (radians)
);
```

### Available functions

| Function | Description |
|---|---|
| `total_mass` / `total_mass_scalar` | $M = m_1 + m_2$ |
| `mass_ratio` / `mass_ratio_scalar` | $q = m_2/m_1$ |
| `symmetric_mass_ratio` / `symmetric_mass_ratio_scalar` | $\eta = m_1 m_2/M^2$ |
| `chirp_mass` / `chirp_mass_scalar` | $\mathcal{M} = (m_1 m_2)^{3/5}/M^{1/5}$ |
| `chi_eff` / `chi_eff_scalar` | Effective inspiral spin |
| `chi_p` / `chi_p_scalar` | Effective precession spin ($m_1 \geq m_2$ required) |

---

## Rust

Add to `Cargo.toml`:

```toml
[dependencies]
puddin = "0.1"
uom    = { version = "0.36", features = ["f64", "si"] }
```

All public functions use [uom](https://crates.io/crates/uom) quantity types
for compile-time dimensional analysis:

```rust
use puddin::binary::{chirp_mass, chi_eff, MSUN};
use uom::si::f64::Mass;
use uom::si::mass::kilogram;

let m = Mass::new::<kilogram>(30.0 * MSUN);
let mc = chirp_mass(m, m);

// Spin parameters (dimensionless f64)
let xe = chi_eff(m, m, 0.5, 0.3, 0.2, 1.1);
```

Attempting to pass a length where a mass is expected is a **compile error** —
no runtime checks needed.
