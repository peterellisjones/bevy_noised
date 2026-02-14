# bevy_noised

CPU + WGSL noise primitives for terrain workflows:

- CPU + GPU function pairs with matching behavior
- parity-tested with `bevy_gpu_test`
- analytical derivatives for fast normal/gradient workflows

## What it has

Right now this crate is intentionally small:

- simplex noise (seeded)
- fbm simplex (seeded)
- ridged fbm simplex (seeded)
- analytical derivatives for the above
- 2D only (for now)

## Example

```rust
use bevy_math::Vec2;
use bevy_noised::fbm_simplex_2d_seeded_derivative;

let pos = Vec2::new(128.0, 256.0);
let (height, grad) = fbm_simplex_2d_seeded_derivative(
    pos,
    0.001, // frequency
    6,     // octaves
    2.0,   // lacunarity
    0.5,   // gain
    42.0,  // seed
);

let normal = Vec2::new(-grad.x, -grad.y).normalize_or_zero();
println!("height={height}, grad={grad:?}, normal_xy={normal:?}");
```

For WGSL, include the shader source from this crate and call the matching functions:

```wgsl
// Example call shape
let p = vec2<f32>(x, y);
let n = fbm_simplex_2d_seeded_with_derivative(p, frequency, octaves, lacunarity, gain, seed);
// n.x = value, n.yz = gradient
```

## Quick comparison

| Crate | CPU+GPU | Parity confidence | Dimensions | Variants | Derivatives |
| --- | --- | --- | --- | --- | --- |
| `bevy_noised` | Yes | High (`bevy_gpu_test` parity tests) | 2D | simplex, fbm, ridged fbm | Yes |
| `noisy_bevy` | Yes | Medium (states parity; no dedicated GPU-vs-CPU parity harness found) | 2D, 3D | simplex, fbm, worley | Not exposed |
| `noiz` | CPU only | N/A | Vec2/Vec3/Vec4-oriented | lots: perlin/simplex/worley/layering/warp | Yes |
| `bevy_compute_noise` | GPU only | N/A | 2D, 3D textures | perlin, worley, fbm wrapper | No |
| `bevy_shader_utils` | GPU only | N/A | mostly 2D/3D shader utils | perlin, simplex, voronoise | No |

Notes:

- "Parity" here means close numeric match under tolerances, not bit-perfect equality.

## Test tolerances

Current CPU/GPU parity tolerances (`tests/gpu_parity_noise_test.rs`):

- simplex value: `1e-5` (0.001%)
- simplex derivative: `2e-7` (0.00002%)
- fbm/ridged value: `2e-5` (0.002%)
- fbm/ridged derivative: `2e-5` (0.002%)

Example parity error ranges from a recent run (`cargo test noise_parity -- --nocapture`):

- value diffs: about `1.95e-6` to `5.84e-6`
- gradient diffs: about `3.77e-8` to `1.16e-5`

Finite-difference derivative tolerances (`tests/finite_difference_derivative_test.rs`):

- simplex derivative vs numerical gradient: `4e-3`
- fbm derivative vs numerical gradient: `2e-4`
- ridged fbm derivative vs numerical gradient: `1e-3`
