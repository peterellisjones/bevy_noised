# bevy_noised

2D seeded noise for Bevy, with matching CPU and WGSL implementations.

- matching CPU and shader function families
- GPU parity tested with `bevy_gpu_test`
- analytical derivatives for normals (critical for vertex-displaced terrain), flow direction, and steepness checks

## Primary use case: terrain query parity

This crate is mainly for CPU query parity with GPU terrain displacement.

If your terrain vertices are displaced in a shader, gameplay systems can sample the
same seeded noise on CPU and get matching heights for picking, raycasts, placement,
and movement logic.

Derivative outputs are useful here because they let you compute slope-aware
normals from the same noise field used for displacement.

## Other Bevy use cases

You can also use it when gameplay systems and shaders should sample the same field:

- CPU picking/raycast height queries that match shader-displaced terrain
- terrain height on CPU and terrain displacement in WGSL
- biome/mask generation in systems + mask visualization in materials
- slope/gradient queries for movement, spawning, erosion, or blending rules

## Quick start (CPU)

```rust
use bevy_math::{Vec2, Vec3};
use bevy_noised::fbm_simplex_2d_seeded_derivative;

let world_pos = Vec2::new(128.0, 256.0);
let (height, grad) = fbm_simplex_2d_seeded_derivative(
    world_pos,
    0.001, // frequency
    6,     // octaves
    2.0,   // lacunarity
    0.5,   // gain
    42.0,  // seed
);

let normal = Vec3::new(-grad.x, 1.0, -grad.y).normalize_or_zero();
let steepness = grad.length();

println!("height={height:.3}, steepness={steepness:.3}, normal={normal:?}");
```

In terrain shaders, these derivatives are typically converted to normals for
lighting and material blending.

## Quick start (WGSL)

Include this crate's WGSL source and call matching functions in your shader:

```rust
use bevy_noised::WGSL_NOISE_SOURCE;

let shader_src = format!(
    "{WGSL_NOISE_SOURCE}\n\n@fragment fn fragment() -> @location(0) vec4<f32> {{\n    let p = vec2<f32>(0.5, 0.25);\n    let n = fbm_simplex_2d_seeded_with_derivative(p, 0.001, 6, 2.0, 0.5, 42.0);\n    return vec4<f32>(n.x, n.y, n.z, 1.0);\n}}"
);
```

In WGSL derivative variants, return shape is:

- `n.x`: noise value
- `n.y`: `d/dx`
- `n.z`: `d/dy`

## Function overview

- `simplex_noise_2d_seeded`: base coherent seeded noise
- `fbm_simplex_2d_seeded`: layered simplex for terrain-like fields
- `ridged_fbm_2d_seeded`: mountain/ridge-focused fractal noise
- `*_derivative` variants: return value + analytical gradient

Current scope is intentionally focused: seeded 2D primitives with derivative support.

## Ecosystem context

If your project needs CPU/GPU parity and derivative output, this crate is focused on that workflow.

| Crate | CPU+GPU | Parity notes | Dimensions | Variants | Derivatives |
| --- | --- | --- | --- | --- | --- |
| `bevy_noised` | Yes | parity tests with `bevy_gpu_test` | 2D | simplex, fbm, ridged fbm | Yes |
| `noisy_bevy` | Yes | parity stated in docs (no dedicated parity harness found) | 2D, 3D | simplex, fbm, worley | Not exposed |
| `noiz` | CPU only | N/A | Vec2/Vec3/Vec4-oriented | perlin/simplex/worley/layering/warp | Yes |
| `bevy_compute_noise` | GPU only | N/A | 2D, 3D textures | perlin, worley, fbm wrapper | No |
| `bevy_shader_utils` | GPU only | N/A | mostly 2D/3D shader utils | perlin, simplex, voronoise | No |

Parity means close numeric match under tolerance, not bit-perfect equality.

## Parity and derivative tolerances

CPU/GPU parity tolerances (`tests/gpu_parity_noise_test.rs`):

- simplex value: `1e-5` (0.001%)
- simplex derivative: `2e-7` (0.00002%)
- fbm/ridged value: `2e-5` (0.002%)
- fbm/ridged derivative: `2e-5` (0.002%)

Recent parity run (`cargo test noise_parity -- --nocapture`):

- value diffs: about `1.95e-6` to `5.84e-6`
- gradient diffs: about `3.77e-8` to `1.16e-5`

Finite-difference derivative tolerances (`tests/finite_difference_derivative_test.rs`):

- simplex derivative vs numerical gradient: `4e-3`
- fbm derivative vs numerical gradient: `2e-4`
- ridged fbm derivative vs numerical gradient: `1e-3`

## Bevy compatibility

- crate `0.1.x` -> Bevy `0.18`

## Roadmap

- 3D variants
- domain warp helpers
- tiling-friendly variants for wrapped worlds
