//! Bevy-first 2D noise primitives for gameplay + rendering parity.
//!
//! `bevy_noised` exposes matching CPU and WGSL noise functions so gameplay systems,
//! terrain generation, and shaders can sample the same fields with near-identical
//! numeric output.
//!
//! The primary target workflow is shader-displaced terrain where CPU picking/raycast
//! queries need to return the same heights as the rendered mesh.
//!
//! Derivatives are key in this workflow because they let you compute normals from
//! the same noise field driving vertex displacement, which keeps terrain lighting
//! stable and consistent.
//!
//! ## Why this crate in a Bevy project?
//!
//! - Deterministic seeded noise on CPU and GPU
//! - Analytical derivatives for slope/normal/flow style workflows
//! - Small API surface focused on common terrain and mask pipelines
//!
//! ## Picking a function
//!
//! - `simplex_noise_2d_seeded`: base coherent noise
//! - `fbm_simplex_2d_seeded`: layered terrain-style noise
//! - `ridged_fbm_2d_seeded`: mountain/ridge-heavy noise
//! - `*_derivative` variants: return `(value, gradient)`
//!
//! ## Parameter guidance
//!
//! - `frequency`: world scale (`0.0005` to `0.01` are common terrain ranges)
//! - `octaves`: layer count (`4` to `8` typical)
//! - `lacunarity`: frequency multiplier per octave (usually around `2.0`)
//! - `gain`: amplitude multiplier per octave (`0.4` to `0.6` common)
//! - `seed`: deterministic variation source
//!
//! ## Example (CPU)
//!
//! ```
//! use bevy_math::{Vec2, Vec3};
//! use bevy_noised::fbm_simplex_2d_seeded_derivative;
//!
//! let world_pos = Vec2::new(128.0, 256.0);
//! let (height, grad) = fbm_simplex_2d_seeded_derivative(world_pos, 0.001, 6, 2.0, 0.5, 42.0);
//!
//! let normal = Vec3::new(-grad.x, 1.0, -grad.y).normalize_or_zero();
//! assert!(height.is_finite());
//! assert!(normal.is_finite());
//! ```
//!
//! ## WGSL source
//!
//! Use [`WGSL_NOISE_SOURCE`] to embed this crate's shader functions into your
//! Bevy shader source. WGSL derivative variants return `vec3(value, dfdx, dfdy)`.
//!
//! Attribution: simplex implementation adapted from MIT-licensed work by
//! Ian McEwan, Stefan Gustavson, Munrocket, and Johan Helsing.

use bevy_math::{vec2, vec3, vec4, Vec2, Vec3, Vec4};

pub const WGSL_NOISE_SOURCE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/assets/shaders/noise.wgsl"
));

/// Fractal Brownian motion (fBm) built from seeded 2D simplex noise.
///
/// Returns layered coherent noise suitable for terrain height, biome masks,
/// and other smoothly-varying gameplay fields.
///
/// WGSL equivalent: `fbm_simplex_2d_seeded`.
#[must_use]
pub fn fbm_simplex_2d_seeded(
    pos: Vec2,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 1.0;
    let mut freq = frequency;

    for _ in 0..octaves {
        sum += simplex_noise_2d_seeded(pos * freq, seed) * amplitude;
        amplitude *= gain;
        freq *= lacunarity;
    }

    sum
}

/// Fractal Brownian motion plus analytical gradient.
///
/// Returns `(value, gradient)`, where `gradient` is the derivative with respect
/// to the input `pos` coordinates.
///
/// Typically used to derive normals for vertex-displaced terrain and to measure
/// steepness for gameplay or material blending.
///
/// WGSL equivalent: `fbm_simplex_2d_seeded_with_derivative` (`vec3(value, dx, dy)`).
#[must_use]
pub fn fbm_simplex_2d_seeded_derivative(
    pos: Vec2,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32,
) -> (f32, Vec2) {
    let mut sum = 0.0;
    let mut gradient = Vec2::ZERO;
    let mut amplitude = 1.0;
    let mut freq = frequency;

    for _ in 0..octaves {
        let (noise, grad) = simplex_noise_2d_seeded_derivative(pos * freq, seed);
        sum += noise * amplitude;
        gradient += grad * amplitude * freq;
        amplitude *= gain;
        freq *= lacunarity;
    }

    (sum, gradient)
}

/// Seeded 2D simplex noise value.
///
/// This is the base primitive used by the fBm and ridged fBm helpers.
///
/// WGSL equivalent: `simplex_noise_2d_seeded`.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn simplex_noise_2d_seeded(v: Vec2, seed: f32) -> f32 {
    const C: Vec4 = vec4(0.211_324_87, 0.366_025_42, -0.577_350_26, 0.024_390_243);

    let mut i: Vec2 = (v + Vec2::dot(v, vec2(C.y, C.y))).floor();
    let x0 = v - i + Vec2::dot(i, vec2(C.x, C.x));

    let i1: Vec2 = if x0.x > x0.y {
        vec2(1.0, 0.0)
    } else {
        vec2(0.0, 1.0)
    };
    let x12: Vec4 =
        vec4(x0.x, x0.y, x0.x, x0.y) + vec4(C.x, C.x, C.z, C.z) - vec4(i1.x, i1.y, 0.0, 0.0);

    i %= Vec2::splat(289.0);

    let mut p = permute_3(permute_3(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    p = permute_3(p + Vec3::splat(seed));

    let mut m = Vec3::max(
        0.5 - vec3(
            Vec2::dot(x0, x0),
            Vec2::dot(vec2(x12.x, x12.y), vec2(x12.x, x12.y)),
            Vec2::dot(vec2(x12.z, x12.w), vec2(x12.z, x12.w)),
        ),
        Vec3::splat(0.0),
    );
    m *= m;
    m *= m;

    let x: Vec3 = 2.0 * (p * Vec3::splat(C.w)).fract() - 1.0;
    let h = x.abs() - 0.5;
    let ox = (x + 0.5).floor();
    let a0 = x - ox;

    m *= 1.792_842_9 - 0.853_734_73 * (a0 * a0 + h * h);
    let g = vec3(
        a0.x * x0.x + h.x * x0.y,
        a0.y * x12.x + h.y * x12.y,
        a0.z * x12.z + h.z * x12.w,
    );

    130.0 * Vec3::dot(m, g)
}

fn permute_3(x: Vec3) -> Vec3 {
    (((x * 34.0) + 1.0) * x) % Vec3::splat(289.0)
}

/// Seeded 2D simplex noise value plus analytical gradient.
///
/// Returns `(value, gradient)`, where `gradient` is the derivative with respect
/// to the input `v` coordinates.
///
/// Useful when simplex directly drives displacement and you need matching normals.
///
/// WGSL equivalent: `simplex_noise_2d_seeded_with_derivative` (`vec3(value, dx, dy)`).
#[must_use]
#[allow(clippy::many_single_char_names, clippy::similar_names)]
pub fn simplex_noise_2d_seeded_derivative(v: Vec2, seed: f32) -> (f32, Vec2) {
    const C: Vec4 = vec4(0.211_324_87, 0.366_025_42, -0.577_350_26, 0.024_390_243);

    let mut i: Vec2 = (v + Vec2::dot(v, vec2(C.y, C.y))).floor();
    let x0 = v - i + Vec2::dot(i, vec2(C.x, C.x));

    let i1: Vec2 = if x0.x > x0.y {
        vec2(1.0, 0.0)
    } else {
        vec2(0.0, 1.0)
    };
    let x12: Vec4 =
        vec4(x0.x, x0.y, x0.x, x0.y) + vec4(C.x, C.x, C.z, C.z) - vec4(i1.x, i1.y, 0.0, 0.0);

    i %= Vec2::splat(289.0);

    let mut p = permute_3(permute_3(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    p = permute_3(p + Vec3::splat(seed));

    let t0 = 0.5 - Vec2::dot(x0, x0);
    let t1 = 0.5 - Vec2::dot(vec2(x12.x, x12.y), vec2(x12.x, x12.y));
    let t2 = 0.5 - Vec2::dot(vec2(x12.z, x12.w), vec2(x12.z, x12.w));

    let x: Vec3 = 2.0 * (p * Vec3::splat(C.w)).fract() - 1.0;
    let h = x.abs() - 0.5;
    let ox = (x + 0.5).floor();
    let a0 = x - ox;

    let grad_norm = 1.792_842_9 - 0.853_734_73 * (a0 * a0 + h * h);

    let g0 = vec2(a0.x, h.x);
    let g1 = vec2(a0.y, h.y);
    let g2 = vec2(a0.z, h.z);

    let (n0, t0_2, t0_4) = if t0 < 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        let t0_2 = t0 * t0;
        let t0_4 = t0_2 * t0_2;
        let n0 = t0_4 * grad_norm.x * Vec2::dot(g0, x0);
        (n0, t0_2, t0_4 * grad_norm.x)
    };

    let (n1, t1_2, t1_4) = if t1 < 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        let t1_2 = t1 * t1;
        let t1_4 = t1_2 * t1_2;
        let n1 = t1_4 * grad_norm.y * Vec2::dot(g1, vec2(x12.x, x12.y));
        (n1, t1_2, t1_4 * grad_norm.y)
    };

    let (n2, t2_2, t2_4) = if t2 < 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        let t2_2 = t2 * t2;
        let t2_4 = t2_2 * t2_2;
        let n2 = t2_4 * grad_norm.z * Vec2::dot(g2, vec2(x12.z, x12.w));
        (n2, t2_2, t2_4 * grad_norm.z)
    };

    let noise = 130.0 * (n0 + n1 + n2);

    let temp0 = if t0 < 0.0 {
        0.0
    } else {
        t0_2 * t0 * Vec2::dot(g0, x0) * grad_norm.x
    };
    let temp1 = if t1 < 0.0 {
        0.0
    } else {
        t1_2 * t1 * Vec2::dot(g1, vec2(x12.x, x12.y)) * grad_norm.y
    };
    let temp2 = if t2 < 0.0 {
        0.0
    } else {
        t2_2 * t2 * Vec2::dot(g2, vec2(x12.z, x12.w)) * grad_norm.z
    };

    let mut dnoise_dx = temp0 * x0.x + temp1 * x12.x + temp2 * x12.z;
    let mut dnoise_dy = temp0 * x0.y + temp1 * x12.y + temp2 * x12.w;
    dnoise_dx *= -8.0;
    dnoise_dy *= -8.0;

    dnoise_dx += t0_4 * g0.x + t1_4 * g1.x + t2_4 * g2.x;
    dnoise_dy += t0_4 * g0.y + t1_4 * g1.y + t2_4 * g2.y;

    let grad = vec2(dnoise_dx, dnoise_dy) * 130.0;

    (noise, grad)
}

/// Ridged fractal noise built from seeded 2D simplex noise.
///
/// Produces sharper mountain-like structures than standard fBm.
///
/// WGSL equivalent: `ridged_fbm_2d_seeded`.
#[must_use]
pub fn ridged_fbm_2d_seeded(
    pos: Vec2,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32,
) -> f32 {
    let mut sum = 0.0;
    let mut weight = 1.0;
    let mut amplitude = 1.0;
    let mut freq = frequency;

    for _ in 0..octaves {
        let mut signal = simplex_noise_2d_seeded(pos * freq, seed).abs();
        signal = 1.0 - signal;
        signal = signal * signal * weight;
        weight = (signal / amplitude).clamp(0.0, 1.0);
        signal *= amplitude;
        amplitude *= gain;
        sum += signal;
        freq *= lacunarity;
    }

    sum
}

/// Ridged fractal noise plus analytical gradient.
///
/// Returns `(value, gradient)`, where `gradient` is the derivative with respect
/// to the input `pos` coordinates.
///
/// Especially useful for ridge-heavy displaced terrain where normal quality has a
/// strong visual impact on lighting.
///
/// WGSL equivalent: `ridged_fbm_2d_seeded_with_derivative` (`vec3(value, dx, dy)`).
#[must_use]
pub fn ridged_fbm_2d_seeded_derivative(
    pos: Vec2,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32,
) -> (f32, Vec2) {
    const ZERO_EPSILON: f32 = 1e-4;

    let mut sum = 0.0;
    let mut gradient = Vec2::ZERO;
    let mut weight = 1.0;
    let mut dweight = Vec2::ZERO;
    let mut amplitude = 1.0;
    let mut freq = frequency;

    for _ in 0..octaves {
        let (noise_val, noise_grad) = simplex_noise_2d_seeded_derivative(pos * freq, seed);
        let scaled_grad = noise_grad * freq;

        let abs_noise = noise_val.abs();
        let abs_grad = if noise_val > ZERO_EPSILON {
            scaled_grad
        } else if noise_val < -ZERO_EPSILON {
            -scaled_grad
        } else {
            Vec2::ZERO
        };

        let inverted = 1.0 - abs_noise;
        let inverted_grad = -abs_grad;

        let base = inverted * inverted;
        let base_grad = 2.0 * inverted * inverted_grad;

        let signal_unscaled = base * weight;
        let signal_unscaled_grad = base_grad * weight + base * dweight;

        sum += signal_unscaled * amplitude;
        gradient += signal_unscaled_grad * amplitude;

        let unclamped_weight = signal_unscaled / amplitude;
        if unclamped_weight <= 0.0 || unclamped_weight >= 1.0 {
            weight = unclamped_weight.clamp(0.0, 1.0);
            dweight = Vec2::ZERO;
        } else {
            weight = unclamped_weight;
            dweight = signal_unscaled_grad / amplitude;
        }

        amplitude *= gain;
        freq *= lacunarity;
    }

    (sum, gradient)
}
