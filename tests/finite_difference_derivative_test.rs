#![allow(clippy::cast_precision_loss)]

use bevy_math::Vec2;
use bevy_noised::{
    fbm_simplex_2d_seeded, fbm_simplex_2d_seeded_derivative, ridged_fbm_2d_seeded,
    ridged_fbm_2d_seeded_derivative, simplex_noise_2d_seeded, simplex_noise_2d_seeded_derivative,
};

const SAMPLE_COUNT: usize = 64;

fn sample_positions() -> Vec<Vec2> {
    let mut state = 0x9E37_79B9_u32;
    let mut out = Vec::with_capacity(SAMPLE_COUNT);

    for _ in 0..SAMPLE_COUNT {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let rx = (state as f32) / (u32::MAX as f32);
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let ry = (state as f32) / (u32::MAX as f32);

        let x = -100.0 + rx * 200.0;
        let y = -100.0 + ry * 200.0;
        out.push(Vec2::new(x, y));
    }

    out
}

fn central_difference_2d<F>(f: F, p: Vec2, h: f32) -> Vec2
where
    F: Fn(Vec2) -> f32,
{
    let ex = Vec2::new(h, 0.0);
    let ey = Vec2::new(0.0, h);

    let dx = (f(p + ex) - f(p - ex)) / (2.0 * h);
    let dy = (f(p + ey) - f(p - ey)) / (2.0 * h);
    Vec2::new(dx, dy)
}

#[test]
fn simplex_derivative_matches_finite_difference() {
    let seed = 42.0;
    let h = 1e-3;
    let mut max_err: f32 = 0.0;

    for p in sample_positions() {
        let (_, grad) = simplex_noise_2d_seeded_derivative(p, seed);
        let numeric = central_difference_2d(|x| simplex_noise_2d_seeded(x, seed), p, h);
        let err = (grad - numeric).abs().max_element();
        max_err = max_err.max(err);
    }

    assert!(
        max_err < 4e-3,
        "simplex derivative too far from finite difference (max err: {max_err:.3e})"
    );
}

#[test]
fn fbm_derivative_matches_finite_difference() {
    let frequency = 0.001;
    let octaves = 6;
    let lacunarity = 2.0;
    let gain = 0.5;
    let seed = 42.0;
    let h = 1e-2;
    let mut max_err: f32 = 0.0;

    for p in sample_positions() {
        let (_, grad) =
            fbm_simplex_2d_seeded_derivative(p, frequency, octaves, lacunarity, gain, seed);
        let numeric = central_difference_2d(
            |x| fbm_simplex_2d_seeded(x, frequency, octaves, lacunarity, gain, seed),
            p,
            h,
        );
        let err = (grad - numeric).abs().max_element();
        max_err = max_err.max(err);
    }

    assert!(
        max_err < 2e-4,
        "fbm derivative too far from finite difference (max err: {max_err:.3e})"
    );
}

#[test]
fn ridged_fbm_derivative_matches_finite_difference() {
    let frequency = 0.001;
    let octaves = 6;
    let lacunarity = 2.0;
    let gain = 0.5;
    let seed = 42.0;
    let h = 1e-2;
    let mut max_err: f32 = 0.0;

    for p in sample_positions() {
        let (_, grad) =
            ridged_fbm_2d_seeded_derivative(p, frequency, octaves, lacunarity, gain, seed);
        let numeric = central_difference_2d(
            |x| ridged_fbm_2d_seeded(x, frequency, octaves, lacunarity, gain, seed),
            p,
            h,
        );
        let err = (grad - numeric).abs().max_element();
        max_err = max_err.max(err);
    }

    assert!(
        max_err < 1e-3,
        "ridged derivative too far from finite difference (max err: {max_err:.3e})"
    );
}
