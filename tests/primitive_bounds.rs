//! Soundness witness for the simplex primitive's bounding constants.
//!
//! [`SIMPLEX_2D_MAX_ABS_VALUE`], [`SIMPLEX_2D_MAX_GRADIENT`] and
//! [`SIMPLEX_2D_MAX_HESSIAN`] are *guaranteed upper bounds* consumed by
//! downstream interval primitives (e.g. terrain_gen's `height_bounds`). This
//! test re-measures the actual extrema over a dense multi-seed sweep and asserts
//! the published constants still bound them — a drift guard, and a witness that
//! the constants are sound.
//!
//! It also asserts the constants are not *absurdly* loose (within a small factor
//! of the densely-measured maximum), so the bound stays tight enough to be
//! useful for pruning.

#![allow(clippy::cast_precision_loss)]

use bevy_math::Vec2;
use bevy_noised::{
    simplex_noise_2d_seeded_derivative, SIMPLEX_2D_MAX_ABS_VALUE, SIMPLEX_2D_MAX_GRADIENT,
    SIMPLEX_2D_MAX_HESSIAN,
};

/// Grid side per seed. This is a *drift guard*: it asserts the published
/// constants still bound a dense sample. The constants themselves were chosen
/// against a finer GRID=4096/12-seed sweep whose grid-justified true-supremum
/// bound (`measured + L · ½·cell_diagonal`) is ≈ 1.0095 for the value and ≈ 7.27
/// for the gradient — comfortably under the 1.02 / 7.4 constants.
const GRID: usize = 1024;
/// World-space window swept per seed (covers many simplex cells).
const WINDOW: f32 = 8.0;
/// Seeds spanning the permutation domain (`% 289`).
const SEEDS: [f32; 10] = [0.0, 1.0, 3.0, 7.0, 13.0, 42.0, 100.0, 137.0, 200.0, 288.0];

fn measure() -> (f32, f32) {
    let mut max_abs_val = 0.0f32;
    let mut max_grad = 0.0f32;
    for &seed in &SEEDS {
        for ix in 0..GRID {
            for iy in 0..GRID {
                let x = (ix as f32 / GRID as f32) * WINDOW;
                let y = (iy as f32 / GRID as f32) * WINDOW;
                let (v, g) = simplex_noise_2d_seeded_derivative(Vec2::new(x, y), seed);
                max_abs_val = max_abs_val.max(v.abs());
                max_grad = max_grad.max(g.length());
            }
        }
    }
    (max_abs_val, max_grad)
}

/// Max Frobenius norm of the Hessian, finite-differencing the analytic gradient.
/// Coarser grid (the extra 4 gradient evals per point are expensive), still dense
/// enough as a drift guard; the constant carries margin for the inter-grid excess.
fn measure_hessian() -> f32 {
    const GRID_H: usize = 512;
    const H: f32 = 0.02;
    let mut max_hess = 0.0f32;
    for &seed in &SEEDS {
        for ix in 0..GRID_H {
            for iy in 0..GRID_H {
                let p = Vec2::new(
                    (ix as f32 / GRID_H as f32) * WINDOW,
                    (iy as f32 / GRID_H as f32) * WINDOW,
                );
                let gx = (simplex_noise_2d_seeded_derivative(p + Vec2::new(H, 0.0), seed).1
                    - simplex_noise_2d_seeded_derivative(p - Vec2::new(H, 0.0), seed).1)
                    / (2.0 * H);
                let gy = (simplex_noise_2d_seeded_derivative(p + Vec2::new(0.0, H), seed).1
                    - simplex_noise_2d_seeded_derivative(p - Vec2::new(0.0, H), seed).1)
                    / (2.0 * H);
                let (a, b, c) = (gx.x, 0.5 * (gx.y + gy.x), gy.y);
                max_hess = max_hess.max((a * a + 2.0 * b * b + c * c).sqrt());
            }
        }
    }
    max_hess
}

#[test]
fn constants_bound_the_simplex_primitive() {
    let (measured_val, measured_grad) = measure();

    // Sound: the published constant must bound every measured sample.
    assert!(
        measured_val <= SIMPLEX_2D_MAX_ABS_VALUE,
        "max|value| {measured_val:.6} exceeds SIMPLEX_2D_MAX_ABS_VALUE {SIMPLEX_2D_MAX_ABS_VALUE}"
    );
    assert!(
        measured_grad <= SIMPLEX_2D_MAX_GRADIENT,
        "max|grad| {measured_grad:.6} exceeds SIMPLEX_2D_MAX_GRADIENT {SIMPLEX_2D_MAX_GRADIENT}"
    );

    // Tight: the constant should be a snug upper bound (small margin), not a
    // loose over-estimate — looseness directly costs downstream pruning payoff.
    assert!(
        SIMPLEX_2D_MAX_ABS_VALUE <= measured_val * 1.05,
        "SIMPLEX_2D_MAX_ABS_VALUE {SIMPLEX_2D_MAX_ABS_VALUE} is too loose vs measured {measured_val:.6}"
    );
    assert!(
        SIMPLEX_2D_MAX_GRADIENT <= measured_grad * 1.05,
        "SIMPLEX_2D_MAX_GRADIENT {SIMPLEX_2D_MAX_GRADIENT} is too loose vs measured {measured_grad:.6}"
    );

    // Hessian (Frobenius). Larger tightness factor: it is finite-differenced, so
    // the margin also absorbs FD error near the kernel features.
    let measured_hess = measure_hessian();
    assert!(
        measured_hess <= SIMPLEX_2D_MAX_HESSIAN,
        "max Hessian {measured_hess:.4} exceeds SIMPLEX_2D_MAX_HESSIAN {SIMPLEX_2D_MAX_HESSIAN}"
    );
    assert!(
        SIMPLEX_2D_MAX_HESSIAN <= measured_hess * 1.5,
        "SIMPLEX_2D_MAX_HESSIAN {SIMPLEX_2D_MAX_HESSIAN} is too loose vs measured {measured_hess:.4}"
    );
}
