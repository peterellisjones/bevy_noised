#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use bevy_math::Vec2;
use bevy_noised::{
    ridged_fbm_2d_seeded, ridged_fbm_2d_seeded_derivative_exact,
    ridged_fbm_2d_seeded_derivative_fast,
};

const PERF_POSITIONS: usize = 4096;
const PERF_OUTER_ITERS: usize = 300;
const ACCURACY_POSITIONS: usize = 256;

#[derive(Clone, Copy)]
struct Params {
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32,
}

#[derive(Clone, Copy)]
struct ErrorStats {
    max_abs: f32,
    mean_abs: f32,
}

fn sample_positions(count: usize, extent: f32) -> Vec<Vec2> {
    let mut state = 0xA53C_9E21_u32;
    let mut out = Vec::with_capacity(count);

    for _ in 0..count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let rx = (state as f32) / (u32::MAX as f32);
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let ry = (state as f32) / (u32::MAX as f32);

        let x = -extent + rx * (2.0 * extent);
        let y = -extent + ry * (2.0 * extent);
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

fn accuracy_stats_fast(params: Params, positions: &[Vec2], h: f32) -> ErrorStats {
    let mut max_abs: f32 = 0.0;
    let mut sum_abs = 0.0;

    for &p in positions {
        let (_, grad) = ridged_fbm_2d_seeded_derivative_fast(
            p,
            params.frequency,
            params.octaves,
            params.lacunarity,
            params.gain,
            params.seed,
        );
        let numeric = central_difference_2d(
            |x| {
                ridged_fbm_2d_seeded(
                    x,
                    params.frequency,
                    params.octaves,
                    params.lacunarity,
                    params.gain,
                    params.seed,
                )
            },
            p,
            h,
        );
        let err = (grad - numeric).abs().max_element();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }

    ErrorStats {
        max_abs,
        mean_abs: sum_abs / (positions.len() as f32),
    }
}

fn accuracy_stats_exact(params: Params, positions: &[Vec2], h: f32) -> ErrorStats {
    let mut max_abs: f32 = 0.0;
    let mut sum_abs = 0.0;

    for &p in positions {
        let (_, grad) = ridged_fbm_2d_seeded_derivative_exact(
            p,
            params.frequency,
            params.octaves,
            params.lacunarity,
            params.gain,
            params.seed,
        );
        let numeric = central_difference_2d(
            |x| {
                ridged_fbm_2d_seeded(
                    x,
                    params.frequency,
                    params.octaves,
                    params.lacunarity,
                    params.gain,
                    params.seed,
                )
            },
            p,
            h,
        );
        let err = (grad - numeric).abs().max_element();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }

    ErrorStats {
        max_abs,
        mean_abs: sum_abs / (positions.len() as f32),
    }
}

fn time_fast(params: Params, positions: &[Vec2]) -> f64 {
    let start = Instant::now();
    let mut sink = 0.0f32;

    for _ in 0..PERF_OUTER_ITERS {
        for &p in positions {
            let (value, grad) = ridged_fbm_2d_seeded_derivative_fast(
                p,
                params.frequency,
                params.octaves,
                params.lacunarity,
                params.gain,
                params.seed,
            );
            sink += value + grad.x + grad.y;
        }
    }

    std::hint::black_box(sink);
    start.elapsed().as_secs_f64()
}

fn time_exact(params: Params, positions: &[Vec2]) -> f64 {
    let start = Instant::now();
    let mut sink = 0.0f32;

    for _ in 0..PERF_OUTER_ITERS {
        for &p in positions {
            let (value, grad) = ridged_fbm_2d_seeded_derivative_exact(
                p,
                params.frequency,
                params.octaves,
                params.lacunarity,
                params.gain,
                params.seed,
            );
            sink += value + grad.x + grad.y;
        }
    }

    std::hint::black_box(sink);
    start.elapsed().as_secs_f64()
}

fn main() {
    let params = Params {
        frequency: 0.001,
        octaves: 6,
        lacunarity: 2.0,
        gain: 0.5,
        seed: 42.0,
    };

    let perf_positions = sample_positions(PERF_POSITIONS, 500.0);
    let accuracy_positions = sample_positions(ACCURACY_POSITIONS, 500.0);
    let h = 1e-2;

    let fast_time = time_fast(params, &perf_positions);
    let exact_time = time_exact(params, &perf_positions);

    let calls = (PERF_OUTER_ITERS * PERF_POSITIONS) as f64;
    let fast_ns = fast_time * 1e9 / calls;
    let exact_ns = exact_time * 1e9 / calls;

    let fast_err = accuracy_stats_fast(params, &accuracy_positions, h);
    let exact_err = accuracy_stats_exact(params, &accuracy_positions, h);

    println!("Ridged Derivative Accuracy/Performance Report");
    println!(
        "params: freq={:.4}, octaves={}, lacunarity={:.2}, gain={:.2}, seed={:.1}",
        params.frequency, params.octaves, params.lacunarity, params.gain, params.seed
    );
    println!();
    println!(
        "Accuracy vs central difference (h = {:.1e}, n = {}):",
        h, ACCURACY_POSITIONS
    );
    println!(
        "  fast : max_abs={:.3e}, mean_abs={:.3e}",
        fast_err.max_abs, fast_err.mean_abs
    );
    println!(
        "  exact: max_abs={:.3e}, mean_abs={:.3e}",
        exact_err.max_abs, exact_err.mean_abs
    );
    println!(
        "  improvement (max): {:.2}x, (mean): {:.2}x",
        fast_err.max_abs / exact_err.max_abs,
        fast_err.mean_abs / exact_err.mean_abs
    );
    println!();
    println!(
        "Performance (release, {} calls):\n  fast : {:.2} ns/call\n  exact: {:.2} ns/call\n  slowdown: {:.2}x",
        PERF_OUTER_ITERS * PERF_POSITIONS,
        fast_ns,
        exact_ns,
        exact_ns / fast_ns
    );
}
