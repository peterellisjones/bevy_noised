#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use bevy_math::Vec2;
use bevy_noised::{ridged_fbm_2d_seeded, ridged_fbm_2d_seeded_derivative};

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

fn accuracy_stats(params: Params, positions: &[Vec2], h: f32) -> ErrorStats {
    let mut max_abs: f32 = 0.0;
    let mut sum_abs = 0.0;

    for &p in positions {
        let (_, grad) = ridged_fbm_2d_seeded_derivative(
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

fn time_derivative(params: Params, positions: &[Vec2]) -> f64 {
    let start = Instant::now();
    let mut sink = 0.0f32;

    for _ in 0..PERF_OUTER_ITERS {
        for &p in positions {
            let (value, grad) = ridged_fbm_2d_seeded_derivative(
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

    let elapsed = time_derivative(params, &perf_positions);
    let calls = (PERF_OUTER_ITERS * PERF_POSITIONS) as f64;
    let ns_per_call = elapsed * 1e9 / calls;
    let err = accuracy_stats(params, &accuracy_positions, h);

    println!("Ridged Derivative Report");
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
        "  max_abs={:.3e}, mean_abs={:.3e}",
        err.max_abs, err.mean_abs
    );
    println!();
    println!(
        "Performance (release, {} calls):\n  {:.2} ns/call",
        PERF_OUTER_ITERS * PERF_POSITIONS,
        ns_per_call
    );
}
