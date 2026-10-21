#![allow(clippy::cast_precision_loss)]

//! Release-mode timing of the CPU noise functions.
//!
//! Run with `cargo run --release --example noise_timing`. Each line is the best
//! of several passes over a fixed set of positions, in nanoseconds per call.

use std::hint::black_box;
use std::time::Instant;

use bevy_noised::{
    fbm_simplex_2d_seeded, fbm_simplex_2d_seeded_derivative, ridged_fbm_2d_seeded,
    ridged_fbm_2d_seeded_derivative, simplex_noise_2d_seeded, simplex_noise_2d_seeded_derivative,
};
use glam::Vec2;

const POSITIONS: usize = 4096;
const PASSES: usize = 9;
const ITERS_PER_PASS: usize = 50;

const FREQUENCY: f32 = 0.001;
const OCTAVES: i32 = 6;
const LACUNARITY: f32 = 2.0;
const GAIN: f32 = 0.5;
const SEED: f32 = 42.0;

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

/// A timed function: its name, its positions, and a call that folds its output
/// into one `f32`.
type Row<'a> = (&'a str, &'a [Vec2], &'a dyn Fn(Vec2) -> f32);

/// Best-of-`PASSES` nanoseconds per call of `f` over `positions`.
fn time_ns_per_call(positions: &[Vec2], f: impl Fn(Vec2) -> f32) -> f64 {
    let mut best = f64::MAX;

    for _ in 0..PASSES {
        let start = Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..ITERS_PER_PASS {
            for &p in positions {
                sink += f(black_box(p));
            }
        }
        black_box(sink);
        let calls = (ITERS_PER_PASS * positions.len()) as f64;
        best = best.min(start.elapsed().as_secs_f64() * 1e9 / calls);
    }

    best
}

fn main() {
    // Simplex is sampled directly in noise space; the fBm variants are scaled
    // by `FREQUENCY`, so give them world-sized positions.
    let noise_positions = sample_positions(POSITIONS, 500.0);
    let world_positions = sample_positions(POSITIONS, 500_000.0);

    let rows: [Row; 6] = [
        ("simplex", &noise_positions, &|p| {
            simplex_noise_2d_seeded(p, SEED)
        }),
        ("simplex_derivative", &noise_positions, &|p| {
            let (v, g) = simplex_noise_2d_seeded_derivative(p, SEED);
            v + g.x + g.y
        }),
        ("fbm", &world_positions, &|p| {
            fbm_simplex_2d_seeded(p, FREQUENCY, OCTAVES, LACUNARITY, GAIN, SEED)
        }),
        ("fbm_derivative", &world_positions, &|p| {
            let (v, g) =
                fbm_simplex_2d_seeded_derivative(p, FREQUENCY, OCTAVES, LACUNARITY, GAIN, SEED);
            v + g.x + g.y
        }),
        ("ridged", &world_positions, &|p| {
            ridged_fbm_2d_seeded(p, FREQUENCY, OCTAVES, LACUNARITY, GAIN, SEED)
        }),
        ("ridged_derivative", &world_positions, &|p| {
            let (v, g) =
                ridged_fbm_2d_seeded_derivative(p, FREQUENCY, OCTAVES, LACUNARITY, GAIN, SEED);
            v + g.x + g.y
        }),
    ];

    println!("Noise timing (best of {PASSES} passes, {OCTAVES} octaves for fBm/ridged)");
    for (name, positions, f) in rows {
        println!(
            "  {name:<20} {:>8.2} ns/call",
            time_ns_per_call(positions, f)
        );
    }
}
