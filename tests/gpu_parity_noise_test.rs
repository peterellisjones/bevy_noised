#![allow(clippy::cast_precision_loss)]

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;
use bevy_gpu_test::ComputeTest;
use bevy_noised::{
    fbm_simplex_2d_seeded, fbm_simplex_2d_seeded_derivative, ridged_fbm_2d_seeded,
    ridged_fbm_2d_seeded_derivative, ridged_fbm_2d_seeded_derivative_exact,
    simplex_noise_2d_seeded, simplex_noise_2d_seeded_derivative,
};

const TEST_POSITIONS: usize = 256;

const SIMPLEX_VALUE_TOLERANCE: f32 = 1e-4;
const FBM_VALUE_TOLERANCE: f32 = 5e-4;
const SIMPLEX_GRADIENT_TOLERANCE: f32 = 1e-6;
const FBM_GRADIENT_TOLERANCE: f32 = 1e-4;

const MODE_SIMPLEX: u32 = 0;
const MODE_SIMPLEX_DERIVATIVE: u32 = 1;
const MODE_FBM: u32 = 2;
const MODE_FBM_DERIVATIVE: u32 = 3;
const MODE_RIDGED_FBM: u32 = 4;
const MODE_RIDGED_FBM_DERIVATIVE: u32 = 5;
const MODE_RIDGED_FBM_DERIVATIVE_EXACT: u32 = 6;

const ALL_MODES: &[(u32, &str)] = &[
    (MODE_SIMPLEX, "simplex_noise"),
    (MODE_SIMPLEX_DERIVATIVE, "simplex_derivative"),
    (MODE_FBM, "fbm_noise"),
    (MODE_FBM_DERIVATIVE, "fbm_derivative"),
    (MODE_RIDGED_FBM, "ridged_fbm_noise"),
    (MODE_RIDGED_FBM_DERIVATIVE, "ridged_fbm_derivative"),
    (
        MODE_RIDGED_FBM_DERIVATIVE_EXACT,
        "ridged_fbm_derivative_exact",
    ),
];

#[derive(Clone, Copy, Debug, ShaderType)]
struct NoiseTestInput {
    x: f32,
    y: f32,
    frequency: f32,
    seed: f32,
    octaves: i32,
    lacunarity: f32,
    persistence: f32,
    _pad: f32,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct NoiseTestOutput {
    value: f32,
    gradient_x: f32,
    gradient_y: f32,
    _pad: f32,
}

#[derive(Debug, Clone)]
struct NoiseParityResult {
    mode_name: String,
    passed: bool,
    test_count: usize,
    max_value_diff: f32,
    max_gradient_diff: f32,
    failure_count: usize,
}

fn generate_test_inputs() -> Vec<NoiseTestInput> {
    let mut inputs = Vec::with_capacity(TEST_POSITIONS);

    let grid_side = (TEST_POSITIONS as f32).sqrt().ceil() as usize;
    let range = 5000.0;
    let step = (2.0 * range) / (grid_side as f32 - 1.0).max(1.0);

    let frequencies = [0.0001, 0.0002, 0.001, 0.002];
    let seeds = [0.0, 42.0, 123.0, 255.0];
    let octaves = [4, 6, 8];

    let mut param_idx = 0;

    for i in 0..grid_side {
        for j in 0..grid_side {
            if inputs.len() >= TEST_POSITIONS {
                break;
            }

            let x = -range + (i as f32) * step;
            let y = -range + (j as f32) * step;

            inputs.push(NoiseTestInput {
                x,
                y,
                frequency: frequencies[param_idx % frequencies.len()],
                seed: seeds[(param_idx / frequencies.len()) % seeds.len()],
                octaves: octaves[(param_idx / (frequencies.len() * seeds.len())) % octaves.len()],
                lacunarity: 2.0,
                persistence: 0.5,
                _pad: 0.0,
            });

            param_idx += 1;
        }
    }

    inputs.truncate(TEST_POSITIONS);
    inputs
}

fn compute_cpu_results(mode: u32, inputs: &[NoiseTestInput]) -> Vec<NoiseTestOutput> {
    inputs
        .iter()
        .map(|input| {
            let pos = Vec2::new(input.x, input.y);
            match mode {
                MODE_SIMPLEX => NoiseTestOutput {
                    value: simplex_noise_2d_seeded(pos * input.frequency, input.seed),
                    ..default()
                },
                MODE_SIMPLEX_DERIVATIVE => {
                    let (value, grad) =
                        simplex_noise_2d_seeded_derivative(pos * input.frequency, input.seed);
                    NoiseTestOutput {
                        value,
                        gradient_x: grad.x * input.frequency,
                        gradient_y: grad.y * input.frequency,
                        _pad: 0.0,
                    }
                }
                MODE_FBM => NoiseTestOutput {
                    value: fbm_simplex_2d_seeded(
                        pos,
                        input.frequency,
                        input.octaves,
                        input.lacunarity,
                        input.persistence,
                        input.seed,
                    ),
                    ..default()
                },
                MODE_FBM_DERIVATIVE => {
                    let (value, grad) = fbm_simplex_2d_seeded_derivative(
                        pos,
                        input.frequency,
                        input.octaves,
                        input.lacunarity,
                        input.persistence,
                        input.seed,
                    );
                    NoiseTestOutput {
                        value,
                        gradient_x: grad.x,
                        gradient_y: grad.y,
                        _pad: 0.0,
                    }
                }
                MODE_RIDGED_FBM => NoiseTestOutput {
                    value: ridged_fbm_2d_seeded(
                        pos,
                        input.frequency,
                        input.octaves,
                        input.lacunarity,
                        input.persistence,
                        input.seed,
                    ),
                    ..default()
                },
                MODE_RIDGED_FBM_DERIVATIVE => {
                    let (value, grad) = ridged_fbm_2d_seeded_derivative(
                        pos,
                        input.frequency,
                        input.octaves,
                        input.lacunarity,
                        input.persistence,
                        input.seed,
                    );
                    NoiseTestOutput {
                        value,
                        gradient_x: grad.x,
                        gradient_y: grad.y,
                        _pad: 0.0,
                    }
                }
                MODE_RIDGED_FBM_DERIVATIVE_EXACT => {
                    let (value, grad) = ridged_fbm_2d_seeded_derivative_exact(
                        pos,
                        input.frequency,
                        input.octaves,
                        input.lacunarity,
                        input.persistence,
                        input.seed,
                    );
                    NoiseTestOutput {
                        value,
                        gradient_x: grad.x,
                        gradient_y: grad.y,
                        _pad: 0.0,
                    }
                }
                _ => NoiseTestOutput::default(),
            }
        })
        .collect()
}

fn run_mode(mode: u32, inputs: Vec<NoiseTestInput>) -> Vec<NoiseTestOutput> {
    ComputeTest::new("shaders/parity_test_noise.wgsl", inputs)
        .with_uniform(mode)
        .with_workgroup_size(64)
        .run()
}

fn compare_results(
    mode: u32,
    mode_name: &str,
    cpu_results: &[NoiseTestOutput],
    gpu_results: &[NoiseTestOutput],
) -> NoiseParityResult {
    let has_gradient = mode == MODE_SIMPLEX_DERIVATIVE
        || mode == MODE_FBM_DERIVATIVE
        || mode == MODE_RIDGED_FBM_DERIVATIVE
        || mode == MODE_RIDGED_FBM_DERIVATIVE_EXACT;
    let (value_tolerance, gradient_tolerance) =
        if mode == MODE_SIMPLEX || mode == MODE_SIMPLEX_DERIVATIVE {
            (SIMPLEX_VALUE_TOLERANCE, SIMPLEX_GRADIENT_TOLERANCE)
        } else {
            (FBM_VALUE_TOLERANCE, FBM_GRADIENT_TOLERANCE)
        };

    let mut max_value_diff: f32 = 0.0;
    let mut max_gradient_diff: f32 = 0.0;
    let mut failure_count = 0;

    for (cpu, gpu) in cpu_results.iter().zip(gpu_results.iter()) {
        let value_diff = (cpu.value - gpu.value).abs();
        max_value_diff = max_value_diff.max(value_diff);
        let mut failed = value_diff > value_tolerance;

        if has_gradient {
            let grad_diff = (cpu.gradient_x - gpu.gradient_x)
                .abs()
                .max((cpu.gradient_y - gpu.gradient_y).abs());
            max_gradient_diff = max_gradient_diff.max(grad_diff);
            if grad_diff > gradient_tolerance {
                failed = true;
            }
        }

        if failed {
            failure_count += 1;
        }
    }

    NoiseParityResult {
        mode_name: mode_name.to_string(),
        passed: failure_count == 0,
        test_count: cpu_results.len(),
        max_value_diff,
        max_gradient_diff,
        failure_count,
    }
}

#[test]
fn noise_parity() {
    let inputs = generate_test_inputs();
    let mut results = Vec::new();

    for &(mode, mode_name) in ALL_MODES {
        let cpu_results = compute_cpu_results(mode, &inputs);
        let gpu_results = run_mode(mode, inputs.clone());
        results.push(compare_results(mode, mode_name, &cpu_results, &gpu_results));
    }

    let mut all_passed = true;
    for result in &results {
        println!(
            "{}: {} ({} tests, max value diff: {:.2e}, max grad diff: {:.2e}, failures: {})",
            result.mode_name,
            if result.passed { "PASS" } else { "FAIL" },
            result.test_count,
            result.max_value_diff,
            result.max_gradient_diff,
            result.failure_count
        );
        if !result.passed {
            all_passed = false;
        }
    }

    assert!(all_passed, "Some bevy_noised parity tests failed");
}
