#import "shaders/noise.wgsl" as noise

const MODE_SIMPLEX: u32 = 0u;
const MODE_SIMPLEX_DERIVATIVE: u32 = 1u;
const MODE_FBM: u32 = 2u;
const MODE_FBM_DERIVATIVE: u32 = 3u;
const MODE_RIDGED_FBM: u32 = 4u;
const MODE_RIDGED_FBM_DERIVATIVE: u32 = 5u;

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

struct NoiseTestOutput {
    value: f32,
    gradient_x: f32,
    gradient_y: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> test_mode: u32;
@group(0) @binding(1) var<storage, read> inputs: array<NoiseTestInput>;
@group(0) @binding(2) var<storage, read_write> outputs: array<NoiseTestOutput>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    let input_count = arrayLength(&inputs);
    if (index >= input_count) {
        return;
    }

    let input = inputs[index];
    let pos = vec2<f32>(input.x, input.y);

    var result: NoiseTestOutput;

    switch test_mode {
        case MODE_SIMPLEX: {
            result.value = noise::simplex_noise_2d_seeded(pos * input.frequency, input.seed);
            result.gradient_x = 0.0;
            result.gradient_y = 0.0;
        }
        case MODE_SIMPLEX_DERIVATIVE: {
            let n = noise::simplex_noise_2d_seeded_with_derivative(pos * input.frequency, input.seed);
            result.value = n.x;
            result.gradient_x = n.y * input.frequency;
            result.gradient_y = n.z * input.frequency;
        }
        case MODE_FBM: {
            result.value = noise::fbm_simplex_2d_seeded(
                pos,
                input.frequency,
                input.octaves,
                input.lacunarity,
                input.persistence,
                input.seed
            );
            result.gradient_x = 0.0;
            result.gradient_y = 0.0;
        }
        case MODE_FBM_DERIVATIVE: {
            let n = noise::fbm_simplex_2d_seeded_with_derivative(
                pos,
                input.frequency,
                input.octaves,
                input.lacunarity,
                input.persistence,
                input.seed
            );
            result.value = n.x;
            result.gradient_x = n.y;
            result.gradient_y = n.z;
        }
        case MODE_RIDGED_FBM: {
            result.value = noise::ridged_fbm_2d_seeded(
                pos,
                input.frequency,
                input.octaves,
                input.lacunarity,
                input.persistence,
                input.seed
            );
            result.gradient_x = 0.0;
            result.gradient_y = 0.0;
        }
        case MODE_RIDGED_FBM_DERIVATIVE: {
            let n = noise::ridged_fbm_2d_seeded_with_derivative(
                pos,
                input.frequency,
                input.octaves,
                input.lacunarity,
                input.persistence,
                input.seed
            );
            result.value = n.x;
            result.gradient_x = n.y;
            result.gradient_y = n.z;
        }
        default: {
            result.value = 0.0;
            result.gradient_x = 0.0;
            result.gradient_y = 0.0;
        }
    }

    outputs[index] = result;
}
