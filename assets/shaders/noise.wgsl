// Attribution: simplex implementation adapted from MIT-licensed work by
// Ian McEwan, Stefan Gustavson, Munrocket, and Johan Helsing.

fn fbm_simplex_2d_seeded(pos: vec2<f32>, frequency: f32, octaves: i32, lacunarity: f32, gain: f32, seed: f32) -> f32 {
    var sum = 0.0;
    var amplitude = 1.0;
    var freq = frequency;

    for (var i = 0; i < octaves; i += 1) {
        sum += simplex_noise_2d_seeded(pos * freq, seed) * amplitude;
        amplitude *= gain;
        freq *= lacunarity;
    }

    return sum;
}

fn fbm_simplex_2d_seeded_with_derivative(
    pos: vec2<f32>,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32
) -> vec3<f32> {
    var sum = 0.0;
    var gradient = vec2<f32>(0.0, 0.0);
    var amplitude = 1.0;
    var freq = frequency;

    for (var i = 0; i < octaves; i += 1) {
        let noise_and_grad = simplex_noise_2d_seeded_with_derivative(pos * freq, seed);
        let noise = noise_and_grad.x;
        let grad = noise_and_grad.yz;

        sum += noise * amplitude;
        gradient += grad * amplitude * freq;

        amplitude *= gain;
        freq *= lacunarity;
    }

    return vec3(sum, gradient.x, gradient.y);
}

fn ridged_fbm_2d_seeded(pos: vec2<f32>, frequency: f32, octaves: i32, lacunarity: f32, gain: f32, seed: f32) -> f32 {
    var sum = 0.0;
    var weight = 1.0;
    var amplitude = 1.0;
    var freq = frequency;

    for (var i = 0; i < octaves; i += 1) {
        var signal = abs(simplex_noise_2d_seeded(pos * freq, seed));
        signal = 1.0 - signal;
        signal = signal * signal * weight;

        weight = clamp(signal / amplitude, 0.0, 1.0);

        signal *= amplitude;
        amplitude *= gain;
        sum += signal;
        freq *= lacunarity;
    }

    return sum;
}

fn ridged_fbm_2d_seeded_with_derivative_fast(
    pos: vec2<f32>,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32
) -> vec3<f32> {
    var sum = 0.0;
    var gradient = vec2<f32>(0.0, 0.0);
    var weight = 1.0;
    var amplitude = 1.0;
    var freq = frequency;

    for (var i = 0; i < octaves; i += 1) {
        let noise_and_grad = simplex_noise_2d_seeded_with_derivative(pos * freq, seed);
        let noise_val = noise_and_grad.x;
        let noise_grad = noise_and_grad.yz;

        let scaled_grad = noise_grad * freq;

        let abs_noise = abs(noise_val);
        let ZERO_EPSILON = 1e-4;
        var abs_grad: vec2<f32>;
        if (noise_val > ZERO_EPSILON) {
            abs_grad = scaled_grad;
        } else if (noise_val < -ZERO_EPSILON) {
            abs_grad = -scaled_grad;
        } else {
            abs_grad = vec2<f32>(0.0, 0.0);
        }

        let inverted = 1.0 - abs_noise;
        let inverted_grad = -abs_grad;

        var signal = inverted * inverted * weight;
        var signal_grad = 2.0 * inverted * inverted_grad * weight;

        let new_weight = clamp(signal / amplitude, 0.0, 1.0);

        signal *= amplitude;
        signal_grad *= amplitude;

        sum += signal;
        gradient += signal_grad;

        weight = new_weight;
        amplitude *= gain;
        freq *= lacunarity;
    }

    return vec3(sum, gradient.x, gradient.y);
}

fn ridged_fbm_2d_seeded_with_derivative_exact(
    pos: vec2<f32>,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32
) -> vec3<f32> {
    var sum = 0.0;
    var gradient = vec2<f32>(0.0, 0.0);
    var weight = 1.0;
    var dweight = vec2<f32>(0.0, 0.0);
    var amplitude = 1.0;
    var freq = frequency;

    let ZERO_EPSILON = 1e-4;

    for (var i = 0; i < octaves; i += 1) {
        let noise_and_grad = simplex_noise_2d_seeded_with_derivative(pos * freq, seed);
        let noise_val = noise_and_grad.x;
        let scaled_grad = noise_and_grad.yz * freq;

        let abs_noise = abs(noise_val);
        var abs_grad: vec2<f32>;
        if (noise_val > ZERO_EPSILON) {
            abs_grad = scaled_grad;
        } else if (noise_val < -ZERO_EPSILON) {
            abs_grad = -scaled_grad;
        } else {
            abs_grad = vec2<f32>(0.0, 0.0);
        }

        let inverted = 1.0 - abs_noise;
        let inverted_grad = -abs_grad;

        let base = inverted * inverted;
        let base_grad = 2.0 * inverted * inverted_grad;

        let signal_unscaled = base * weight;
        let signal_unscaled_grad = base_grad * weight + base * dweight;

        sum += signal_unscaled * amplitude;
        gradient += signal_unscaled_grad * amplitude;

        let unclamped_weight = signal_unscaled / amplitude;
        if (unclamped_weight <= 0.0 || unclamped_weight >= 1.0) {
            weight = clamp(unclamped_weight, 0.0, 1.0);
            dweight = vec2<f32>(0.0, 0.0);
        } else {
            weight = unclamped_weight;
            dweight = signal_unscaled_grad / amplitude;
        }

        amplitude *= gain;
        freq *= lacunarity;
    }

    return vec3(sum, gradient.x, gradient.y);
}

fn ridged_fbm_2d_seeded_with_derivative(
    pos: vec2<f32>,
    frequency: f32,
    octaves: i32,
    lacunarity: f32,
    gain: f32,
    seed: f32
) -> vec3<f32> {
    return ridged_fbm_2d_seeded_with_derivative_fast(pos, frequency, octaves, lacunarity, gain, seed);
}

fn simplex_noise_2d_seeded(v: vec2<f32>, seed: f32) -> f32 {
    let C = vec4(
        0.211324865405187,
        0.366025403784439,
        -0.577350269189626,
        0.024390243902439
    );

    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);

    var i1 = select(vec2(0., 1.), vec2(1., 0.), x0.x > x0.y);
    var x12 = x0.xyxy + C.xxzz - vec4(i1, 0., 0.);

    i = i % vec2(289.);

    var p = permute_3_(permute_3_(i.y + vec3(0., i1.y, 1.)) + i.x + vec3(0., i1.x, 1.));
    p = permute_3_(p + vec3(seed));
    var m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), vec3(0.));
    m *= m;
    m *= m;

    let x = 2. * fract(p * C.www) - 1.;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;

    m = m * (1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h));

    let g = vec3(a0.x * x0.x + h.x * x0.y, a0.yz * x12.xz + h.yz * x12.yw);
    return 130. * dot(m, g);
}

fn simplex_noise_2d_seeded_with_derivative(v: vec2<f32>, seed: f32) -> vec3<f32> {
    let C = vec4(
        0.211324865405187,
        0.366025403784439,
        -0.577350269189626,
        0.024390243902439
    );

    var i = floor(v + dot(v, C.yy));
    let x0 = v - i + dot(i, C.xx);

    var i1 = select(vec2(0., 1.), vec2(1., 0.), x0.x > x0.y);
    var x12 = x0.xyxy + C.xxzz - vec4(i1, 0., 0.);

    i = i % vec2(289.);

    var p = permute_3_(permute_3_(i.y + vec3(0., i1.y, 1.)) + i.x + vec3(0., i1.x, 1.));
    p = permute_3_(p + vec3(seed));

    let t0 = 0.5 - dot(x0, x0);
    let t1 = 0.5 - dot(x12.xy, x12.xy);
    let t2 = 0.5 - dot(x12.zw, x12.zw);

    let x = 2. * fract(p * C.www) - 1.;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;

    let grad_norm = 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);

    let g0 = vec2(a0.x, h.x);
    let g1 = vec2(a0.y, h.y);
    let g2 = vec2(a0.z, h.z);

    var n0: f32; var t0_2: f32; var t0_4: f32;
    if (t0 < 0.0) {
        n0 = 0.0; t0_2 = 0.0; t0_4 = 0.0;
    } else {
        t0_2 = t0 * t0;
        t0_4 = t0_2 * t0_2;
        n0 = t0_4 * grad_norm.x * dot(g0, x0);
        t0_4 = t0_4 * grad_norm.x;
    }

    var n1: f32; var t1_2: f32; var t1_4: f32;
    if (t1 < 0.0) {
        n1 = 0.0; t1_2 = 0.0; t1_4 = 0.0;
    } else {
        t1_2 = t1 * t1;
        t1_4 = t1_2 * t1_2;
        n1 = t1_4 * grad_norm.y * dot(g1, x12.xy);
        t1_4 = t1_4 * grad_norm.y;
    }

    var n2: f32; var t2_2: f32; var t2_4: f32;
    if (t2 < 0.0) {
        n2 = 0.0; t2_2 = 0.0; t2_4 = 0.0;
    } else {
        t2_2 = t2 * t2;
        t2_4 = t2_2 * t2_2;
        n2 = t2_4 * grad_norm.z * dot(g2, x12.zw);
        t2_4 = t2_4 * grad_norm.z;
    }

    let noise = 130.0 * (n0 + n1 + n2);

    var temp0: f32; var temp1: f32; var temp2: f32;
    if (t0 < 0.0) {
        temp0 = 0.0;
    } else {
        temp0 = t0_2 * t0 * dot(g0, x0) * grad_norm.x;
    }
    if (t1 < 0.0) {
        temp1 = 0.0;
    } else {
        temp1 = t1_2 * t1 * dot(g1, x12.xy) * grad_norm.y;
    }
    if (t2 < 0.0) {
        temp2 = 0.0;
    } else {
        temp2 = t2_2 * t2 * dot(g2, x12.zw) * grad_norm.z;
    }

    var dnoise_dx = temp0 * x0.x + temp1 * x12.x + temp2 * x12.z;
    var dnoise_dy = temp0 * x0.y + temp1 * x12.y + temp2 * x12.w;
    dnoise_dx *= -8.0;
    dnoise_dy *= -8.0;

    dnoise_dx += t0_4 * g0.x + t1_4 * g1.x + t2_4 * g2.x;
    dnoise_dy += t0_4 * g0.y + t1_4 * g1.y + t2_4 * g2.y;

    let grad = vec2(dnoise_dx, dnoise_dy) * 130.0;

    return vec3(noise, grad.x, grad.y);
}

fn permute_3_(x: vec3<f32>) -> vec3<f32> {
    return (((x * 34.) + 1.) * x) % vec3(289.);
}
