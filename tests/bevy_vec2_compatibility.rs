use bevy::math::Vec2;
use bevy_noised::simplex_noise_2d_seeded;

#[test]
fn accepts_bevy_math_vec2() {
    let position = Vec2::new(128.0, 256.0);

    let value = simplex_noise_2d_seeded(position, 42.0);

    assert!(value.is_finite());
}
