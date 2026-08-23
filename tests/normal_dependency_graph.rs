use std::process::Command;

#[test]
fn normal_dependency_graph_excludes_bevy_family_packages() {
    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .args([
            "tree",
            "--edges",
            "normal",
            "--prefix",
            "none",
            "--package",
            "bevy_noised",
        ])
        .current_dir(option_env!("CARGO_MANIFEST_DIR").unwrap_or("."))
        .output()
        .expect("Cargo must be available to inspect the dependency graph");

    assert!(
        output.status.success(),
        "cargo tree failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let graph = String::from_utf8_lossy(&output.stdout);
    let bevy_packages: Vec<_> = graph
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .filter(|package| *package == "bevy" || package.starts_with("bevy_"))
        .collect();

    assert!(
        bevy_packages.is_empty(),
        "normal dependency graph contains Bevy-family packages: {bevy_packages:?}"
    );
}
