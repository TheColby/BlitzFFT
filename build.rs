// build.rs — compile fft.metal → fft.metallib when the `metal` feature is enabled
use std::{env, path::PathBuf, process::Command};

fn main() {
    // Only compile the Metal shader when the feature is active.
    if env::var("CARGO_FEATURE_METAL").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_src = PathBuf::from("shaders/fft.metal");
    let air_out    = out_dir.join("fft.air");
    let lib_out    = out_dir.join("fft.metallib");

    println!("cargo:rerun-if-changed=shaders/fft.metal");

    // metal → .air (intermediate representation)
    let status = Command::new("xcrun")
        .args([
            "-sdk", "macosx",
            "metal",
            "-c", shader_src.to_str().unwrap(),
            "-o", air_out.to_str().unwrap(),
        ])
        .status()
        .expect("xcrun metal not found — install Xcode command-line tools");

    assert!(status.success(), "Metal shader compilation failed");

    // .air → .metallib
    let status = Command::new("xcrun")
        .args([
            "-sdk", "macosx",
            "metallib",
            air_out.to_str().unwrap(),
            "-o", lib_out.to_str().unwrap(),
        ])
        .status()
        .expect("xcrun metallib not found");

    assert!(status.success(), "metallib link step failed");

    // Expose path to Rust code
    println!("cargo:rustc-env=METAL_LIBRARY_PATH={}", lib_out.display());
}
