use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/kissfft/kiss_fft.c");
    println!("cargo:rerun-if-changed=vendor/kissfft/kiss_fftr.c");
    println!("cargo:rerun-if-changed=vendor/kissfft/kiss_fft.h");
    println!("cargo:rerun-if-changed=vendor/kissfft/kiss_fftr.h");
    println!("cargo:rerun-if-changed=vendor/pocketfft/pocketfft_hdronly.h");
    println!("cargo:rerun-if-changed=src/native/pocketfft_bridge.cc");
    println!("cargo:rerun-if-changed=shaders/blitz_fft.cu");

    compile_kissfft(&out_dir);
    compile_pocketfft_bridge(&out_dir);
    link_fftw();

    if env::var("CARGO_FEATURE_METAL").is_ok() {
        compile_metal_shader(&out_dir);
    }

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        compile_cuda_kernel(&out_dir);
    }
}

fn compile_kissfft(out_dir: &Path) {
    let include = "vendor/kissfft";
    let kiss_fft_o = out_dir.join("kiss_fft.o");
    let kiss_fftr_o = out_dir.join("kiss_fftr.o");
    let lib_path = out_dir.join("libkissfft.a");

    run_checked(
        Command::new("cc").args([
            "-O3",
            "-DNDEBUG",
            "-I",
            include,
            "-c",
            "vendor/kissfft/kiss_fft.c",
            "-o",
            kiss_fft_o.to_str().unwrap(),
        ]),
        "compile vendor/kissfft/kiss_fft.c",
    );

    run_checked(
        Command::new("cc").args([
            "-O3",
            "-DNDEBUG",
            "-I",
            include,
            "-c",
            "vendor/kissfft/kiss_fftr.c",
            "-o",
            kiss_fftr_o.to_str().unwrap(),
        ]),
        "compile vendor/kissfft/kiss_fftr.c",
    );

    run_checked(
        Command::new("ar").args([
            "rcs",
            lib_path.to_str().unwrap(),
            kiss_fft_o.to_str().unwrap(),
            kiss_fftr_o.to_str().unwrap(),
        ]),
        "archive libkissfft.a",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kissfft");
}

fn compile_pocketfft_bridge(out_dir: &Path) {
    let bridge_o = out_dir.join("pocketfft_bridge.o");
    let lib_path = out_dir.join("libpocketfft_bridge.a");

    run_checked(
        Command::new("c++").args([
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-I",
            "vendor/pocketfft",
            "-c",
            "src/native/pocketfft_bridge.cc",
            "-o",
            bridge_o.to_str().unwrap(),
        ]),
        "compile pocketfft bridge",
    );

    run_checked(
        Command::new("ar").args([
            "rcs",
            lib_path.to_str().unwrap(),
            bridge_o.to_str().unwrap(),
        ]),
        "archive libpocketfft_bridge.a",
    );

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=pocketfft_bridge");
    println!("cargo:rustc-link-lib=dylib=c++");
}

fn link_fftw() {
    let output = Command::new("pkg-config")
        .args(["--libs", "--cflags", "fftw3f"])
        .output();

    if let Ok(output) = output {
        if output.status.success() {
            let flags = String::from_utf8_lossy(&output.stdout);
            for token in flags.split_whitespace() {
                if let Some(path) = token.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={path}");
                } else if let Some(lib) = token.strip_prefix("-l") {
                    println!("cargo:rustc-link-lib={lib}");
                }
            }
            return;
        }
    }

    let fallback_prefixes = ["/opt/homebrew", "/usr/local"];
    for prefix in fallback_prefixes {
        let lib_dir = Path::new(prefix).join("lib");
        let dylib = lib_dir.join("libfftw3f.dylib");
        if dylib.exists() {
            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=fftw3f");
            return;
        }
    }

    panic!("Unable to locate FFTW3 single-precision library (fftw3f)");
}

fn compile_metal_shader(out_dir: &Path) {
    let shader_src = PathBuf::from("shaders/fft.metal");
    let air_out = out_dir.join("fft.air");
    let lib_out = out_dir.join("fft.metallib");

    println!("cargo:rerun-if-changed=shaders/fft.metal");

    run_checked(
        Command::new("xcrun").args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            shader_src.to_str().unwrap(),
            "-o",
            air_out.to_str().unwrap(),
        ]),
        "compile Metal shader",
    );

    run_checked(
        Command::new("xcrun").args([
            "-sdk",
            "macosx",
            "metallib",
            air_out.to_str().unwrap(),
            "-o",
            lib_out.to_str().unwrap(),
        ]),
        "link Metal shader library",
    );

    println!("cargo:rustc-env=METAL_LIBRARY_PATH={}", lib_out.display());
}

fn compile_cuda_kernel(out_dir: &Path) {
    let src = PathBuf::from("shaders/blitz_fft.cu");
    let ptx = out_dir.join("blitz_fft.ptx");

    // Try to locate nvcc.
    let nvcc_result = Command::new("nvcc")
        .args([
            "--ptx",
            "-O3",
            "-arch=sm_70",               // Volta+ baseline; fatbin would cover more arches
            src.to_str().unwrap(),
            "-o",
            ptx.to_str().unwrap(),
        ])
        .status();

    match nvcc_result {
        Ok(status) if status.success() => {
            println!("cargo:rustc-cfg=blitz_cuda_kernel");
            println!("cargo:warning=BlitzFFT CUDA kernel compiled to PTX.");
        }
        Ok(status) => {
            println!(
                "cargo:warning=nvcc found but compilation failed (status {status}). \
                 The CUDA backend will be unavailable."
            );
        }
        Err(_) => {
            println!(
                "cargo:warning=nvcc not found. \
                 Install the CUDA Toolkit to enable the native CUDA backend."
            );
        }
    }
}

fn run_checked(cmd: &mut Command, context: &str) {
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("{context}: {err}"));
    assert!(status.success(), "{context} failed with status {status}");
}
