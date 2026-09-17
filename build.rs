//! Compiles every `.metal` source under `kernels/` into a single `.metallib`,
//! which `src/metal/mod.rs` embeds with `include_bytes!` and hands to
//! `newLibraryWithData:`.
//!
//! The flags are copied from mlx's own kernel build
//! (`vendor/mlx/mlx/backend/metal/kernels/CMakeLists.txt`):
//!
//!   xcrun -sdk macosx metal -x metal -Wall -Wextra -fno-fast-math \
//!       -Wno-c++17-extensions -Wno-c++20-extensions -Wmetal-addr-spaces \
//!       -c <src> -I<root> -o <out>.air
//!   xcrun -sdk macosx metallib *.air -o kernels.metallib
//!
//! `-fno-fast-math` matters: with fast math the compiler is free to contract,
//! reassociate and flush denormals, and the ported kernels would stop matching
//! mlx numerically.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kernels = manifest.join("kernels");
    // Verbatim mlx headers live under kernels/mlx/ with their original
    // directory layout, so `#include "mlx/backend/metal/kernels/utils.h"`
    // resolves unchanged against this include root.
    let mlx_root = kernels.join("mlx");

    let mut sources = Vec::new();
    collect_metal(&kernels, &mut sources);
    sources.sort();
    assert!(!sources.is_empty(), "no .metal sources under {:?}", kernels);

    println!("cargo:rerun-if-changed=kernels");
    for s in &sources {
        println!("cargo:rerun-if-changed={}", s.display());
    }
    // Headers are pulled in by #include, so watch the whole tree.
    for h in walk(&kernels) {
        if matches!(h.extension().and_then(|e| e.to_str()), Some("h")) {
            println!("cargo:rerun-if-changed={}", h.display());
        }
    }

    let mut airs = Vec::new();
    for src in &sources {
        let stem = src.file_stem().unwrap().to_str().unwrap();
        // Two kernels could share a stem across subdirectories; disambiguate
        // with the path relative to kernels/.
        let rel = src.strip_prefix(&kernels).unwrap();
        let tag = rel.with_extension("").to_string_lossy().replace('/', "_");
        let air = out_dir.join(format!("{tag}.air"));
        let status = Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-x", "metal"])
            .args(["-Wall", "-Wextra", "-fno-fast-math"])
            // mlx compiles its JIT kernels with MTLCompileOptions'
            // MathModeSafe (CompileOptions::math_mode defaults to Safe in
            // mlx/backend/common/metal_kernel.h). The command-line compiler
            // defaults to a laxer mode, and the difference shows up as a
            // last-bit disagreement in exp.
            .args(["-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise"])
            .arg("-std=metal4.0")
            .args(["-Wno-c++17-extensions", "-Wno-c++20-extensions", "-Wmetal-addr-spaces"])
            .arg("-c")
            .arg(src)
            .arg(format!("-I{}", mlx_root.display()))
            .arg(format!("-I{}", kernels.display()))
            .arg("-o")
            .arg(&air)
            .status()
            .unwrap_or_else(|e| panic!("failed to run xcrun metal for {stem}: {e}"));
        assert!(status.success(), "metal compile failed for {}", src.display());
        airs.push(air);
    }

    let metallib = out_dir.join("kernels.metallib");
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib"])
        .args(&airs)
        .arg("-o")
        .arg(&metallib)
        .status()
        .expect("failed to run xcrun metallib");
    assert!(status.success(), "metallib link failed");

    println!("cargo:rustc-env=ORNITH_METALLIB={}", metallib.display());
}

fn collect_metal(dir: &Path, out: &mut Vec<PathBuf>) {
    for p in walk(dir) {
        if matches!(p.extension().and_then(|e| e.to_str()), Some("metal")) {
            out.push(p);
        }
    }
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}
