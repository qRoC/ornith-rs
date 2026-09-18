//! Reading the golden references written by `tools/ref.py`.
//!
//! A reference directory is `manifest.json` plus one raw little-endian file
//! per tensor. Tests load the same inputs mlx was given, run the ported code
//! and compare against mlx's output.

use std::path::{Path, PathBuf};

use crate::array::{Array, Dtype, bf16_to_f32};
use crate::json;
use crate::metal::Context;

pub struct RefCase {
    pub dir: PathBuf,
    manifest: json::Value,
}

pub struct RefTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl RefCase {
    pub fn open(name: &str) -> Option<RefCase> {
        let dir = refs_root().join(name);
        let raw = std::fs::read(dir.join("manifest.json")).ok()?;
        let manifest = json::parse(&raw).ok()?;
        Some(RefCase { dir, manifest })
    }

    /// Open, or explain how to produce the reference and skip.
    pub fn open_or_skip(name: &str) -> Option<RefCase> {
        match RefCase::open(name) {
            Some(c) => Some(c),
            None => {
                eprintln!(
                    "skipping: no reference {name:?}. Generate it with\n  \
                     cd tools && PYTHONPATH=../../vendor/mlx-lm python3 dump_{name}.py"
                );
                None
            }
        }
    }

    pub fn meta(&self) -> Option<&json::Value> {
        self.manifest.get("meta")
    }

    pub fn meta_usize(&self, dotted: &str) -> usize {
        self.meta()
            .and_then(|m| m.path(dotted))
            .and_then(|v| v.as_usize())
            .unwrap_or_else(|| panic!("meta.{dotted} missing"))
    }

    /// Whether the reference carries this tensor.
    ///
    /// `dump_model.py` can be asked for every layer instead of the eight the
    /// frozen case holds, and a test that checks whatever is there needs no
    /// second list to keep in step with the dumper's.
    pub fn has(&self, name: &str) -> bool {
        self.manifest.path(&format!("tensors.{name}")).is_some()
    }

    pub fn get(&self, name: &str) -> RefTensor {
        let t = self
            .manifest
            .path(&format!("tensors.{name}"))
            .unwrap_or_else(|| panic!("reference {:?} has no tensor {name:?}", self.dir));
        let dtype = match t.get("dtype").and_then(|v| v.as_str()).unwrap() {
            "bfloat16" => Dtype::BF16,
            "float16" => Dtype::F16,
            "float32" => Dtype::F32,
            "uint32" => Dtype::U32,
            "int32" => Dtype::I32,
            "uint8" => Dtype::U8,
            "bool" => Dtype::Bool,
            d => panic!("unhandled reference dtype {d}"),
        };
        let shape: Vec<usize> = t
            .get("shape")
            .and_then(|v| v.as_arr())
            .unwrap()
            .iter()
            .map(|v| v.as_usize().unwrap())
            .collect();
        let file = t.get("file").and_then(|v| v.as_str()).unwrap();
        let bytes = std::fs::read(self.dir.join(file))
            .unwrap_or_else(|e| panic!("{}/{file}: {e}", self.dir.display()));
        let expect: usize = shape.iter().product::<usize>() * dtype.size();
        assert_eq!(bytes.len(), expect, "{name}: file is {} bytes, want {expect}", bytes.len());
        RefTensor { dtype, shape, bytes }
    }

    /// Upload a reference tensor to the device as-is.
    pub fn array(&self, ctx: &Context, name: &str) -> Array {
        let t = self.get(name);
        Array::from_slice(ctx, &t.bytes, &t.shape, t.dtype)
    }

    pub fn f32(&self, name: &str) -> Vec<f32> {
        self.get(name).to_f32()
    }

    /// A side file written by `Case.put_json`: text and token lists, which do
    /// not fit the raw-tensor format.
    pub fn json(&self, name: &str) -> json::Value {
        let t = self
            .manifest
            .path(&format!("tensors.{name}"))
            .unwrap_or_else(|| panic!("reference {:?} has no file {name:?}", self.dir));
        assert_eq!(t.get("dtype").and_then(|v| v.as_str()), Some("json"), "{name} is a tensor");
        let file = t.get("file").and_then(|v| v.as_str()).unwrap();
        let raw = std::fs::read(self.dir.join(file))
            .unwrap_or_else(|e| panic!("{}/{file}: {e}", self.dir.display()));
        json::parse(&raw).unwrap_or_else(|e| panic!("{}/{file}: {e}", self.dir.display()))
    }

    /// The same, as a list of strings.
    pub fn strings(&self, name: &str) -> Vec<String> {
        self.json(name)
            .as_arr()
            .unwrap_or_else(|| panic!("{name} is not an array"))
            .iter()
            .map(|v| v.as_str().expect("not a string").to_string())
            .collect()
    }
}

impl RefTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Dtype::BF16 => self
                .bytes
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            Dtype::U32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            Dtype::I32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect(),
            d => panic!("to_f32 for {}", d.name()),
        }
    }

    pub fn to_u32(&self) -> Vec<u32> {
        match self.dtype {
            Dtype::U32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            Dtype::I32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
                .collect(),
            d => panic!("to_u32 for {}", d.name()),
        }
    }
}

pub fn refs_root() -> PathBuf {
    if let Ok(p) = std::env::var("ORNITH_REFS") {
        return PathBuf::from(p);
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("refs")
}

pub fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("ORNITH_MODEL").unwrap_or_else(|_| {
        "/Users/qroc/.mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx".into()
    }))
}

// --- comparison --------------------------------------------------------------

pub struct Diff {
    pub max_abs: f32,
    pub max_rel: f32,
    pub mean_abs: f32,
    pub argmax: usize,
    pub got: f32,
    pub want: f32,
    pub n: usize,
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} max_abs={:.3e} (at {}: got {} want {}) max_rel={:.3e} mean_abs={:.3e}",
            self.n, self.max_abs, self.argmax, self.got, self.want, self.max_rel, self.mean_abs
        )
    }
}

pub fn compare(got: &[f32], want: &[f32]) -> Diff {
    assert_eq!(got.len(), want.len(), "length mismatch: {} vs {}", got.len(), want.len());
    let mut d = Diff {
        max_abs: 0.0,
        max_rel: 0.0,
        mean_abs: 0.0,
        argmax: 0,
        got: 0.0,
        want: 0.0,
        n: got.len(),
    };
    let mut sum = 0.0f64;
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let a = (g - w).abs();
        sum += a as f64;
        if a > d.max_abs {
            d.max_abs = a;
            d.argmax = i;
            d.got = *g;
            d.want = *w;
        }
        d.max_rel = d.max_rel.max(a / scale);
    }
    d.mean_abs = (sum / got.len() as f64) as f32;
    d
}
