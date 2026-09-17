//! safetensors reader backed by `mmap`.
//!
//! The artifact is ~28 GB across 19 files. Copying that into process memory at
//! load time would be pointless on unified memory: instead every file is mapped
//! once, and each Metal buffer is created *over the mapping* with
//! `newBufferWithBytesNoCopy`. Weights are then paged in by the GPU on first
//! touch and the page cache owns them, which is what `mx.load` does via
//! `mlx::core::load_safetensors` too.
//!
//! File format (from the safetensors spec):
//!   [u64 little-endian header length N][N bytes of JSON header][raw tensor data]
//! Header entries are `{"name": {"dtype": "...", "shape": [...],
//! "data_offsets": [begin, end]}}`, offsets relative to the start of the data
//! segment.

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use crate::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F16,
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "BF16" => Dtype::BF16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "U8" => Dtype::U8,
            "U16" => Dtype::U16,
            "U32" => Dtype::U32,
            "U64" => Dtype::U64,
            "I8" => Dtype::I8,
            "I16" => Dtype::I16,
            "I32" => Dtype::I32,
            "I64" => Dtype::I64,
            "BOOL" => Dtype::Bool,
            _ => return None,
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8 | Dtype::Bool => 1,
            Dtype::BF16 | Dtype::F16 | Dtype::U16 | Dtype::I16 => 2,
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::U64 | Dtype::I64 => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Dtype::BF16 => "bfloat16",
            Dtype::F16 => "float16",
            Dtype::F32 => "float32",
            Dtype::F64 => "float64",
            Dtype::U8 => "uint8",
            Dtype::U16 => "uint16",
            Dtype::U32 => "uint32",
            Dtype::U64 => "uint64",
            Dtype::I8 => "int8",
            Dtype::I16 => "int16",
            Dtype::I32 => "int32",
            Dtype::I64 => "int64",
            Dtype::Bool => "bool",
        }
    }
}

/// One tensor's location inside a mapped file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Index into [`SafeTensors::files`].
    pub file: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte offset from the start of the mapping (data segment already added).
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// --- mmap ------------------------------------------------------------------

unsafe extern "C" {
    fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    fn getpagesize() -> i32;
}

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;
const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;
const MADV_WILLNEED: i32 = 3;

/// A read-only mapping of an entire file, page-aligned at both ends so it can
/// back a `newBufferWithBytesNoCopy` Metal buffer directly.
pub struct Mapping {
    ptr: *mut c_void,
    /// Mapped length, rounded up to a page. The tail page past EOF reads zero.
    pub map_len: usize,
    /// Real file length.
    pub file_len: usize,
    pub path: PathBuf,
}

// The mapping is read-only and never mutated after construction.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    pub fn open(path: &Path) -> std::io::Result<Mapping> {
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let page = unsafe { getpagesize() } as usize;
        let map_len = file_len.div_ceil(page) * page;
        let ptr = unsafe {
            mmap(std::ptr::null_mut(), map_len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0)
        };
        if ptr == MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mapping { ptr, map_len, file_len, path: path.to_path_buf() })
    }

    /// Hint the kernel to start faulting the range in. Called before the first
    /// forward pass so the GPU does not stall on cold pages mid-decode.
    pub fn will_need(&self) {
        unsafe {
            madvise(self.ptr, self.map_len, MADV_WILLNEED);
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn slice(&self, offset: usize, len: usize) -> &[u8] {
        assert!(offset + len <= self.file_len, "slice past EOF in {:?}", self.path);
        unsafe { std::slice::from_raw_parts(self.as_ptr().add(offset), len) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr, self.map_len);
        }
    }
}

// --- the archive ------------------------------------------------------------

pub struct SafeTensors {
    pub files: Vec<Mapping>,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    /// Read one `.safetensors` file.
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let mut st = SafeTensors { files: Vec::new(), tensors: HashMap::new() };
        st.add_file(path)?;
        Ok(st)
    }

    /// Read every shard listed in a `model.safetensors.index.json`, plus any
    /// extra files given explicitly (the MTP sidecar lives outside the index).
    pub fn open_many(paths: &[PathBuf]) -> Result<SafeTensors, String> {
        let mut st = SafeTensors { files: Vec::new(), tensors: HashMap::new() };
        for p in paths {
            st.add_file(p)?;
        }
        Ok(st)
    }

    fn add_file(&mut self, path: &Path) -> Result<(), String> {
        let map = Mapping::open(path).map_err(|e| format!("mmap {:?}: {e}", path))?;
        if map.file_len < 8 {
            return Err(format!("{:?}: too short for a safetensors header", path));
        }
        let hdr_len = u64::from_le_bytes(map.slice(0, 8).try_into().unwrap()) as usize;
        if 8 + hdr_len > map.file_len {
            return Err(format!("{:?}: header length {hdr_len} past EOF", path));
        }
        let header = json::parse(map.slice(8, hdr_len))
            .map_err(|e| format!("{:?}: {e}", path))?;
        let obj = header.as_obj().ok_or_else(|| format!("{:?}: header is not an object", path))?;

        let data_start = 8 + hdr_len;
        let file_idx = self.files.len();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_s = meta
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{name}: missing dtype"))?;
            let dtype = Dtype::parse(dtype_s)
                .ok_or_else(|| format!("{name}: unsupported dtype {dtype_s}"))?;
            let shape: Vec<usize> = meta
                .get("shape")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| format!("{name}: missing shape"))?
                .iter()
                .map(|v| v.as_usize().unwrap_or(0))
                .collect();
            let offs = meta
                .get("data_offsets")
                .and_then(|v| v.as_arr())
                .ok_or_else(|| format!("{name}: missing data_offsets"))?;
            let begin = offs[0].as_usize().unwrap();
            let end = offs[1].as_usize().unwrap();
            let nbytes = end - begin;
            let expect: usize = shape.iter().product::<usize>() * dtype.size();
            if nbytes != expect {
                return Err(format!(
                    "{name}: data_offsets span {nbytes} bytes but shape {shape:?} x {dtype_s} needs {expect}"
                ));
            }
            if data_start + end > map.file_len {
                return Err(format!("{name}: data past EOF in {:?}", path));
            }
            self.tensors.insert(
                name.clone(),
                TensorInfo {
                    file: file_idx,
                    dtype,
                    shape,
                    offset: data_start + begin,
                    nbytes,
                },
            );
        }
        self.files.push(map);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn bytes(&self, info: &TensorInfo) -> &[u8] {
        self.files[info.file].slice(info.offset, info.nbytes)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    pub fn will_need(&self) {
        for f in &self.files {
            f.will_need();
        }
    }
}

/// Resolve `model.safetensors.index.json` into the shard list, in a stable
/// order so buffer indices are reproducible between runs.
pub fn shards_from_index(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let idx_path = model_dir.join("model.safetensors.index.json");
    let raw = std::fs::read(&idx_path).map_err(|e| format!("{:?}: {e}", idx_path))?;
    let idx = json::parse(&raw).map_err(|e| format!("{:?}: {e}", idx_path))?;
    let map = idx
        .get("weight_map")
        .and_then(|v| v.as_obj())
        .ok_or_else(|| format!("{:?}: no weight_map", idx_path))?;
    let mut files: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
    files.sort();
    files.dedup();
    Ok(files.into_iter().map(|f| model_dir.join(f)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> PathBuf {
        PathBuf::from(std::env::var("ORNITH_MODEL").unwrap_or_else(|_| {
            "/Users/qroc/.mtplx/models/ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx".into()
        }))
    }

    #[test]
    fn reads_the_index_and_headers() {
        let dir = model_dir();
        if !dir.exists() {
            eprintln!("model not present, skipping");
            return;
        }
        // How many body shards there are is not a property of the model: it
        // is `mlx_lm`'s `MAX_FILE_SIZE_GB` greedily packing the tensors, and
        // it moved from 17 to 5 when the artifact was rebuilt on a newer
        // toolchain. What the index owes us is that it resolves every shard
        // it names, that the vision tower is one of them, and that the whole
        // set carries the 1970 tensors the artifact is made of.
        let shards = shards_from_index(&dir).unwrap();
        assert!(shards.len() >= 2, "at least one body shard and the vision one");
        assert!(
            shards.iter().any(|p| p.file_name().unwrap().to_str().unwrap().contains("vision")),
            "the vision shard is in the index"
        );
        let st = SafeTensors::open_many(&shards).unwrap();
        assert_eq!(st.tensors.len(), 1970);

        let e = st.get("language_model.model.embed_tokens.weight").unwrap();
        assert_eq!(e.dtype, Dtype::U32);
        assert_eq!(e.shape, vec![248320, 512]);

        let n = st.get("language_model.model.norm.weight").unwrap();
        assert_eq!(n.dtype, Dtype::BF16);
        assert_eq!(n.shape, vec![2048]);
        assert_eq!(st.bytes(n).len(), 2048 * 2);
    }

    /// The artifact packs tensors back to back with no padding, so offsets
    /// land on odd byte boundaries. That rules out pointing a Metal buffer at
    /// the mapping and addressing tensors by offset, and is why the loader
    /// repacks each shard into an aligned device buffer -- which is also what
    /// mlx does (`load_safetensors` builds a `Load` primitive that reads each
    /// tensor into a fresh allocation).
    #[test]
    fn tensor_offsets_are_not_aligned_so_we_must_repack() {
        let dir = model_dir();
        if !dir.exists() {
            return;
        }
        let shards = shards_from_index(&dir).unwrap();
        let st = SafeTensors::open_many(&shards).unwrap();
        let unaligned = st.tensors.values().filter(|i| i.offset % 16 != 0).count();
        assert!(unaligned > 0, "if this ever becomes 0, zero-copy weights are possible");
    }
}
