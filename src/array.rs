//! The array type: a device buffer plus a shape and a dtype.
//!
//! mlx's `array` is a lazy graph node with strides, broadcasting and a whole
//! evaluation engine behind it. Here arrays are eager and always row-contiguous
//! -- every kernel this port dispatches either takes contiguous inputs or is
//! given the batch strides explicitly, which is exactly how mlx calls them
//! after its own `ensure_row_contiguous`.

use std::rc::Rc;

use crate::metal::{Buffer, Context};
pub use crate::safetensors::Dtype;

#[derive(Clone)]
pub struct Array {
    pub buf: Rc<Buffer>,
    /// Byte offset into `buf`. Non-zero for views: one expert out of a stacked
    /// MoE weight, one half of a fused projection, one layer's slice of a
    /// cache. mlx expresses these as strided views; here a view is always a
    /// contiguous sub-range, which is all the model needs.
    pub offset: usize,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl Array {
    pub fn empty(ctx: &Context, shape: &[usize], dtype: Dtype) -> Array {
        let n: usize = shape.iter().product();
        Array {
            buf: Rc::new(ctx.alloc(n * dtype.size())),
            offset: 0,
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Size of dimension `i`, counted from the end when negative.
    pub fn dim(&self, i: isize) -> usize {
        let n = self.shape.len() as isize;
        self.shape[((i % n) + n) as usize % n as usize]
    }

    /// Row-major strides in elements, like `mlx::array::strides()`.
    pub fn strides(&self) -> Vec<i64> {
        let mut s = vec![1i64; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.shape[i + 1] as i64;
        }
        s
    }

    /// A view with a different shape over the same buffer (no copy), the
    /// equivalent of `mx.reshape` on a contiguous array.
    pub fn reshape(&self, shape: &[usize]) -> Array {
        let n: usize = shape.iter().product();
        assert_eq!(n, self.numel(), "reshape {:?} -> {:?} changes size", self.shape, shape);
        Array { buf: self.buf.clone(), offset: self.offset, shape: shape.to_vec(), dtype: self.dtype }
    }

    /// `a[i]` along the leading axis, as a contiguous view.
    pub fn index(&self, i: usize) -> Array {
        assert!(self.ndim() >= 1 && i < self.shape[0], "index {i} out of range for {:?}", self.shape);
        let sub: usize = self.shape[1..].iter().product();
        Array {
            buf: self.buf.clone(),
            offset: self.offset + i * sub * self.dtype.size(),
            shape: self.shape[1..].to_vec(),
            dtype: self.dtype,
        }
    }

    /// `a[..., start..start + len]` along the last axis, valid only when the
    /// array is 1-D or the slice covers whole rows (leading axes are kept).
    pub fn slice_last(&self, start: usize, len: usize) -> Array {
        let last = self.shape[self.ndim() - 1];
        assert!(start + len <= last);
        assert!(self.ndim() == 1, "slice_last on >1-D would not stay contiguous");
        Array {
            buf: self.buf.clone(),
            offset: self.offset + start * self.dtype.size(),
            shape: vec![len],
            dtype: self.dtype,
        }
    }

    /// `a[start..start + len]` along the leading axis.
    pub fn slice_leading(&self, start: usize, len: usize) -> Array {
        assert!(start + len <= self.shape[0]);
        let sub: usize = self.shape[1..].iter().product();
        let mut shape = self.shape.clone();
        shape[0] = len;
        Array {
            buf: self.buf.clone(),
            offset: self.offset + start * sub * self.dtype.size(),
            shape,
            dtype: self.dtype,
        }
    }

    // --- host transfer ------------------------------------------------------

    /// Upload host data.
    ///
    /// The allocation deliberately bypasses the buffer pool. A pooled buffer
    /// can be one an already-encoded dispatch still reads -- encoding is not
    /// execution -- and writing it from the CPU before that command buffer runs
    /// silently corrupts the earlier op. A fresh allocation is referenced by
    /// nothing, so the write is safe without synchronising the queue.
    pub fn from_slice<T: Copy>(ctx: &Context, data: &[T], shape: &[usize], dtype: Dtype) -> Array {
        let n: usize = shape.iter().product();
        assert_eq!(n * dtype.size(), size_of_val(data), "data does not match shape x dtype");
        let a = Array {
            buf: Rc::new(ctx.alloc_exact(n * dtype.size())),
            offset: 0,
            shape: shape.to_vec(),
            dtype,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                a.contents(),
                size_of_val(data),
            );
        }
        a
    }

    pub fn from_f32(ctx: &Context, data: &[f32], shape: &[usize]) -> Array {
        Array::from_slice(ctx, data, shape, Dtype::F32)
    }

    pub fn from_f32_as_bf16(ctx: &Context, data: &[f32], shape: &[usize]) -> Array {
        let bits: Vec<u16> = data.iter().map(|v| f32_to_bf16(*v)).collect();
        Array::from_slice(ctx, &bits, shape, Dtype::BF16)
    }

    /// Host pointer to this array's first element.
    pub fn contents(&self) -> *mut u8 {
        unsafe { self.buf.contents().add(self.offset) }
    }

    /// Copy back to the host as f32. Syncs first: the caller is reading memory
    /// a dispatch may still be writing.
    pub fn to_vec_f32(&self, ctx: &Context) -> Vec<f32> {
        ctx.sync();
        let n = self.numel();
        match self.dtype {
            Dtype::F32 => unsafe { self.buf.as_slice::<f32>(self.offset, n).to_vec() },
            Dtype::BF16 => unsafe {
                self.buf.as_slice::<u16>(self.offset, n).iter().map(|b| bf16_to_f32(*b)).collect()
            },
            Dtype::F16 => unsafe {
                self.buf.as_slice::<u16>(self.offset, n).iter().map(|b| f16_to_f32(*b)).collect()
            },
            Dtype::U32 => unsafe {
                self.buf.as_slice::<u32>(self.offset, n).iter().map(|v| *v as f32).collect()
            },
            Dtype::I32 => unsafe {
                self.buf.as_slice::<i32>(self.offset, n).iter().map(|v| *v as f32).collect()
            },
            d => panic!("to_vec_f32 for {}", d.name()),
        }
    }

    pub fn to_vec_u32(&self, ctx: &Context) -> Vec<u32> {
        ctx.sync();
        let n = self.numel();
        match self.dtype {
            Dtype::U32 => unsafe { self.buf.as_slice::<u32>(self.offset, n).to_vec() },
            Dtype::I32 => unsafe {
                self.buf.as_slice::<i32>(self.offset, n).iter().map(|v| *v as u32).collect()
            },
            d => panic!("to_vec_u32 for {}", d.name()),
        }
    }

    /// The mlx type name used in kernel names (`get_type_string`).
    pub fn type_string(&self) -> &'static str {
        match self.dtype {
            Dtype::F32 => "float",
            Dtype::F16 => "float16_t",
            Dtype::BF16 => "bfloat16_t",
            Dtype::U32 => "uint32_t",
            Dtype::I32 => "int32_t",
            d => panic!("no kernel type string for {}", d.name()),
        }
    }
}

// --- scalar conversions -----------------------------------------------------

/// bfloat16 -> f32 is just a left shift: bf16 is the top 16 bits of an f32.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 -> bfloat16, round-to-nearest-even, matching Metal's `bfloat16_t`
/// conversion (`vendor/mlx/mlx/backend/metal/kernels/bf16.h`).
pub fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let lsb = (bits >> 16) & 1;
    let rounded = bits + 0x7fff + lsb;
    (rounded >> 16) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal: renormalise.
            let shift = frac.leading_zeros() - 21;
            let e = 127 - 15 - shift;
            (sign << 31) | (e << 23) | ((frac << (shift + 1)) & 0x7f_ffff)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 3.14159, -2.71828, 1e-8, 1e8] {
            let r = bf16_to_f32(f32_to_bf16(v));
            let rel = if v == 0.0 { r.abs() } else { ((r - v) / v).abs() };
            assert!(rel < 0.01, "{v} -> {r}");
        }
        assert_eq!(f32_to_bf16(0.0), 0);
        // Round to nearest even at the tie: 1.0 + 2^-9 sits exactly between
        // two bf16 values and must round to the one with an even mantissa.
        let tie = f32::from_bits(0x3F80_8000);
        assert_eq!(f32_to_bf16(tie), 0x3F80);
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333252).abs() < 1e-5);
    }

    #[test]
    fn host_roundtrip() {
        let ctx = Context::new().unwrap();
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 8.0).collect();
        let a = Array::from_f32(&ctx, &data, &[8, 8]);
        assert_eq!(a.to_vec_f32(&ctx), data);
        let b = Array::from_f32_as_bf16(&ctx, &data, &[8, 8]);
        let back = b.to_vec_f32(&ctx);
        for (x, y) in data.iter().zip(back.iter()) {
            assert!((x - y).abs() <= 0.06, "{x} vs {y}");
        }
    }

    #[test]
    fn strides_are_row_major() {
        let ctx = Context::new().unwrap();
        let a = Array::empty(&ctx, &[2, 3, 4], Dtype::BF16);
        assert_eq!(a.strides(), vec![12, 4, 1]);
        assert_eq!(a.reshape(&[6, 4]).strides(), vec![4, 1]);
    }
}
