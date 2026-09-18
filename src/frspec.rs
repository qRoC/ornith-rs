//! FR-Spec: a draft head over a frequency-ranked slice of the vocabulary.
//!
//! Port of MTPLX's `mtplx/frspec_draft.py`. A draft step's `lm_head` is the
//! single most expensive read in the speculative loop -- 8-bit over 248320
//! rows is ~540 MB, and a depth-3 chain reads it three times, which is most of
//! a whole trunk forward spent on tokens that are then mostly rejected. FR-Spec
//! drafts from a pruned head instead and lets the target keep verifying over
//! the full vocabulary.
//!
//! **It cannot change what the model emits.** Every committed token is the
//! verify forward's own argmax: the loop accepts draft `i` only where it
//! already equals target `i`, and takes target `k` itself otherwise. So the
//! draft decides how many tokens a cycle commits, never which. Pruning its
//! support moves the acceptance rate and nothing else -- MTPLX makes the same
//! argument from the other side, where probability-ratio acceptance is valid
//! for any proposal distribution.
//!
//! Where MTPLX scatters the pruned logits back into a full-vocabulary row of
//! `-1e30` and argmaxes that, this argmaxes the pruned row and maps the index
//! back through the id list. The two pick the same token -- a row that was not
//! gathered cannot win against one that was -- and the second never builds the
//! 248320-wide row at all. MTPLX's `MTPLX_FRSPEC_LEGACY` lane does the same
//! remap for the same reason.
//!
//! The id list is not in this repository: it is MTPLX's artifact, and
//! `tools/dump_frspec.py` is its writer. See that file for what it is and
//! what its coverage costs.

use std::path::{Path, PathBuf};

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::embedding::gather_rows;
use crate::ops::kquant::KMatrix;
use crate::ops::linear::Linear;
use crate::ops::quantized::QuantMatrix;

pub struct FrSpec {
    /// The chosen ids, on the device. The draft chain never brings its token
    /// to the host, so the subset index has to be mapped back there too.
    ids: Array,
    /// `lm_head`'s rows for those ids, and nothing else.
    pub head: Linear,
    pub n: usize,
    pub vocab: usize,
}

/// `lm_head[ids]`, in whichever family the head belongs to.
///
/// **Pruning is a row gather and nothing else** -- `mx.take(..., axis=0)` on
/// each tensor the module holds, which is what `install_frspec_draft_head`
/// does. What differs between the families is only how many tensors that is
/// and what they hold, and a row is a whole number of each in all of them:
///
///   affine  `weight` + `scales` + `biases`
///   nvfp4   `weight` + `scales`, uint8 E4M3, no zero point
///   k-quant `qs` + `scales` + `dm`, and a row is `I / 256` whole super-blocks
///           -- 8 of them at this model's 2048-wide head -- so the 12-byte
///           scale packing and the nibble interleave are inside the blocks
///           being copied and never have to be understood here.
///
/// This file used to refuse the last one (`as_mlx` panicked on it), which made
/// FR-Spec unavailable on exactly the artifacts that most want it: a k-quant
/// decode step is 13.1% cheaper, so the draft chain's fixed cost is
/// proportionally larger, and `--mtp` measured *worse* than autoregression on
/// `q6k-full` for want of this gather.
pub fn prune(ctx: &Context, lm_head: &Linear, ids: &Array, n: usize) -> Linear {
    match lm_head {
        Linear::Mlx(q) => Linear::Mlx(QuantMatrix {
            w: gather_rows(ctx, &q.w, ids, q.w.shape[1]),
            scales: gather_rows(ctx, &q.scales, ids, q.scales.shape[1]),
            // nvfp4 has no zero point, so there is a third row gather in
            // affine mode and not in nvfp4 -- the pruning is per row and
            // indifferent to what a row holds.
            biases: q.biases.as_ref().map(|b| gather_rows(ctx, b, ids, b.shape[1])),
            mode: q.mode,
            group_size: q.group_size,
            bits: q.bits,
        }),
        Linear::K(q) => {
            // `scales` and `dm` are `[O, blocks, fields]`; the gather takes a
            // whole row of each and the block axis is put back by shape.
            let (sb, db) = (q.scales.shape[1], q.dm.shape[1]);
            let (sf, df) = (q.scales.shape[2], q.dm.shape[2]);
            Linear::K(KMatrix::new(
                q.kind,
                gather_rows(ctx, &q.qs, ids, q.qs.shape[1]),
                gather_rows(ctx, &q.scales, ids, sb * sf).reshape(&[n, sb, sf]),
                gather_rows(ctx, &q.dm, ids, db * df).reshape(&[n, db, df]),
            ))
        }
    }
}

impl FrSpec {
    /// Read the id list and gather the head's rows for it.
    ///
    /// The gather is `mx.take(..., axis=0)` on each of `weight`, `scales` and
    /// `biases`, which is what `install_frspec_draft_head` does and what this
    /// port already does for every embedding lookup.
    pub fn load(ctx: &Context, lm_head: &Linear, path: &Path) -> Result<FrSpec, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() < 4 {
            return Err(format!("{}: too short for a count", path.display()));
        }
        let n = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;
        if raw.len() != 4 + n * 4 {
            return Err(format!(
                "{}: says {n} ids but carries {} bytes",
                path.display(),
                raw.len() - 4
            ));
        }
        let ids: Vec<u32> = raw[4..]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // The same checks MTPLX makes, because a bad list here is a wrong
        // token rather than a slow one.
        let vocab = lm_head.out_dims();
        if n == 0 || n >= vocab {
            return Err(format!("{n} ids is not a pruning of {vocab}"));
        }
        if ids.iter().any(|&i| i as usize >= vocab) {
            return Err(format!("an id is outside the {vocab}-row vocabulary"));
        }
        let mut seen = std::collections::HashSet::with_capacity(n);
        if !ids.iter().all(|i| seen.insert(*i)) {
            return Err("the id list has duplicates".into());
        }

        let ids = Array::from_slice(ctx, &ids, &[n], Dtype::U32);
        Ok(FrSpec { ids: ids.clone(), head: prune(ctx, lm_head, &ids, n), n, vocab })
    }

    /// Turn one index into the pruned head's rows into the token id it stands
    /// for, writing it where the caller wants it -- the draft chain's next
    /// input slot, so nothing goes to the host.
    pub fn map_id(&self, ctx: &Context, local: &Array, out: &Array) {
        ctx.kernel("gather_u32").arr(&self.ids).arr(local).out(out).threads((1, 1, 1), (1, 1, 1));
    }

    /// What the draft reads per step against what it read before.
    pub fn ratio(&self) -> f64 {
        self.n as f64 / self.vocab as f64
    }
}

/// Where the id list lives unless `ORNITH_FRSPEC` says otherwise. Written by
/// `tools/dump_frspec.py`, and outside the repository for the same reason
/// `refs/` is.
pub fn default_path() -> PathBuf {
    if let Ok(p) = std::env::var("ORNITH_FRSPEC") {
        return PathBuf::from(p);
    }
    crate::refs::refs_root().join("frspec/ids_64k.bin")
}
