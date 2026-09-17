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
use crate::ops::quantized::QuantMatrix;

pub struct FrSpec {
    /// The chosen ids, on the device. The draft chain never brings its token
    /// to the host, so the subset index has to be mapped back there too.
    ids: Array,
    /// `lm_head`'s rows for those ids, and nothing else.
    pub head: QuantMatrix,
    pub n: usize,
    pub vocab: usize,
}

impl FrSpec {
    /// Read the id list and gather the head's rows for it.
    ///
    /// The gather is `mx.take(..., axis=0)` on each of `weight`, `scales` and
    /// `biases`, which is what `install_frspec_draft_head` does and what this
    /// port already does for every embedding lookup.
    pub fn load(ctx: &Context, lm_head: &QuantMatrix, path: &Path) -> Result<FrSpec, String> {
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
        let head = QuantMatrix {
            w: gather_rows(ctx, &lm_head.w, &ids, lm_head.w.shape[1]),
            scales: gather_rows(ctx, &lm_head.scales, &ids, lm_head.scales.shape[1]),
            // nvfp4 has no zero point, so there is a third row gather in
            // affine mode and not in nvfp4 -- the pruning is per row and
            // indifferent to what a row holds.
            biases: lm_head
                .biases
                .as_ref()
                .map(|b| gather_rows(ctx, b, &ids, b.shape[1])),
            mode: lm_head.mode,
            group_size: lm_head.group_size,
            bits: lm_head.bits,
        };
        Ok(FrSpec { ids, head, n, vocab })
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
