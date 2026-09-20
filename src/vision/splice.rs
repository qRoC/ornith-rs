//! `VisionSplice` and `spliced_chunk_embeddings` from
//! `vendor/MTPLX/mtplx/vision/splice.py`.
//!
//! The chat template renders one `<|image_pad|>` per image; the request then
//! expands it to as many copies as the tower produced rows
//! (`_expand_image_pads`), and prefill replaces those rows' *embeddings* with
//! the tower's output. So the image never goes through `embed_tokens` at all:
//! the pad id only marks the positions.
//!
//! Prefill consumes the prompt strictly left to right, so the splice is a
//! sequential queue -- one cursor into the tower's rows, advanced by the
//! number of pads each chunk held. Starting the cursor past zero is how a
//! reused cache is continued: the rows before it are already in the cache.
//!
//! `keys` is MTPLX's `vision_bank_key_ids`, and the reason it has to exist
//! is that ids cannot do its job -- every image expands to the same
//! `<|image_pad|>`, so a token-id prefix match says two different pictures
//! are the same prompt. MTPLX's `vision_image_spans` has no counterpart
//! here, because a cursor over a queue already knows where the spans are.

use crate::array::{Array, Dtype};
use crate::metal::Context;
use crate::ops::embedding::embedding;
use crate::ops::quantized::QuantMatrix;

/// Per-request vision state consumed during prefill.
pub struct VisionSplice {
    pub image_pad_token_id: u32,
    /// `[total_pad_tokens, text_hidden_size]`, in prompt order.
    pub embeddings: Array,
    /// How many pad tokens each image expands to, in prompt order -- what
    /// `expand_image_pads` used, kept so an underflow can say which image ran
    /// out. MTPLX also carries the raw `(t, h, w)` grids here, but only to
    /// build an M-RoPE table, and this family does not take that path.
    pub pad_counts: Vec<usize>,
    /// A content digest per image, in prompt order -- MTPLX's
    /// `vision_bank_key_ids`. The ids alone cannot tell two pictures apart,
    /// so this is what a cache-reuse check compares.
    pub keys: Vec<u64>,
    pub cursor: usize,
}

impl VisionSplice {
    pub fn total_rows(&self) -> usize {
        self.embeddings.shape[0]
    }

    pub fn remaining(&self) -> usize {
        self.total_rows() - self.cursor
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// How many images a run of `pad_rows` pad tokens covers, or `None` if it
    /// stops inside one. Every pad token is one tower row, so a prefix of the
    /// prompt holds a whole number of images or it holds none usefully.
    pub fn images_in(&self, pad_rows: usize) -> Option<usize> {
        let mut acc = 0;
        for (i, n) in self.pad_counts.iter().enumerate() {
            if acc == pad_rows {
                return Some(i);
            }
            acc += n;
        }
        (acc == pad_rows).then_some(self.pad_counts.len())
    }

    /// The pad tokens in `ids`, which is both the number of rows they consume
    /// and the cursor a continuation starts from.
    pub fn pad_rows_in(&self, ids: &[u32]) -> usize {
        ids.iter().filter(|&&t| t == self.image_pad_token_id).count()
    }
}

/// `_expand_image_pads`: replace each `<|image_pad|>` in the rendered prompt
/// with `pad_counts[i]` copies of itself.
pub fn expand_image_pads(
    prompt_ids: &[u32],
    image_pad_id: u32,
    pad_counts: &[usize],
) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(prompt_ids.len());
    let mut image = 0usize;
    for &t in prompt_ids {
        if t == image_pad_id {
            if image >= pad_counts.len() {
                return Err("prompt contains more image placeholders than images".into());
            }
            out.extend(std::iter::repeat_n(t, pad_counts[image]));
            image += 1;
        } else {
            out.push(t);
        }
    }
    if image != pad_counts.len() {
        return Err(format!(
            "chat template dropped image placeholders: expected {}, found {image}",
            pad_counts.len()
        ));
    }
    Ok(out)
}

/// `spliced_chunk_embeddings`: embed one prefill chunk, replacing the
/// image-pad rows with the next rows from the queue.
///
/// Returns `None` when the chunk holds no pads, so the caller keeps the plain
/// token-id path -- which is what mlx does, and which matters because the
/// splice is the only thing that forces `embed_tokens` to be materialised
/// separately from the first layer.
pub fn spliced_chunk_embeddings(
    ctx: &Context,
    embed_tokens: &QuantMatrix,
    chunk: &[u32],
    ids: &Array,
    splice: &mut VisionSplice,
) -> Option<Array> {
    // mlx counts the pads and finds their positions on the host too
    // (`mask.sum().item()` and a list comprehension over `flat_mask.tolist()`).
    let positions: Vec<u32> = chunk
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == splice.image_pad_token_id)
        .map(|(i, _)| i as u32)
        .collect();
    if positions.is_empty() {
        return None;
    }
    assert!(
        splice.remaining() >= positions.len(),
        "vision splice underflow: prompt has more image pad tokens ({}) than \
         vision rows ({})",
        splice.cursor + positions.len(),
        splice.total_rows()
    );

    let embedded = embedding(ctx, embed_tokens, ids);
    let dims = embedded.dim(-1);
    let rows = splice.embeddings.slice_leading(splice.cursor, positions.len());
    splice.cursor += positions.len();
    assert_eq!(rows.dtype, embedded.dtype, "the tower's rows must match the embedding dtype");

    let pos = Array::from_slice(ctx, &positions, &[positions.len()], Dtype::U32);
    let n = rows.numel();
    let tg = 256.min(n.next_power_of_two()).max(1);
    ctx.kernel(&format!(
        "splice_rows_{}",
        match embedded.dtype {
            Dtype::BF16 => "bfloat16",
            Dtype::F32 => "float32",
            d => panic!("no splice for {}", d.name()),
        }
    ))
    // `splice_rows` takes the embedding buffer as `device T* flat` and writes
    // the tower's rows into it in place.
    .out(&embedded)
    .arr(&rows)
    .arr(&pos)
    .bytes(&(dims as u32))
    .threads((n, 1, 1), (tg, 1, 1));
    Some(embedded)
}
