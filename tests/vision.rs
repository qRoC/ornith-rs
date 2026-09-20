//! The vision tower against MTPLX, step by step and end to end.
//!
//! Reference: tools/dump_vision.py
//!
//! Three images, so every dispatcher branches both ways: 256x256 (256 patch
//! rows -- aligned everywhere, split-K projections), 320x224 (280 rows --
//! aligned nowhere) and 512x512 (1024 rows -- the regular steel gemm).
//!
//! The tower is 0.9 GB, which fits alongside anything, so unlike the trunk
//! tests these run in the default `cargo test`.

use ornith_rs::array::{Array, Dtype};
use ornith_rs::metal::Context;
use ornith_rs::refs::{RefCase, compare, model_dir};
use ornith_rs::vision::preprocess::{Image, PreprocessorConfig, preprocess};
use ornith_rs::vision::tower::VisionTower;
use ornith_rs::vision::VisionConfig;
use ornith_rs::weights::WeightStore;

/// Already a multiple of `patch_size * merge_size`: `smart_resize` is a no-op
/// and the tower sees the file's own pixels.
const TAGS: [&str; 3] = ["256x256", "320x224", "512x512"];
/// Sizes `smart_resize` changes, so the composed preprocessing goes through
/// Pillow's resampler. 700x500 rounds up in both axes; 300x220 lands under
/// `min_pixels` and takes the other branch.
const RESIZE_TAGS: [&str; 2] = ["700x500", "300x220"];

fn exact(name: &str, got: &[f32], want: &[f32]) {
    let d = compare(got, want);
    assert_eq!(d.max_abs, 0.0, "{name}: {d}");
}

fn read_image(case: &RefCase, tag: &str) -> Image {
    Image::read(&case.dir.join(format!("img_{tag}.ppm")), None).unwrap()
}

/// The preprocessing, against `preprocess_images`: reading the pixels,
/// `smart_resize` (a no-op for these sizes), the rescale and normalise, and
/// the reshape into one row per patch with the 2x2 merge blocks contiguous.
#[test]
fn preprocess_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("vision") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let pre = PreprocessorConfig::load(&dir).unwrap();
    for tag in TAGS.iter().chain(RESIZE_TAGS.iter()) {
        let image = read_image(&case, tag);
        let p = preprocess(&image, &pre).unwrap();
        let want_grid = case.get(&format!("{tag}_grid")).to_u32();
        assert_eq!(
            [p.grid.0 as u32, p.grid.1 as u32, p.grid.2 as u32],
            want_grid[..],
            "{tag}: grid"
        );
        exact(&format!("{tag} pixels"), &p.values, &case.f32(&format!("{tag}_pixels")));
    }
}

/// The host half of `fast_pos_embed_interpolate`: `mx.linspace`, the
/// truncating cast to int32, and the four bilinear weights -- all float32,
/// then rounded to bfloat16 as `mx.array(weight_list, dtype=bfloat16)` does.
#[test]
fn interpolation_table_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("vision") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let (cfg, _) = VisionConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let tower = VisionTower::load(&ctx, &mut ws, cfg);

    for tag in TAGS {
        let grid = grid_of(&case, tag);
        let (idx, wts, total) = tower.interpolation_table(&[grid]);
        assert_eq!(total, grid.1 * grid.2);
        let want_idx = case.get(&format!("{tag}_interp_idx")).to_u32();
        assert_eq!(idx, want_idx, "{tag}: interpolation indices");
        let got: Vec<f32> = wts.iter().map(|b| ornith_rs::array::bf16_to_f32(*b)).collect();
        exact(&format!("{tag} interpolation weights"), &got, &case.f32(&format!("{tag}_interp_w")));
    }
}

fn grid_of(case: &RefCase, tag: &str) -> (usize, usize, usize) {
    let g = case.get(&format!("{tag}_grid")).to_u32();
    (g[0] as usize, g[1] as usize, g[2] as usize)
}

/// The tower itself, checked at every seam.
#[test]
fn tower_matches_mlx() {
    let Some(case) = RefCase::open_or_skip("vision") else { return };
    let dir = model_dir();
    if !dir.exists() {
        return;
    }
    let ctx = Context::new().unwrap();
    let (cfg, _) = VisionConfig::load(&dir).unwrap();
    let mut ws = WeightStore::open(&dir, &[]).unwrap();
    let tower = VisionTower::load(&ctx, &mut ws, cfg);
    let taps: Vec<usize> = case
        .meta()
        .unwrap()
        .get("taps")
        .unwrap()
        .as_arr()
        .unwrap()
        .iter()
        .map(|v| v.as_usize().unwrap())
        .collect();

    for tag in TAGS {
        let grid = grid_of(&case, tag);
        let pixels = case.array(&ctx, &format!("{tag}_pixels"));
        assert_eq!(pixels.dtype, Dtype::F32);

        // --- patch embedding ---------------------------------------------
        let embedded = tower.patch_embed(&ctx, &pixels);
        exact(
            &format!("{tag} patch_embed"),
            &embedded.to_vec_f32(&ctx),
            &case.f32(&format!("{tag}_patch_embed")),
        );

        // --- interpolated position embedding ------------------------------
        let pos = tower.fast_pos_embed_interpolate(&ctx, &[grid]);
        exact(
            &format!("{tag} pos_embed"),
            &pos.to_vec_f32(&ctx),
            &case.f32(&format!("{tag}_pos_embed")),
        );
        let hidden = ornith_rs::ops::elementwise::add(&ctx, &embedded, &pos);
        exact(
            &format!("{tag} embedded"),
            &hidden.to_vec_f32(&ctx),
            &case.f32(&format!("{tag}_embedded")),
        );

        // --- 2D rotary embedding ------------------------------------------
        let (cos, sin) = tower.rot_pos_emb(&ctx, &[grid]);
        exact(&format!("{tag} cos"), &cos.to_vec_f32(&ctx), &case.f32(&format!("{tag}_cos")));
        exact(&format!("{tag} sin"), &sin.to_vec_f32(&ctx), &case.f32(&format!("{tag}_sin")));

        // --- block 0, piece by piece ---------------------------------------
        let blk = &tower.blocks[0];
        let n1 = ornith_rs::ops::norm::layer_norm(
            &ctx,
            &hidden,
            &blk.norm1.weight,
            &blk.norm1.bias,
            1e-6,
        );
        exact(
            &format!("{tag} block0 norm1"),
            &n1.to_vec_f32(&ctx),
            &case.f32(&format!("{tag}_b0_norm1")),
        );
        let attn = tower.attention(&ctx, blk, &n1, &cos, &sin);
        exact(
            &format!("{tag} block0 attn"),
            &attn.to_vec_f32(&ctx),
            &case.f32(&format!("{tag}_b0_attn")),
        );

        // --- every block, then the merger ----------------------------------
        let mut h = hidden;
        for (i, blk) in tower.blocks.iter().enumerate() {
            h = tower.block_forward(&ctx, blk, &h, &cos, &sin);
            if taps.contains(&i) {
                exact(
                    &format!("{tag} block{i}"),
                    &h.to_vec_f32(&ctx),
                    &case.f32(&format!("{tag}_block{i}")),
                );
            }
        }
        let merged = tower.merge(&ctx, &h);
        exact(
            &format!("{tag} merged"),
            &merged.to_vec_f32(&ctx),
            &case.f32(&format!("{tag}_merged")),
        );

        // ... and the whole thing in one call.
        let one_shot = tower.forward(&ctx, &pixels, &[grid]);
        exact(
            &format!("{tag} forward"),
            &one_shot.to_vec_f32(&ctx),
            &case.f32(&format!("{tag}_merged")),
        );
        assert_eq!(
            one_shot.shape,
            vec![grid.0 * grid.1 * grid.2 / 4, tower.cfg.out_hidden_size]
        );
    }
}

/// `_expand_image_pads` and the splice's row bookkeeping. The scatter itself
/// is covered end to end by `tests/vision_generate.rs`, where a misplaced row
/// would change the generated text.
#[test]
fn pad_expansion_matches_mtplx() {
    use ornith_rs::vision::splice::{VisionSplice, expand_image_pads};
    let ctx = Context::new().unwrap();

    let pad = 248056u32;
    let ids = [1u32, pad, 2, pad, 3];
    let expanded = expand_image_pads(&ids, pad, &[3, 2]).unwrap();
    assert_eq!(expanded, vec![1, pad, pad, pad, 2, pad, pad, 3]);

    // A synthetic "embedding table" and "tower rows" -- the point of the test
    // is the scatter, not the arithmetic.
    let dims = 4usize;
    let rows_data: Vec<f32> = (0..5 * dims).map(|i| 100.0 + i as f32).collect();
    let rows = Array::from_f32_as_bf16(&ctx, &rows_data, &[5, dims]);
    let mut splice = VisionSplice {
        image_pad_token_id: pad,
        embeddings: rows,
        pad_counts: vec![3, 2],
        // No cache to reuse in a test, so nothing keys the images.
        keys: Vec::new(),
        cursor: 0,
    };
    assert_eq!(splice.total_rows(), 5);
    assert_eq!(splice.remaining(), 5);
    splice.reset();
    assert_eq!(splice.cursor, 0);
}
