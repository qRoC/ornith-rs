"""Reference outputs for the vision tower, end to end and step by step.

The reference is MTPLX's `Qwen3VLVisionTower`; mlx-lm has no vision path at all
(`qwen3_5_moe.Model.sanitize` drops every `vision_tower.*` tensor), so this is
the one case where the vendored mlx-lm is not the authority.

Three images, chosen so the dispatchers branch both ways:

  256x256  16x16 patches, 256 rows  -- M % 64 == 0, qL % 32 == 0, split-K proj
  320x224  20x14 patches, 280 rows  -- every alignment false
  512x512  32x32 patches, 1024 rows -- the regular steel gemm, not split-K

The images themselves are written next to the reference so the Rust side reads
exactly the same bytes: no image decoding happens on either side.

    cd tools && PYTHONPATH=../../vendor/mlx-lm:../../vendor/MTPLX \\
        python3 dump_vision.py
"""

import json
import os

import mlx.core as mx
import numpy as np

import ref

from mtplx.vision.processing import preprocess_images
from mtplx.vision.qwen3_vl_tower import (
    Qwen3VLVisionTower,
    _apply_rotary_pos_emb_vision,
    _fused_sdpa,
)

from PIL import Image

# (width, height). These are already multiples of patch_size * merge_size, so
# `smart_resize` is a no-op and the tower sees the file's own pixels.
SIZES = [(256, 256), (320, 224), (512, 512)]
# Sizes that `smart_resize` changes, so the composed preprocessing -- rounding,
# Pillow's bicubic resample, then the patch reorder -- is checked too. 700x500
# rounds up in both axes; 300x220 falls under `min_pixels` and is scaled up by
# the other branch.
RESIZE_SIZES = [(700, 500), (300, 220)]
TAPS = [0, 1, 26]


def make_image(width, height, seed):
    """A deterministic, structured test image.

    Structure matters: a uniform-noise image makes the position embedding and
    the rotary embedding almost invisible in the output, so a bug in either
    could hide. Gradients plus blocks plus noise exercise all of it.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    r = (xx * 255 // max(width - 1, 1)).astype(np.uint8)
    g = (yy * 255 // max(height - 1, 1)).astype(np.uint8)
    b = (((xx // 37) + (yy // 29)) % 2 * 200).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    noise = rng.integers(0, 40, size=img.shape, dtype=np.uint8)
    return (img // 2 + noise).astype(np.uint8)


def write_ppm(path, arr):
    h, w, _ = arr.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())


def capture_tower(tower, pixel_values, grid_thw, case, tag):
    """`Qwen3VLVisionTower.__call__`, with the intermediates written out.

    Copied statement for statement from qwen3_vl_tower.py so the capture
    cannot drift from the thing it is capturing.
    """
    grid = [(int(t), int(h), int(w)) for t, h, w in grid_thw]

    hidden_states = tower.patch_embed(
        pixel_values.astype(tower.patch_embed.proj.weight.dtype)
    )
    case.put(f"{tag}_patch_embed", hidden_states)

    pos = tower.fast_pos_embed_interpolate(grid)
    case.put(f"{tag}_pos_embed", pos)
    hidden_states = hidden_states + pos
    case.put(f"{tag}_embedded", hidden_states)

    rotary_pos_emb = tower.rot_pos_emb(grid)
    case.put(f"{tag}_rotary", rotary_pos_emb)
    case.put(f"{tag}_cos", mx.cos(rotary_pos_emb))
    case.put(f"{tag}_sin", mx.sin(rotary_pos_emb))

    split_indices = []
    total = 0
    for t, h, w in grid:
        for _ in range(t):
            total += h * w
            split_indices.append(total)
    split_indices = split_indices[:-1]
    assert split_indices == [], "a single still image has no interior split"

    # Block 0, broken into its four halves, so a failure in the composed
    # block test says which one.
    blk = tower.blocks[0]
    n1 = blk.norm1(hidden_states)
    case.put(f"{tag}_b0_norm1", n1)
    a = blk.attn(n1, split_indices, rotary_pos_emb)
    case.put(f"{tag}_b0_attn", a)
    h1 = hidden_states + a
    n2 = blk.norm2(h1)
    case.put(f"{tag}_b0_norm2", n2)
    m = blk.mlp(n2)
    case.put(f"{tag}_b0_mlp", m)

    # ... and the attention's own pieces.
    seq_length = n1.shape[0]
    qkv = (
        blk.attn.qkv(n1)
        .reshape(seq_length, 3, blk.attn.num_heads, -1)
        .transpose(1, 0, 2, 3)
    )
    case.put(f"{tag}_b0_qkv", qkv)
    q, k, v = mx.split(qkv, 3)
    q = _apply_rotary_pos_emb_vision(mx.expand_dims(q, 0), rotary_pos_emb)[0]
    k = _apply_rotary_pos_emb_vision(mx.expand_dims(k, 0), rotary_pos_emb)[0]
    case.put(f"{tag}_b0_q_roped", q)
    case.put(f"{tag}_b0_k_roped", k)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    o = _fused_sdpa(q, k, v, blk.attn.scale)
    case.put(f"{tag}_b0_sdpa", o)

    for layer_num, blk in enumerate(tower.blocks):
        hidden_states = blk(hidden_states, split_indices, rotary_pos_emb)
        if layer_num in TAPS:
            case.put(f"{tag}_block{layer_num}", hidden_states)

    out = tower.merger(hidden_states)
    case.put(f"{tag}_merged", out)
    return out


def main():
    case = ref.Case("vision")
    tower = Qwen3VLVisionTower.from_model_dir(ref.MODEL)

    with open(os.path.join(ref.MODEL, "preprocessor_config.json")) as f:
        pre_cfg = json.load(f)

    case.note(
        sizes=[list(s) for s in SIZES],
        resize_sizes=[list(s) for s in RESIZE_SIZES],
        taps=TAPS,
        spatial_merge_size=tower.spatial_merge_size,
        num_grid_per_side=tower.num_grid_per_side,
    )

    for seed, (w, h) in enumerate(SIZES):
        tag = f"{w}x{h}"
        arr = make_image(w, h, seed)
        path = os.path.join(case.dir, f"img_{tag}.ppm")
        write_ppm(path, arr)
        case.tensors[f"img_{tag}"] = {
            "dtype": "raw",
            "shape": [h, w, 3],
            "file": f"img_{tag}.ppm",
        }

        image = Image.open(path)
        pixel_values, grids = preprocess_images([image], pre_cfg)
        assert len(grids) == 1
        case.put(f"{tag}_pixels", pixel_values)
        case.put(
            f"{tag}_grid",
            mx.array([list(grids[0])], dtype=mx.int32),
        )

        # The host half of `fast_pos_embed_interpolate`, so a mismatch in the
        # bilinear table is distinguishable from one in the gather.
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _, gh, gw in [tuple(int(x) for x in grids[0])]:
            h_idxs = mx.linspace(0, tower.num_grid_per_side - 1, gh)
            w_idxs = mx.linspace(0, tower.num_grid_per_side - 1, gw)
            h_floor = h_idxs.astype(mx.int32)
            w_floor = w_idxs.astype(mx.int32)
            h_ceil = mx.minimum(h_floor + 1, tower.num_grid_per_side - 1)
            w_ceil = mx.minimum(w_floor + 1, tower.num_grid_per_side - 1)
            dh = h_idxs - h_floor.astype(mx.float32)
            dw = w_idxs - w_floor.astype(mx.float32)
            base_h = h_floor * tower.num_grid_per_side
            base_h_ceil = h_ceil * tower.num_grid_per_side
            indices = [
                (base_h[:, None] + w_floor[None, :]).flatten(),
                (base_h[:, None] + w_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        case.put(f"{tag}_interp_idx", mx.array(idx_list, dtype=mx.int32))
        case.put(
            f"{tag}_interp_w",
            mx.array(weight_list, dtype=tower.pos_embed.weight.dtype),
        )

        capture_tower(tower, pixel_values, grids, case, tag)
        mx.clear_cache()

    # Preprocessing only: the tower is already covered at three sizes, and what
    # these add is the resample.
    for seed, (w, h) in enumerate(RESIZE_SIZES):
        tag = f"{w}x{h}"
        arr = make_image(w, h, 100 + seed)
        path = os.path.join(case.dir, f"img_{tag}.ppm")
        write_ppm(path, arr)
        case.tensors[f"img_{tag}"] = {
            "dtype": "raw",
            "shape": [h, w, 3],
            "file": f"img_{tag}.ppm",
        }
        pixel_values, grids = preprocess_images([Image.open(path)], pre_cfg)
        case.put(f"{tag}_pixels", pixel_values)
        case.put(f"{tag}_grid", mx.array([list(grids[0])], dtype=mx.int32))
        print(f"{tag}: resized to grid {tuple(int(x) for x in grids[0])}", flush=True)

    case.write()


if __name__ == "__main__":
    main()
