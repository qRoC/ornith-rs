"""The end-to-end vision reference: an image and a prompt in, tokens out.

This is the counterpart of `dump_generate.py` for the vision path. It runs the
tower with MTPLX, splices the rows over the expanded `<|image_pad|>` run the
way `mtplx.vision.splice` does, and hands the result to mlx-lm's
`generate_step` as `input_embeddings` -- which chunks the prefill at 2048 and
decodes exactly as the text path does.

The images are written next to the reference so the Rust side reads the same
bytes; there is no image decoder on either side. The photographs go through
`sips` to BMP: one at the size `smart_resize` would ask for anyway, and one at
an ordinary photograph size, which the processor then resamples -- so the
ported Pillow resampler is on this path too.

    cd tools && PYTHONPATH=../../vendor/mlx-lm:../../vendor/MTPLX \\
        python3 dump_vision_generate.py
"""

import json
import os
import pathlib
import shutil
import subprocess

import mlx.core as mx

import ref

from mtplx.vision.processing import preprocess_images, image_pad_token_count
from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionTower

from PIL import Image

MAX_TOKENS = 96
SOURCE_PHOTO = "/System/Library/CoreServices/DefaultDesktop.heic"
# Two sizes on purpose. 768x448 is exactly what `smart_resize` asks for at this
# aspect ratio and factor 32, so the processor's resize is a no-op; 1000x667 is
# an ordinary photograph shape, and the processor resamples it to 992x672 --
# which is what puts `vision/resample.rs` on the end-to-end path.
PHOTOS = {"photo.bmp": (768, 448), "photo_odd.bmp": (1000, 667)}


def make_photos(out_dir):
    """`sips` twice per file: decode to BMP, then set the pixel dimensions."""
    for name, (w, h) in PHOTOS.items():
        bmp = os.path.join(out_dir, name)
        subprocess.run(
            ["sips", "-s", "format", "bmp", SOURCE_PHOTO, "--out", bmp],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["sips", "-z", str(h), str(w), bmp, "--out", bmp],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main():
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.utils import load

    case = ref.Case("vision_generate")

    # The photos, plus one of the synthetic images the tower test already uses.
    make_photos(case.dir)
    synthetic = os.path.join(ref.REFS, "vision", "img_256x256.ppm")
    if not os.path.exists(synthetic):
        raise SystemExit("run dump_vision.py first: it writes img_256x256.ppm")
    shutil.copy(synthetic, os.path.join(case.dir, "img_256x256.ppm"))

    with open(os.path.join(ref.MODEL, "config.json")) as f:
        cfg = json.load(f)
    image_token_id = int(cfg["image_token_id"])
    with open(os.path.join(ref.MODEL, "preprocessor_config.json")) as f:
        pre_cfg = json.load(f)

    tower = Qwen3VLVisionTower.from_model_dir(ref.MODEL)

    # Tower rows per image file, cached the way MTPLX caches them by digest.
    rows_by_file = {}

    def tower_rows(name):
        if name not in rows_by_file:
            image = Image.open(os.path.join(case.dir, name))
            pixel_values, grids = preprocess_images([image], pre_cfg)
            rows, _deep = tower(pixel_values, grids)
            mx.eval(rows)
            rows_by_file[name] = (rows, image_pad_token_count(grids[0]))
        return rows_by_file[name]

    model, tokenizer = load(pathlib.Path(ref.MODEL))

    cases = [
        (["photo.bmp"], "Describe this image in one sentence."),
        (["photo.bmp"], "What are the dominant colours here?"),
        (["img_256x256.ppm"], "What do you see in this image?"),
        (["photo.bmp", "img_256x256.ppm"], "How do these two images differ?"),
        (["photo_odd.bmp"], "Describe this image in one sentence."),
    ]

    files, prompts, all_ids, texts = [], [], [], []
    for names, question in cases:
        content = [{"type": "image"} for _ in names]
        content.append({"type": "text", "text": question})
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        blocks, pad_counts = [], []
        for name in names:
            rows, pad_count = tower_rows(name)
            blocks.append(rows)
            pad_counts.append(pad_count)
        embeddings = blocks[0] if len(blocks) == 1 else mx.concatenate(blocks, axis=0)

        # `_expand_image_pads`, then the splice over the whole prompt at once:
        # `generate_step` chunks `input_embeddings` at prefill_step_size, which
        # is the same thing MTPLX does chunk by chunk.
        expanded = []
        image_index = 0
        for t in prompt_ids:
            if t == image_token_id:
                expanded.extend([t] * pad_counts[image_index])
                image_index += 1
            else:
                expanded.append(t)
        assert image_index == len(pad_counts)

        ids = mx.array(expanded)
        embedded = model.language_model.model.embed_tokens(ids)
        mask = ids == image_token_id
        positions = mx.array(
            [i for i, hit in enumerate(mask.tolist()) if hit], dtype=mx.int32
        )
        assert int(positions.size) == int(embeddings.shape[0])
        embedded[positions] = embeddings.astype(embedded.dtype)
        mx.eval(embedded)

        cache = make_prompt_cache(model)
        toks = []
        detok = tokenizer.detokenizer
        detok.reset()
        for token, _lp in generate_step(
            ids,
            model,
            max_tokens=MAX_TOKENS,
            sampler=make_sampler(temp=0.0),
            prompt_cache=cache,
            input_embeddings=embedded,
        ):
            token = int(token)
            toks.append(token)
            if token in tokenizer.eos_token_ids:
                break
            detok.add_token(token)
        detok.finalize()

        files.append(names)
        prompts.append(prompt_text)
        all_ids.append(toks)
        texts.append(detok.text)
        print(f"{names} -> {len(toks)} tokens: {detok.text[:80]!r}", flush=True)

    case.put_json("files", files)
    case.put_json("prompts", prompts)
    case.put_json("ids", all_ids)
    case.put_json("texts", texts)
    case.note(max_tokens=MAX_TOKENS, n=len(cases), image_token_id=image_token_id)
    case.write()


if __name__ == "__main__":
    main()
