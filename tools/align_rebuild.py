"""Put a 2.11.3 forge output back into the shape the published artifact had.

    python3 tools/align_rebuild.py <rebuilt artifact> [<parent>]

`REBUILD.md` Phase 2. The published artifact's `config.json` says
`mtplx_version: 2.9.1`; this machine has 2.11.3 and an mlx-lm newer than
whatever 2.9.1 pulled in. The *weights* come out the same -- the quantization
block matches module for module -- but the metadata around them does not, and
two of the differences change what the port reads rather than how the files
are named: the tokenizer, and `num_nextn_predict_layers`. This script fixes exactly those, is idempotent, and prints what it
did; run it after every forge build, including each Phase 4 variant, so no
variant is compared against the others through a different tokenizer.

What it restores, and why each one is not cosmetic:

  tokenizer.json, tokenizer_config.json, vocab.json, merges.txt
      The published artifact carried the parent's four files byte for byte
      (sha256 checked against `baseline/SHA256SUMS`). mlx-lm's `save()` calls
      `tokenizer.save_pretrained`, so transformers 5.14.1 re-serialises them
      instead, and the round trip is *not* faithful:
        - the Split pattern loses `\\p{M}` from both character classes, which
          is precisely the transformers-vs-tokenizer.json split `HANDOFF.md`
          documents -- the artifact would stop being able to express the
          `Dialect::TokenizerJson` behaviour at all, because its own pattern
          would be gone;
        - ByteLevel flips `trim_offsets` false -> true, and the decoder flips
          `add_prefix_space` and `use_regex` false -> true, which changes
          detokenisation;
        - `merges` is rewritten from "a b" to ["a", "b"];
        - `added_tokens` goes from 26 to 33, folding in the 7 that only
          `tokenizer_config.json` carried;
        - `vocab.json` and `merges.txt` are not written at all, and those two
          are what `transformers`' `Qwen2Tokenizer` rebuilds itself from --
          the default dialect of this port and of `mlx_lm.generate`;
        - `tokenizer_config.json` shrinks 16718 -> 1161 bytes, losing
          `added_tokens_decoder`, which `src/tokenizer/mod.rs` reads.
      None of this would be caught by a golden test: `dump_tokenizer.py` reads
      the same file the port does, so both would agree on the wrong pattern.

  configuration.json
      Copied from the parent by 2.9.1, not written by 2.11.3. Inert here, but
      it is one of the published files and costs nothing.

  config.json: num_nextn_predict_layers, mtp_num_hidden_layers
      2.9.1's `_patch_config_for_mtp` wrote them; 2.11.3's does not.
      `TextConfig::load` reads `num_nextn_predict_layers` and falls back to
      **0**, which silently turns the MTP head off in the port rather than
      failing. This is the one that would have been hardest to notice.

  config.json: eos_token_id
      mlx-lm's convert widens the parent's scalar 248046 to
      [248046, 248044] -- 248044 is the pad token. The port reads eos from
      `generation_config.json`, so this changes nothing here, but the
      published artifact had the scalar and a variant should not differ from
      it for a reason unrelated to the recipe.

  config.json: mlx_lm_extra_tensors.mtp_tensor_count, vision_config.dtype
      Recorded by 2.9.1, dropped by 2.11.3. Documentation only.

  the vision tower's patch_embed.proj.weight
      The only difference here that is a *weight layout*. mlx's Conv3d wants
      `(out, kD, kH, kW, in)`; a torch checkpoint stores
      `(out, in, kD, kH, kW)`. 2.9.1 transposed it when it grafted the tower
      in, so the published artifact held `[1152, 2, 16, 16, 3]`; 2.11.3's
      `graft_vision_tower` copies the parent's `[1152, 3, 2, 16, 16]`
      verbatim. MTPLX itself does not care -- `_conv_weight_in_mlx_layout`
      transposes at load -- but this port reads the tensor as it lies and
      `VisionTower::load` asserts on the shape, so it fails closed rather than
      quietly convolving the wrong axes. `tools/dump_vision_ops.py`, which
      reads the raw tensor too, is where it surfaced.

**The MTP sidecar's seven RMSNorm gains are deliberately left alone**, and
this is worth writing down because it looks wrong and is not. Forge adds 1.0
to all seven on the way out (`shift_delta_mtp_norms`), which reads as a double
shift if you compare the result with the *parent's* norms: the parent's MTP
gains sit right inside the distribution of the parent's trunk gains, so the
head looks like it was already absolute. It was not. Both sides of the parent
are delta-encoded, and the trunk's get the same +1.0 during conversion from
`qwen3_5.TextModel.sanitize`, which applies it whenever the checkpoint still
carries `mtp.*` weights or a torch-layout `conv1d` -- both true of a parent,
neither true of an artifact. Two sets that are both delta agree with each
other perfectly and prove nothing.

Restoring the parent's seven tensors byte for byte was tried, and the head
then drafted **0 of 95** tokens correctly. `sanitize`'s own `norm_keys` list
is the reference for which trunk norms move, and forge's suffix lists are the
reference for the sidecar. `check_rebuild.py` asserts the gains are *not* the
parent's.

Deliberately NOT changed, because they are naming and not content:

  mtp.safetensors (2.9.1: mtp/weights.safetensors)
      `config.json`'s `mlx_lm_extra_tensors.mtp_file` names it and both the
      port and MTPLX follow that key, so the file stays where forge put it.
  model-vision.safetensors (2.9.1: model-vision-00001-of-00001.safetensors)
      Resolved through `model.safetensors.index.json` the same way.
  5 shards instead of 17
      mlx-lm's `make_shards` packs to `MAX_FILE_SIZE_GB = 5`. Nothing reads a
      shard by name; the index maps every tensor. This is why a byte-identical
      rebuild is not reachable on this toolchain -- see `REBUILD.md`.
"""

import hashlib
import json
import pathlib
import shutil
import sys

FROM_PARENT = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "configuration.json",
]
DEFAULT_PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"
BASE = pathlib.Path(__file__).resolve().parent.parent / "baseline"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def published_sums():
    """sha256 of every file of the artifact as it was staged for release."""
    sums = {}
    path = BASE / "SHA256SUMS"
    if path.exists():
        for line in path.read_text().splitlines():
            digest, name = line.split()
            sums[name] = digest
    return sums


def fix_vision_conv_layout(art, cfg):
    """Transpose patch_embed.proj.weight into mlx's Conv3d layout.

    The same test MTPLX applies at load time (`_conv_weight_in_mlx_layout`):
    a 5-D weight whose last axis is the channel count is already mlx's.
    """
    import mlx.core as mx

    vision = None
    index = art / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        for k, f in wm.items():
            if k.endswith("patch_embed.proj.weight"):
                vision, key = art / f, k
                break
    if vision is None or not vision.exists():
        print("  [skip] vision tower -- no patch_embed.proj.weight in the index")
        return 0

    weights = dict(mx.load(str(vision)))
    w = weights[key]
    in_channels = int((cfg.get("vision_config") or {}).get("in_channels", 3))
    if w.ndim != 5:
        print(f"  [skip] {vision.name} -- {key} is {w.ndim}-D, not a Conv3d weight")
        return 0
    if w.shape[-1] == in_channels:
        print(f"  [ok  ] {vision.name} -- patch_embed is already {list(w.shape)}")
        return 0
    if w.shape[1] != in_channels:
        raise SystemExit(f"{key} is {list(w.shape)}, neither mlx nor torch layout")

    weights[key] = w.transpose(0, 2, 3, 4, 1)
    # `mx.load` is lazy and the arrays stay bound to the file, so saving over
    # that same path truncates it out from under the reads still to come --
    # which empties the sidecar. Materialise first, then write beside it and
    # rename.
    mx.eval(*weights.values())
    # The name has to keep the .safetensors extension: mx.save_safetensors
    # appends one when it is missing, and the rename would then find nothing.
    tmp = vision.with_name(vision.stem + ".new.safetensors")
    mx.save_safetensors(str(tmp), weights, metadata={"format": "mlx"})
    tmp.replace(vision)
    print(f"  [edit] {vision.name} -- patch_embed {list(w.shape)} -> "
          f"{list(weights[key].shape)}, mlx Conv3d layout")
    return 1


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    art = pathlib.Path(sys.argv[1]).expanduser()
    parent = pathlib.Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else DEFAULT_PARENT
    if not art.is_dir():
        sys.exit(f"not a directory: {art}")
    if not parent.is_dir():
        sys.exit(f"parent not found: {parent}")

    sums = published_sums()
    changed = 0

    for name in FROM_PARENT:
        src, dst = parent / name, art / name
        if not src.exists():
            print(f"  [skip] {name} -- not in the parent")
            continue
        if dst.exists() and sha256(dst) == sha256(src):
            print(f"  [ok  ] {name} -- already the parent's")
            continue
        shutil.copy2(src, dst)
        changed += 1
        want = sums.get(name)
        mark = "" if want is None else (
            " == published" if sha256(dst) == want else " -- DIFFERS from published"
        )
        print(f"  [copy] {name} <- parent{mark}")

    cfg_path = art / "config.json"
    cfg = json.loads(cfg_path.read_text())
    before = json.dumps(cfg, sort_keys=True)

    cfg.setdefault("num_nextn_predict_layers", 1)
    cfg.setdefault("mtp_num_hidden_layers", 1)
    if isinstance(cfg.get("eos_token_id"), list):
        # The parent's scalar, which is also what the published artifact had.
        parent_cfg = json.loads((parent / "config.json").read_text())
        if "eos_token_id" in parent_cfg:
            cfg["eos_token_id"] = parent_cfg["eos_token_id"]
    extra = cfg.get("mlx_lm_extra_tensors")
    if isinstance(extra, dict):
        extra.setdefault("mtp_tensor_count", 785)
    vision = cfg.get("vision_config")
    if isinstance(vision, dict):
        vision.setdefault("dtype", "bfloat16")

    if json.dumps(cfg, sort_keys=True) != before:
        cfg_path.write_text(json.dumps(cfg, indent=4))
        changed += 1
        print("  [edit] config.json -- num_nextn_predict_layers, mtp_num_hidden_layers, "
              "eos_token_id, mtp_tensor_count, vision dtype")
    else:
        print("  [ok  ] config.json -- already aligned")

    changed += fix_vision_conv_layout(art, cfg)

    print(f"\n{changed} file(s) changed. Re-run tools/check_rebuild.py next.")
    return 0


sys.exit(main())
