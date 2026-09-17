"""Check a rebuilt artifact against the baseline. `REBUILD.md` Phase 2 and 3.

    python3 tools/check_rebuild.py <path to rebuilt artifact>

Three classes of check, and the distinction is the point:

  INVARIANT   must hold for any rebuild, whatever the recipe says. These are
              properties of the parent and of the conversion, not of the
              quantization -- and the ones marked (blind) are ones the golden
              tests *cannot* catch, because MLX reads the same artifact and
              would make the same mistake.
  RECIPE      must hold when the recipe is unchanged (Phase 2, reproducing
              what we had). Expected to differ once `body_mode` or the
              overrides change (Phase 4).
  INFO        reported, never judged.

Exit code is 1 if any INVARIANT fails, else 0. RECIPE failures are reported
and left to the reader, because in Phase 4 they are the intended outcome.

**On byte-identity.** The first version of this script compared per-file
sha256 against `baseline/.mtplx-source.json` and expected a match. It cannot
match, for a reason that has nothing to do with the weights: the published
artifact says `mtplx_version: 2.9.1` in its own config, and this machine has
2.11.3 with a newer mlx-lm. That toolchain shards the body with
`make_shards(MAX_FILE_SIZE_GB = 5)` -- 5 files where 2.9.1 wrote 17 -- and
names the sidecars `mtp.safetensors` and `model-vision.safetensors` where
2.9.1 wrote `mtp/weights.safetensors` and `model-vision-00001-of-00001`.
Every tensor is in the index either way. So the file-level comparison is
INFO, the metadata files are checked by content instead, and the question
byte-identity was there to answer -- *are these the same numbers* -- is
answered by `baseline/logprobs.bin`, which does not care how the weights were
split across files. See `REBUILD.md` Phase 2.
"""

import hashlib
import json
import pathlib
import struct
import sys

BASE = pathlib.Path(__file__).resolve().parent.parent / "baseline"
PARENT = pathlib.Path.home() / ".mtplx/models/ornith-ai--Ornith-1.5-35B-A3B"

# The parent's index has 1811 tensors: 785 `mtp.*`, 333 vision, 693 the rest.
# Only two of those counts survive into an artifact unchanged, which is what
# the first version of this script got wrong -- it checked all six against the
# artifact and two of them can never hold there.
#
# What the parent's numbers become on the other side of the conversion.
#   693 parent tensors that are neither `mtp.*` nor vision, plus one more per
#   layer because mlx-lm's sanitize splits the fused `mlp.experts.gate_up_proj`
#   into `switch_mlp.gate_proj` and `switch_mlp.up_proj`: 693 + 40 = 733.
# A module is one to three tensors depending on whether the recipe quantized
# it, so the module count is invariant where a tensor count is not -- which is
# exactly why the old check failed here on a correct rebuild.
TEXT_MODULES = 733
VISION_TENSORS = 333
MTP_TENSORS = 785

# The recipe's own split of those 733, from `conversion_receipt.json`.
QUANTIZED_MODULES = 452
FLOAT_MODULES = TEXT_MODULES - QUANTIZED_MODULES   # 281: 60 excluded by the
                                                   # recipe, 221 never
                                                   # quantizable (norms,
                                                   # conv1d, A_log, dt_bias)

# Files the published artifact carried as *content* rather than as a record of
# its own build. `README.md`, `config.json`, `conversion_receipt.json`,
# `BUILD_RECIPE.json`, `mtplx_runtime.json` and the index are all build
# records and are left out on purpose.
PUBLISHED_CONTENT = [
    "chat_template.jinja", "configuration.json", "generation_config.json",
    "merges.txt", "preprocessor_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json",
    "vocab.json",
]

ok = True


def check(kind, name, passed, detail=""):
    global ok
    if kind == "INVARIANT" and not passed:
        ok = False
    mark = "ok  " if passed else ("FAIL" if kind == "INVARIANT" else "diff")
    print(f"  [{mark}] {kind:<9} {name}{(' -- ' + detail) if detail else ''}")


def info(name, detail):
    print(f"  [    ] {'INFO':<9} {name} -- {detail}")


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def module_of(tensor):
    for suffix in (".weight", ".scales", ".biases"):
        if tensor.endswith(suffix):
            return tensor[: -len(suffix)]
    return tensor


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    art = pathlib.Path(sys.argv[1]).expanduser()
    if not art.is_dir():
        sys.exit(f"not a directory: {art}")
    # Pointed at the bf16 parent instead of a rebuilt artifact, every check
    # below fails for the right reasons and reads like a disaster. The parent
    # has no quantization block, its `conv1d.weight` is still the torch
    # `[D, 1, K]` that conversion transposes, and its `rope_parameters` still
    # says `rope_type` where mlx writes `type`.
    cfg = json.loads((art / "config.json").read_text())
    if not cfg.get("quantization"):
        sys.exit(
            f"{art} has no quantization block, so it is the bf16 parent and not a\n"
            "rebuilt artifact. Nothing here applies to it: conv1d.weight is still\n"
            "[D, 1, K] until conversion transposes it, and rope_parameters says\n"
            "rope_type until mlx normalises it. Point this at the forge output."
        )
    print(f"rebuilt artifact: {art}\nbaseline:         {BASE}\n")

    index = json.loads((art / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    is_vision = lambda k: "visual" in k.lower() or "vision" in k.lower()

    # --- the parent's modules all survived the conversion ---
    text = [k for k in wm if not is_vision(k)]
    modules = {module_of(k) for k in text}
    check("INVARIANT", "text modules", len(modules) == TEXT_MODULES,
          f"{len(modules)} vs {TEXT_MODULES}")
    check("INVARIANT", "vision tensors",
          sum(1 for k in wm if is_vision(k)) == VISION_TENSORS,
          f"{sum(1 for k in wm if is_vision(k))} vs {VISION_TENSORS}")
    for key, want in (("A_log", 30), ("in_proj_a", 30), ("in_proj_b", 30)):
        got = sum(1 for k in wm if key in k)
        check("INVARIANT", f"tensors {key}", got == want, f"{got} vs {want}")

    # The MTP head is a sidecar and is not in the index; `mlx_lm_extra_tensors`
    # names the file, and both this port and MTPLX follow that key rather than
    # a fixed path, which is why the 2.9.1/2.11.3 rename is harmless.
    mtp_rel = (cfg.get("mlx_lm_extra_tensors") or {}).get("mtp_file", "mtp/weights.safetensors")
    mtp_path = art / mtp_rel
    if not mtp_path.exists():
        check("INVARIANT", "mtp sidecar", False, f"{mtp_rel} not found")
    else:
        got = sum(1 for k in safetensors_header(mtp_path) if k != "__metadata__")
        check("INVARIANT", "mtp tensors", got == MTP_TENSORS,
              f"{got} vs {MTP_TENSORS} in {mtp_rel}")

    # --- the ones the golden tests are blind to ---
    conv = next((k for k in wm if k.endswith("conv1d.weight")), None)
    if conv is None:
        check("INVARIANT", "conv1d.weight present (blind)", False)
    else:
        shape = safetensors_header(art / wm[conv])[conv]["shape"]
        # Already sanitised means [D, K, 1]; anything else and mlx-lm's
        # TextModel.sanitize adds 1.0 to the norm weights -- the "double
        # shift". MLX would do it too, so max_abs == 0.0 stays green.
        check("INVARIANT", "conv1d.weight is [D, K, 1] (blind)",
              len(shape) == 3 and shape[2] == 1, f"{conv} {shape}")

    # The MTP head's RMSNorm gains must be the *shifted* ones. The parent
    # stores all seven delta-encoded, exactly as it stores the trunk's, and
    # both get +1.0 on the way in -- the trunk's by `qwen3_5.TextModel.
    # sanitize` (whenever the checkpoint still has `mtp.*` weights or a
    # torch-layout conv1d), the sidecar's by forge's `shift_delta_mtp_norms`.
    # The port applies no shift of its own, so a sidecar left delta-encoded
    # gives a head that drafts nothing -- measured at 0 of 95 -- while every
    # golden test stays green, because MLX reads the same file.
    if not PARENT.is_dir():
        info("mtp norms (blind)", f"parent not at {PARENT}; check skipped")
    elif mtp_path.exists():
        pidx = json.loads((PARENT / "model.safetensors.index.json").read_text())["weight_map"]
        sh = safetensors_header(mtp_path)
        sh.pop("__metadata__", None)
        sbase = 8 + struct.unpack("<Q", mtp_path.open("rb").read(8))[0]
        unshifted = []
        with open(mtp_path, "rb") as side:
            for name, meta in sh.items():
                if len(meta["shape"]) != 1 or name not in pidx:
                    continue
                s0, e0 = meta["data_offsets"]
                side.seek(sbase + s0)
                got = side.read(e0 - s0)
                shard = PARENT / pidx[name]
                ph = safetensors_header(shard)
                p0, p1 = ph[name]["data_offsets"]
                with open(shard, "rb") as src:
                    src.seek(8 + struct.unpack("<Q", shard.open("rb").read(8))[0] + p0)
                    if src.read(p1 - p0) == got:
                        unshifted.append(name)
        check("INVARIANT", "mtp norm gains are shifted (blind)", not unshifted,
              "" if not unshifted else
              f"{len(unshifted)} of 7 still equal the parent's delta encoding")

    # The vision patch embedding must be in mlx's Conv3d layout,
    # (out, kD, kH, kW, in). 2.9.1 transposed it when it grafted the tower in;
    # 2.11.3 copies the parent's torch layout verbatim. MTPLX transposes at
    # load either way, but this port reads the tensor as it lies --
    # `VisionTower::load` asserts, so it fails closed rather than convolving
    # the wrong axes. `tools/align_rebuild.py` restores it.
    conv = next((k for k in wm if k.endswith("patch_embed.proj.weight")), None)
    if conv is None:
        info("vision patch embed", "no patch_embed.proj.weight in the index")
    else:
        vshape = safetensors_header(art / wm[conv])[conv]["shape"]
        want = int((cfg.get("vision_config") or {}).get("in_channels", 3))
        check("INVARIANT", "patch_embed is in mlx Conv3d layout",
              len(vshape) == 5 and vshape[-1] == want, f"{vshape}")

    tc = cfg.get("text_config", cfg)
    rope = (tc.get("rope_parameters") or {}).get("type")
    check("INVARIANT", "rope_parameters.type is default (blind)",
          rope == "default", str(rope))

    # `TextConfig::load` reads this and falls back to 0, and `MtpHead::load`
    # then asserts. MTPLX 2.11.3 does not write it -- 2.9.1 did -- so a
    # rebuild that skips tools/align_rebuild.py loses the MTP head here.
    mtp_layers = cfg.get("num_nextn_predict_layers", tc.get("num_nextn_predict_layers"))
    check("INVARIANT", "num_nextn_predict_layers is 1", mtp_layers == 1, str(mtp_layers))

    # The artifact's own pre-tokenizer. transformers 5.x re-serialises this
    # file through its *older* Qwen2 pattern, which has no `\p{M}` in either
    # character class -- and then `dump_tokenizer.py` and the port both read
    # the re-serialised file and agree on it. See `HANDOFF.md`, "The tokenizer
    # has two behaviours".
    tok = art / "tokenizer.json"
    if not tok.exists():
        check("INVARIANT", "tokenizer.json present (blind)", False)
    else:
        tj = json.loads(tok.read_text())
        pats = [p.get("pattern", {}).get("Regex", "")
                for p in (tj.get("pre_tokenizer") or {}).get("pretokenizers", [])
                if p.get("type") == "Split"]
        pat = pats[0] if pats else ""
        keeps = r"\p{M}" in pat
        check("INVARIANT", r"pre-tokenizer keeps \p{M} (blind)", keeps,
              "" if keeps else ("the transformers pattern was written into the "
                                "artifact" if pat else "no Split pre-tokenizer"))
        bl = next((p for p in (tj.get("pre_tokenizer") or {}).get("pretokenizers", [])
                   if p.get("type") == "ByteLevel"), {})
        check("INVARIANT", "ByteLevel trim_offsets is false (blind)",
              bl.get("trim_offsets") is False,
              "" if bl.get("trim_offsets") is False else str(bl.get("trim_offsets")))
        merges = tj.get("model", {}).get("merges") or []
        strings = not merges or isinstance(merges[0], str)
        check("INVARIANT", "merges are strings, not pairs (blind)", strings,
              "" if strings else "tokenizers >= 0.22 rewrote them as [\"a\", \"b\"]")

    # --- the recipe produced what it produced last time ---
    base_cfg = json.loads((BASE / "config.json").read_text())

    def split(c):
        q = c.get("quantization") or {}
        out = {}
        for v in q.values():
            if isinstance(v, dict):
                k = (v.get("bits"), v.get("group_size"), v.get("mode"))
                out[k] = out.get(k, 0) + 1
        return q, out

    new_q, new_split = split(cfg)
    old_q, old_split = split(base_cfg)
    check("RECIPE", "quantization default",
          (new_q.get("bits"), new_q.get("group_size"), new_q.get("mode"))
          == (old_q.get("bits"), old_q.get("group_size"), old_q.get("mode")),
          f"{new_q.get('bits')}b/g{new_q.get('group_size')}/{new_q.get('mode')} vs "
          f"{old_q.get('bits')}b/g{old_q.get('group_size')}/{old_q.get('mode')}")
    check("RECIPE", "per-module overrides", new_split == old_split,
          f"{dict(new_split)} vs {dict(old_split)}")
    new_names = {k for k, v in (cfg.get("quantization") or {}).items() if isinstance(v, dict)}
    old_names = {k for k, v in (base_cfg.get("quantization") or {}).items() if isinstance(v, dict)}
    check("RECIPE", "override module names", new_names == old_names,
          f"{len(new_names & old_names)} shared, {len(new_names ^ old_names)} different")

    counts = {}
    for k in text:
        counts[module_of(k)] = counts.get(module_of(k), 0) + 1
    quantized = sum(1 for n in counts.values() if n > 1)
    check("RECIPE", "quantized / float modules",
          (quantized, len(counts) - quantized) == (QUANTIZED_MODULES, FLOAT_MODULES),
          f"{quantized} / {len(counts) - quantized} vs "
          f"{QUANTIZED_MODULES} / {FLOAT_MODULES}")

    # --- the published files that are content, by sha256 ---
    sums = {}
    for line in (BASE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest
    same = [n for n in PUBLISHED_CONTENT
            if (art / n).exists() and sha256(art / n) == sums.get(n)]
    check("RECIPE", "published content files",
          len(same) == len(PUBLISHED_CONTENT),
          f"{len(same)} of {len(PUBLISHED_CONTENT)} byte-identical"
          + ("" if len(same) == len(PUBLISHED_CONTENT)
             else "; run tools/align_rebuild.py"))

    # --- file layout, reported and not judged ---
    marker = json.loads((BASE / ".mtplx-source.json").read_text())
    published = [n for n in marker.get("files", {}) if n.endswith(".safetensors")]
    here = sorted(p.name for p in art.glob("*.safetensors"))
    info("shard layout", f"{len(here)} files here, {len(published)} published "
                         f"(mlx-lm packs to MAX_FILE_SIZE_GB = 5; 2.9.1 wrote ~2 GB)")
    info("body bytes", f"{sum(p.stat().st_size for p in art.glob('model-0*.safetensors'))/1e9:.3f} GB")
    info("equivalence", "byte-identity is unreachable on this toolchain; "
                        "baseline/logprobs.bin is the check that survives")

    print(f"\n{'PASS' if ok else 'FAIL'}: invariants "
          f"{'hold' if ok else 'BROKEN -- stop here'}")
    print("RECIPE differences are expected once the recipe changes; in Phase 2 "
          "they are not.")
    return 0 if ok else 1


sys.exit(main())
