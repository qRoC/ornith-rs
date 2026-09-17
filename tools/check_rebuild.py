"""Check a rebuilt artifact against the baseline. `REBUILD.md` Phase 2 and 3.

    python3 tools/check_rebuild.py <path to rebuilt artifact>

Three classes of check, and the distinction is the point:

  INVARIANT   must hold for any rebuild, whatever the recipe says. These are
              properties of the parent and of the conversion, not of the
              quantization -- and the two marked (blind) are ones the golden
              tests *cannot* catch, because MLX reads the same artifact and
              would make the same mistake.
  RECIPE      must hold when the recipe is unchanged (Phase 2, reproducing
              what we had). Expected to differ once `body_mode` or the
              overrides change (Phase 4).
  INFO        reported, never judged.

Exit code is 1 if any INVARIANT fails, else 0. RECIPE failures are reported
and left to the reader, because in Phase 4 they are the intended outcome.
"""

import hashlib
import json
import pathlib
import struct
import sys

BASE = pathlib.Path(__file__).resolve().parent.parent / "baseline"

# From BUILD_RECIPE.json's source_signature, verified against the parent's
# own index before any download: these come from the parent and a rebuild
# that loses them has lost part of the model.
INVENTORY = {"total": 1811, "mtp": 785, "vision": 333, "A_log": 30,
             "in_proj_a": 30, "in_proj_b": 30}

ok = True


def check(kind, name, passed, detail=""):
    global ok
    if kind == "INVARIANT" and not passed:
        ok = False
    mark = "ok  " if passed else ("FAIL" if kind == "INVARIANT" else "diff")
    print(f"  [{mark}] {kind:<9} {name}{(' -- ' + detail) if detail else ''}")


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    art = pathlib.Path(sys.argv[1]).expanduser()
    if not art.is_dir():
        sys.exit(f"not a directory: {art}")
    print(f"rebuilt artifact: {art}\nbaseline:         {BASE}\n")

    # --- the parent's tensors survived the conversion ---
    index = json.loads((art / "model.safetensors.index.json").read_text())
    wm = index["weight_map"]
    got = {
        "total": len(wm),
        "mtp": sum(1 for k in wm if "mtp" in k.lower()),
        "vision": sum(1 for k in wm if "visual" in k.lower() or "vision" in k.lower()),
        "A_log": sum(1 for k in wm if "A_log" in k),
        "in_proj_a": sum(1 for k in wm if "in_proj_a" in k),
        "in_proj_b": sum(1 for k in wm if "in_proj_b" in k),
    }
    for k, want in INVENTORY.items():
        check("INVARIANT", f"tensors {k}", got[k] == want, f"{got[k]} vs {want}")

    # --- the two the golden tests are blind to ---
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

    cfg = json.loads((art / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    rope = (tc.get("rope_parameters") or {}).get("type")
    check("INVARIANT", "rope_parameters.type is default (blind)",
          rope == "default", str(rope))

    # --- the recipe produced what it produced last time ---
    base_cfg = json.loads((BASE / "config.json").read_text())
    for label, c in (("new", cfg), ("baseline", base_cfg)):
        q = c.get("quantization") or {}
        mods = {k: v for k, v in q.items() if isinstance(v, dict)}
        split = {}
        for v in mods.values():
            split[(v.get("bits"), v.get("group_size"), v.get("mode"))] = (
                split.get((v.get("bits"), v.get("group_size"), v.get("mode")), 0) + 1
            )
        if label == "new":
            new_q, new_split, new_mods = q, split, mods
        else:
            old_q, old_split, old_mods = q, split, mods
    check("RECIPE", "quantization default",
          (new_q.get("bits"), new_q.get("group_size"), new_q.get("mode"))
          == (old_q.get("bits"), old_q.get("group_size"), old_q.get("mode")),
          f"{new_q.get('bits')}b/g{new_q.get('group_size')}/{new_q.get('mode')} vs "
          f"{old_q.get('bits')}b/g{old_q.get('group_size')}/{old_q.get('mode')}")
    check("RECIPE", "per-module overrides", new_split == old_split,
          f"{dict(new_split)} vs {dict(old_split)}")

    # --- byte-identical? ---
    marker = json.loads((BASE / ".mtplx-source.json").read_text())
    files = {k: v for k, v in marker.get("files", {}).items() if "sha256" in v}
    same = missing = differ = 0
    for name, meta in sorted(files.items()):
        f = art / name
        if not f.exists():
            missing += 1
            continue
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                h.update(block)
        if h.hexdigest() == meta["sha256"]:
            same += 1
        else:
            differ += 1
    check("RECIPE", "sha256 of shards", differ == 0 and missing == 0,
          f"{same} identical, {differ} differ, {missing} missing, of {len(files)}")

    print(f"\n{'PASS' if ok else 'FAIL'}: invariants "
          f"{'hold' if ok else 'BROKEN -- stop here'}")
    print("RECIPE differences are expected once the recipe changes; in Phase 2 "
          "they are not.")
    return 0 if ok else 1


sys.exit(main())
