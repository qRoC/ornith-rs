"""Write the probe set as JSON, so the port can capture the same 64 contexts.

`Q4K.md`'s hidden cost: an artifact in a mode mlx has no quantization mode for
cannot be captured by `dump_probes.py`, because `mlx_lm.load` cannot read it.
The Rust port *can* run it, so the port becomes the second instrument -- but
only if it runs the same contexts, in the same order. `probe_set.py` stays the
single source of truth; this writes it out for `src/bin/probes.rs` to read.

    cd tools && python3 export_probes.py
"""

import json
import pathlib

from probe_set import PROBES

OUT = pathlib.Path(__file__).resolve().parent.parent / "refs" / "probe_set.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(
        [{"group": g, "id": p, "text": t} for g, p, t in PROBES],
        indent=1,
        ensure_ascii=False,
    )
)
print(f"wrote {OUT}  {len(PROBES)} probes")
