# Phase 5 probe captures

What `tools/dump_probes.py` recorded for each artifact on the 64-context
probe set (`tools/probe_set.py`), kept here for the same reason
`baseline/variants/` is: `refs/` is out of git, and these are only
regenerable while the 28 GB artifacts they came from are still on disk.

`comparison.txt` is `tools/compare_probes.py`'s output over all five, and
`REBUILD.md` Phase 5 reads it.

| file | artifact | median KL | max KL |
|---|---|---:|---:|
| `new.json` | the control, `-attention8-bf16recurrence-vision-mtplx` | -- | -- |
| `int4-head-linattn.json` | `recipes/int4-head-linattn.json` | **0.0272** | 0.5029 |
| `fp8scale-twin.json` | `tools/forge_fp8scale_twin.py` | 0.0340 | 0.4890 |
| `int4-except-gates.json` | `recipes/int4-except-gates.json` | 0.0549 | 1.5938 |
| `nvfp4-head-linattn.json` | `recipes/nvfp4-head-linattn.json` | 0.0649 | 6.6391 |

Medians are over the 60 live probes, KL against the control.

What is **not** here is `probes.bin` -- 64 x 248320 float32, 61 MB an
artifact. Those stay in `refs/artifact/<name>/`, and `dump_probes.py`
rebuilds one in about six minutes from an artifact that still exists.

The JSON keeps each probe's text, token count, and the control-relative
things that do not need the distributions: entropy, top-1 mass and the top-1
token. That is enough to see *what the model answered* on every probe without
61 MB a variant -- it is how the `russian-decl` failure is legible here at
all: the control and two variants say ` Чел`, the other two say ` Ю`.
