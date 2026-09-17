# Phase 4 variant captures

The numbers `tools/dump_artifact_baseline.py` produced for each variant, kept
here because `refs/` is out of git and these are only regenerable while the
28 GB artifacts they came from are still on disk.

| file | recipe | ppl | mean NLL |
|---|---|---:|---:|
| `int4-head-linattn.json` | `recipes/int4-head-linattn.json` | 6.5022 | 1.872134 |
| `int4-except-gates.json` | `recipes/int4-except-gates.json` | 6.6422 | 1.893443 |

Artifacts are `~/.mtplx/models/ornith-1.5-35b-a3b-<name>`.

The control is `baseline/baseline.json` at 6.3327 / 1.845725, and its
`logprobs.bin` is what the KL in `REBUILD.md` Phase 4 is measured against.
The variants' own `logprobs.bin` files stay in
`refs/artifact/{int4-head-linattn,int4-except-gates}/` --
7.9 MB each, and regenerable from the artifacts.
