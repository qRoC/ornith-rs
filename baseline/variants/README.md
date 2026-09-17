# Phase 4 variant captures

The numbers `tools/dump_artifact_baseline.py` produced for each variant, kept
here because `refs/` is out of git and these are only regenerable while the
28 GB artifacts they came from are still on disk.

| file | recipe | artifact | ppl | mean NLL |
|---|---|---|---|---|
| `b4.json` | `recipes/phase4-B.json` | `~/.mtplx/models/ornith-1.5-35b-a3b-b4` | 6.5022 | 1.872134 |
| `r8.json` | `recipes/phase4-R-balanced.json` | `~/.mtplx/models/ornith-1.5-35b-a3b-r8` | 6.6422 | 1.893443 |

The control is `baseline/baseline.json` at 6.3327 / 1.845725, and its
`logprobs.bin` is what the KL in `REBUILD.md` Phase 4 is measured against.
The variants' own `logprobs.bin` files stay in `refs/artifact/{b4,r8}/` --
7.9 MB each, and regenerable from the artifacts.
