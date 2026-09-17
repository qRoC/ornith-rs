# Phase 4 recipes

Inputs to `mtplx forge build`, one per variant, **named for what each one puts
at 4 bits** rather than for any tier a publisher sold it as. `REBUILD.md`
Phase 4 has the arithmetic and the measured results.

| recipe | moves to 4-bit g32 | 8-bit left | bf16 | GB/token | ppl | max KL |
|---|---|---:|---:|---:|---:|---:|
| `baseline/BUILD_RECIPE.json` | -- the control | 260 | 60 | 2.737 | 6.3327 | -- |
| **`int4-head-linattn.json`** | `lm_head`, `linear_attn.in_proj_qkv`, `.in_proj_z`, `.out_proj` | 169 | 60 | 2.074 (-24.2%) | 6.5022 | **0.048** |
| `int4-except-gates.json` | everything but the 80 router and shared-expert gates | 80 | 60 | 1.855 (-32.2%) | 6.6422 | **0.323** |

**`int4-head-linattn` is the one to keep.** Past it the distribution cost per
point of traffic is seventeen times higher -- 0.0020 nats a point up to it,
0.0344 a point beyond -- for 4.3% more decode.

The artifacts are `~/.mtplx/models/ornith-1.5-35b-a3b-int4-head-linattn` and
`-int4-except-gates`; the control is
`-attention8-bf16recurrence-vision-mtplx`, which is also the port's default.

**Not built, and why.** A third published sibling quantizes the recurrence
inputs `in_proj_a` / `in_proj_b` as well, which the other two recipes keep in
bf16 deliberately. It buys 0.2 percentage points of traffic over
`int4-except-gates` -- 1.850 against 1.855 GB a token. Near-zero prize against
real risk; priced in seconds, one build and one qualification run saved.

**The bf16 parent is the row that is missing**, and it cannot be measured on
this host: 67.0 GiB of weights against a 51.8 GiB GPU working set. See
`REBUILD.md` Phase 4.

**Run order after any build:**

    python3 tools/align_rebuild.py ~/.mtplx/models/<name>
    python3 tools/check_rebuild.py ~/.mtplx/models/<name>
    cd tools && ORNITH_BASELINE_OUT=../refs/artifact/<short> \
      PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py ~/.mtplx/models/<name>
    python3 tools/compare_baselines.py refs/artifact/<short>

`check_rebuild.py`'s RECIPE lines are *expected* to differ for a variant; its
INVARIANT lines are not. `align_rebuild.py` is not optional -- it restores the
tokenizer and transposes the vision patch embedding, and without it the
variants are not comparable to each other.
