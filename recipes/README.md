# Phase 4 recipes

Inputs to `mtplx forge build`, one per variant, **named for what each one puts
at 4 bits** rather than for any tier a publisher sold it as. `REBUILD.md`
Phase 4 has the arithmetic and the measured results.

| recipe | what moves to 4 bits | 8-bit left | bf16 | GB/token | GPU busy | ppl | max KL |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline/BUILD_RECIPE.json` | -- the control | 260 | 60 | 2.739 | 10.40 ms | 6.3327 | -- |
| **`int4-head-linattn.json`** | `lm_head`, `linear_attn.in_proj_qkv`, `.in_proj_z`, `.out_proj`, at affine g32 | 169 | 60 | 2.076 (-24.2%) | 9.17 ms | 6.5022 | **0.048** |
| `nvfp4-head-linattn.json` | the same four families, at **nvfp4 g16** | 169 | 60 | 1.925 (-29.7%) | 8.55 ms | 6.6010 | **0.429** |
| `int4-except-gates.json` | everything but the 80 router and shared-expert gates | 80 | 60 | 1.858 (-32.2%) | 8.29 ms | 6.6422 | **0.323** |

**`int4-head-linattn` is still the one to keep.** Past it the distribution
cost per point of traffic is seventeen times higher -- 0.0020 nats a point up
to it, 0.0344 a point beyond -- for 4.3% more decode.

**And `nvfp4-head-linattn` is the one not to keep**, which is the new result.
It delivered its bytes exactly as predicted (-7.3% against
`int4-head-linattn`, -6.7% of GPU busy) and its quality did not: 0.048 ->
0.429 nats, a factor of 8.9. That leaves it **strictly dominated by
`int4-except-gates`** -- more bytes, more GPU time and a third more KL than a
variant this project had already turned down. The only column it wins is
perplexity, which is the column Phase 4 established cannot rank these.

The measured reason is in `REBUILD.md`, "Why the prediction missed": nvfp4's
error on the model's own weights is 10.4-13.1%, not the 9.2-9.5% the
synthetic study predicted, and `tools/weight_error.py` is the tool that says
so.

**With one caveat that is worth more than the verdict.** The root cause is
that E4M3's smallest normal is 2**-6 while this model's group scales sit near
2**-7.7, so 99.6% of them are subnormal and lose most of their precision --
and nvfp4 has an input for exactly that, `global_scale`, one fp32 a tensor,
which `mlx_lm.convert` does not pass. With it the same weights come out at
**9.4-9.5%** instead of 10.4-13.1%. So what was measured is the rung the
toolchain can forge, not the ceiling of the format, and whether a
global-scaled nvfp4 clears the knee is open. `REBUILD.md` prices what it
would take: a quantization pass outside `mlx_lm.convert`, the `_hgs` kernels
this port did not instantiate, and a loader path for the third tensor.

The kernels stay in the port -- they are verified, they cost nothing at rest,
and on an M5 the `_r_2` narrow-qmv path changes the speed side of this trade
even though it cannot change the quality side.

The artifacts are `~/.mtplx/models/ornith-1.5-35b-a3b-{int4-head-linattn,
nvfp4-head-linattn, int4-except-gates}`; the control is
`-attention8-bf16recurrence-vision-mtplx`, which is also the port's default.

**Not built, and why.** A third published sibling quantizes the recurrence
inputs `in_proj_a` / `in_proj_b` as well, which every recipe here keeps in
bf16 deliberately. It buys 0.2 percentage points of traffic over
`int4-except-gates` -- 1.850 against 1.855 GB a token. Near-zero prize against
real risk; priced in seconds, one build and one qualification run saved.

**The bf16 parent is still the row that is missing**, and its ppl and KL
cannot be measured on this host: 67.0 GiB of weights against a 51.8 GiB GPU
working set, so `mlx_lm.load` dies in the loader. What it *can* now give is
the weight error each variant pays against it, one module at a time --
`tools/weight_error.py`, which loads a tensor rather than a model. See
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

Then the two that do not need a model load:

    python3 tools/step_bytes.py ~/.mtplx/models/<name>       # bytes a decode step
    PYTHONPATH=../vendor/mlx-lm python3 tools/weight_error.py # error vs the parent

**A note on counting modules for an nvfp4 variant.** `check_rebuild.py` counts
*modules*, not tensors -- a module is quantized if it has more than one tensor
-- so 733 text modules and the 452/281 quantized/float split hold unchanged
even though every nvfp4 module is two tensors rather than three. That is the
check working as designed and not a coincidence; the tensor counts do move
(452 scales, 169 biases in `nvfp4-head-linattn`) and nothing asserts them.
