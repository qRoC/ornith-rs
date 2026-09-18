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
| *(no recipe)* `fp8scale-twin` | the four families at **affine int4 g32 with E4M3 scale+bias** | 169 | 60 | 1.925 (-29.7%) | -- | 6.5035 | **0.443** |
| *(no recipe)* **`gptq-head-linattn`** | `int4-head-linattn` exactly, re-rounded by GPTQ | 169 | 60 | 2.076 (identical) | 9.17 ms | 6.4238 | **0.0417** |

**`gptq-head-linattn` costs nothing to take, and that is the strongest thing
that can be said for it.** Same bits, same coverage, same three tensors, same
shards, same kernels, same dispatch profile count for count -- only which
integer each weight rounds to differs, and `full_forward_matches_mlx` passes at
`max_abs == 0.0`. Forged by `tools/forge_gptq.py` from `refs/gptq/`
(`tools/dump_gptq_stats.py`); no `align_rebuild.py`, the base was aligned.

**But the margin depends on what you measure from, and Phase 8 measured from
the parent.** The KL columns in the table above are against `A`, which is
itself quantized:

| | vs `A` | vs the bf16 parent |
|---|---|---|
| median KL | 0.0187 vs 0.0285 | 0.0303 vs 0.0427 |
| p90 | **0.0985** vs 0.2003 | 0.2510 vs **0.2076** |
| max | 0.9229 vs **0.5029** | 1.0122 vs **0.7394** |
| sign test | 38/58, p = 0.025 | **31/57, p = 0.60** |

Against the parent GPTQ is still ahead on the median and is still the least
damaged of the forged variants, but the two are **not separable probe by
probe**, and GPTQ's tail is the worse of the two. `REBUILD.md` Phase 8 has the
mechanism: these Hessians were captured on `A`, so GPTQ reconstructs `A`'s
layer outputs and was then scored on its distance from `A`. Re-capturing them
from the parent is the open follow-up.

The last row has no recipe because it is not an mlx mode and `mtplx forge`
cannot build it. `tools/forge_fp8scale_twin.py` builds a stand-in instead --
an E4M3 value times a power of two is exactly representable in bfloat16, so
the candidate format can be written out in the ordinary affine layout and
run unmodified. Its GB/token is the arithmetic of the format it stands for
(4.50 bits), not of the file on disk (5.00).

**`int4-head-linattn` is the knee of the byte/quality trade, and Phase 7
supersedes it only in rounding, not in coverage** -- `gptq-head-linattn` is
that same recipe re-rounded. Past it the distribution
cost per point of traffic is seventeen times higher -- 0.0020 nats a point up
to it, 0.0344 a point beyond -- for 4.3% more decode.

> **REOPENED 2026-09-18 by Phase 14.** A third 4.50-bit route -- a Q4_K-shaped
> grid, 6-bit integer scale and min against power-of-two super-scales -- lands
> at median KL **0.0403 against the base's 0.0404**, i.e. the same distribution
> for 7.3% fewer bytes, and strictly dominates `int4-head-linattn` on the pair.
> The sentence below was true of the two routes tested and is not true of the
> space. Realising the bytes still needs a kernel mlx has no mode for.

**Both 4.50-bit routes lose, and lose the same way.** `nvfp4-head-linattn`
and the fp8-scale twin land at 0.429 and 0.443 nats -- nine times
`int4-head-linattn` -- and both are dominated by `int4-except-gates`, which
reads fewer bytes at a lower KL. They get there from very different weight
errors (+34% and +3% relative), which is the other result: **how far the
weights moved does not predict where the distribution ended up**, so
`tools/weight_error.py` explains a format and cannot rank one.

**And the whole ranking rests on one context of eight** -- `Дважды два
равно`, the only Russian prompt. On the other seven all four variants sit
within 0.1 nats of the control. `REBUILD.md`, "Seven of the eight contexts
cannot tell these variants apart", has the table; the short version is that
the verdict holds and the confidence in it does not, and a wider probe set
comes before the next format.

**`nvfp4-head-linattn` is the one not to keep**, which was the first result.
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

**Four artifacts are on disk and the rest were deleted, deliberately.** What a
build produces is not the artifact -- it is `refs/artifact/<name>/probes.bin`,
[64, 248320] log-probabilities in 64 MB, which is what `compare_probes.py`
reads and what every verdict in this file rests on. Six measured-and-rejected
variants (`nvfp4-head-linattn`, `int4-except-gates`, `fp8scale-twin`,
`awq-linattn`, `awq-ml`, `gptqp-head-linattn`) were 149 GB of artifact whose
measurements are 384 MB of capture; the captures are kept and the artifacts are
gone. Any of them re-forges from its recipe or its `tools/forge_*.py` in about
two minutes.

What is kept, and why:

    ornith-ai--Ornith-1.5-35B-A3B                              67 G  the bf16 parent; a 70 GB re-download
    ornith-1.5-35b-a3b-attention8-bf16recurrence-vision-mtplx   27 G  control A, and the port's default model_dir
    ornith-1.5-35b-a3b-int4-head-linattn                        26 G  the RTN base, and what every forge builds from
    ornith-1.5-35b-a3b-gptq-head-linattn                        26 G  the current best

**Not built, and why.** A third published sibling quantizes the recurrence
inputs `in_proj_a` / `in_proj_b` as well, which every recipe here keeps in
bf16 deliberately. It buys 0.2 percentage points of traffic over
`int4-except-gates` -- 1.850 against 1.855 GB a token. Near-zero prize against
real risk; priced in seconds, one build and one qualification run saved.

**The bf16 parent's KL row is filled as of Phase 8; its perplexity is not.**
`mlx_lm.load` does die -- 67.0 GiB against a 51.8 GiB working set -- but that
is about residency, not about the forward.
`tools/dump_parent_probes.py` streams the parent one layer at a time (measured
peak **1.73 GiB**, whole 64-probe capture **24 s**) and writes
`refs/artifact/parent/probes.bin`, so `compare_probes.py` now scores every
variant against the parent instead of against `A`. Perplexity is the same pass
and is not done. `tools/weight_error.py` still gives the per-module weight
error against the parent without loading a model.

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
