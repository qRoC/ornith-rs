# Phase 4 recipes

Inputs to `mtplx forge build`, one per variant. `REBUILD.md` Phase 4 has the
arithmetic that ranks them; this is what each file is.

| file | 8-bit g64 | bf16 | 4-bit g32 | GB/token | ppl | max KL |
|---|---|---|---|---|---|---|
| `baseline/BUILD_RECIPE.json` | 260 | 60 | 192 | 2.737 | 6.3327 | control |
| **`phase4-B.json`** | 169 | 60 | 283 | 2.074 (-24.2%) | 6.5022 | **0.048** |
| `phase4-R-balanced.json` | 80 | 60 | 372 | 1.855 (-32.2%) | 6.6422 | **0.323** |

**B is the one to keep.** Past it the distribution cost per point of traffic
is seventeen times higher -- 0.0020 nats a point up to B, 0.0344 a point from
B to R -- for 4.3% more decode. `REBUILD.md` Phase 4 has the full table.

**The three published siblings.** The artifact this port was built on is one
of a matrix the publisher shipped -- their recipes carry
`adapter_id: qwen35-moe-ornith15-quant-matrix-v1` and
`profile: control-candidate`, so the matrix was deliberate:

- `...-attention8-bf16recurrence-vision-mtplx` -- "Flagship / Highest
  Quality", 260 modules at 8 bits. This is `baseline/BUILD_RECIPE.json`.
- `...-affine4-router8-bf16recurrence-vision-mtplx` -- "Balanced". Only the
  80 router and shared-expert gates stay at 8 bits; `in_proj_a` / `in_proj_b`
  stay bf16. Copied here as `phase4-R-balanced.json`.
- `...-affine4-router8-vision-mtplx` -- "Most Compact". The same 80 gates, and
  the recurrence inputs quantized too.

So the question Phase 4 exists to answer -- do these modules tolerate 4 bits
-- has a published answer from someone who went further than any rung the plan
had drawn, and independently landed on the plan's own rule that the router
gates never move. What is still missing is a *measurement*: their labels are
labels. `baseline/logprobs.bin` and the 6.3327 perplexity are the Flagship's,
on a fixed corpus, so the cost of their Balanced can be priced exactly rather
than trusted.

**Most Compact is not worth building.** It buys 0.2 percentage points of
traffic over Balanced -- 1.850 against 1.855 GB a token -- and pays for them
by quantizing the recurrence inputs the other two recipes deliberately keep in
bf16. Near-zero prize, real risk, the same shape as the argument for leaving
the router gates alone.

**Run order after any build:**

    python3 tools/align_rebuild.py ~/.mtplx/models/<name>
    python3 tools/check_rebuild.py ~/.mtplx/models/<name>
    cd tools && ORNITH_BASELINE_OUT=../refs/artifact/<name> \
      PYTHONPATH=../../vendor/mlx-lm python3 dump_artifact_baseline.py ~/.mtplx/models/<name>
    python3 tools/compare_baselines.py refs/artifact/<name>

`check_rebuild.py`'s RECIPE lines are *expected* to differ for a variant;
its INVARIANT lines are not.
