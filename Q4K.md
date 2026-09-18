# A k-quant kernel for this port: the design, and the one decision that changes it

`REBUILD.md` Phase 14 measured a Q4_K-shaped grid at **the same median KL as
the 5.00-bit base for 7.3% fewer bytes** -- the first 4.50-bit format here that
does not lose. It was measured as a *stand-in*: on disk it is still 5.00-bit
affine, and the 4.50 is arithmetic. Realising it needs a kernel.

This file is what that kernel should be, decided before any of it is written.

---

## The decision: build **real Q4_K**, not the twin's variant

The twin carries two constraints that exist only so it could be measured
without a kernel:

- **power-of-two super-scales**, because `2**k * s_i` with a 6-bit `s_i` is
  exactly representable in bfloat16 and therefore writes into the ordinary
  affine layout;
- **magnitude-plus-sign 6-bit fields**, because `aq.quantize`'s scale carries a
  sign.

Neither is a property of the format. Both were scaffolding. **With a kernel,
drop them** -- and doing so buys something the twin cannot have:

> **llama.cpp becomes the oracle.**

That is the whole argument for this file. mlx has four quantization modes and
none of them is a k-quant, so a Q4_K kernel written here has **no MLX
reference**, and `max_abs == 0.0` against MLX -- the check that has caught every
bug in this port -- simply does not exist for it. Writing the mode into
vendored mlx as well does not fix that; it makes the port agree with code from
the same hand.

**And it is not going to exist.** ml-explore/mlx#1934, "Q4_K_M support?",
closed by the lead maintainer an hour after it was opened, 2025-03-06:

> "We don't quite have exactly Q4_K_M and **aren't planning to add exactly
> that**. But we do support mixed quants with a similar strategy as Q4_K_M
> (e.g. using higher precision for the same weights). You can try e.g. the
> mixed 3_6 ... Another option is to use 4-bit with a smaller group size
> `--q-group-size 32` or to use 6-bit."

**Read that precisely, because it answers a different question.** Q4_K_M is a
*recipe* -- which bit width each tensor role gets. Q4_K is a *grid* -- integer
sub-block scales against a super-scale. The alternatives offered are all the
former: `mixed_quant_predicate_builder` in `mlx_lm/convert.py` sets
`mode = "affine"` and assigns low/high bits per module by layer position and by
`down_proj`. That is the coverage axis, which Phases 4 and 5 mapped here and put
a knee on, and it is **orthogonal to what Phase 14 measured** -- the twin used
uniform 4 bits at the base's own coverage, so every point of its gain came from
the grid and none from bit allocation.

So the position is settled and the grid question was never put. Treat "mlx will
not have a k-quant grid" as sourced rather than assumed.

`ggml` is a mature, independent, widely-deployed implementation of exactly this
format. If the port's kernel reproduces `dequantize_row_q4_K` **bit for bit** on
bytes produced by `llama-quantize`, the contract is intact in the form that
matters: two implementations that did not learn it from each other agree
exactly. The twin's variant forfeits that, because ggml cannot read it.

**MEASURED 2026-09-18, and it beats the prediction.** `vendor/ggml` built,
`quantize_q4_K` called through ctypes on the parent's own weights, read back
with `tools/q4k.py`:

    module         real ggml Q4_K   affine g32 bf16   the Phase 14 twin
                       4.50 bits        5.00 bits        (stand-in)
    in_proj_qkv          7.144%           8.111%           7.86%
    out_proj             7.547%           8.523%           8.30%
    lm_head              7.121%           8.087%           7.86%

**~12% better relative weight error than the 5.00-bit body, at half a bit
less**, and 0.7 points better than the stand-in. Two causes: fp16 super-scales
are finer than powers of two, and `quantize_q4_K` runs a search rather than
plain min/max -- worth 0.77 points on synthetic draws on its own.

The comparison that stings: **IDEAS #1**, the best 5.00-bit scheme in the table
and priced out in Phase 13 for having no oracle, measures 7.01-7.24%. Real
Q4_K very nearly matches it **at 4.50 bits**, and has one.

So Phase 14's `0.0396` median KL is a bound on the damage, not an estimate --
and a loose one. The real number needs the kernel, because the real format
cannot be stood in: `d * sc` is fp16 times a 6-bit integer, which needs 17
significand bits and bfloat16 has 8.

### Step 1 is done, and the oracle works

`tools/q4k.py` ports the format from `vendor/ggml/src/ggml-quants.c`. Against a
C harness calling ggml's own `dequantize_row_q4_K` on ggml's own blocks:

    DEQUANT vs ggml   max_abs = 0   exact = True

**That is the check a kernel needs** -- a kernel reads blocks, it does not write
them -- and it covers all three traps at once: the 12-byte scale packing, the
nibble interleave, and the subtraction. `q4k.quantize` reproduces none of
ggml's blocks (0 of 64) and is 0.77 points worse, which is the expected gap
between plain min/max and ggml's search. **So do not write a quantizer**: call
`quantize_q4_K` through ctypes, which is what produced the table above.

### The imatrix argument, which simplifies the combination

`quantize_q4_K(src, dst, nrows, n_per_row, imatrix)` takes an importance vector
natively. **`diag(H)` is exactly that**, and `refs/gptq-parent/hessians/` already
holds one for all 61 input groups. That is a far cheaper combination than the
GPTQ-plus-Q4_K problem described below -- no super-block coordination, no
changes to `gptq.py`, one extra argument. It is weighting rather than error
compensation, so expect less than GPTQ's 27%; measure it before assuming it
composes.

**MEASURED 2026-09-18, `q4k_ggml.py --imatrix`, and it does what it should.**
Two error columns, because only one of them is the objective -- a weighting
scheme is *supposed* to lose on unweighted error, and reporting only that
would be measuring the wrong object:

    module                      unweighted rel RMS        H-weighted rel RMS
                              plain +imat  change       plain +imat   change
    lm_head                   7.121  7.223  +1.44%      6.798  6.237   -8.3%
    layers.0.in_proj_qkv      7.144  7.238  +1.31%      6.652  5.682  -14.6%
    layers.0.out_proj         7.547  7.942  +5.24%     21.618  4.304  -80.1%
    layers.1.in_proj_qkv      7.360  7.635  +3.74%      8.187  5.583  -31.8%
    layers.1.out_proj         7.427  7.687  +3.50%      8.776  7.071  -19.4%
    layers.10.in_proj_qkv     7.595  7.875  +3.69%      8.048  5.618  -30.2%

**8 to 80% off the quantity `diag(H)` says the model cares about, for 1.3 to
5.2% of the quantity it does not, and one extra argument.** Layer 0's
`out_proj` is the shape of the thing: its H-weighted error is 21.6% where its
unweighted error is 7.5%, so its damage is concentrated exactly where the
calibration says it matters, and the imatrix removes four fifths of it.

Two cautions. The Hessian files hold the whole **`H`, [in, in]** -- passing
the array as it loads hands `quantize_q4_K` H's first *row*, which has
negative entries, and the quantizer answers with 38589% error. Take
`np.diag`. And this is still weight error: whether ~25% of a weighted norm
becomes KL is the kernel's question, not this one.

## The format, as ggml defines it

```c
#define QK_K 256
typedef struct {
    ggml_half d;         // super-scale for the sub-block scales
    ggml_half dmin;      // super-scale for the sub-block minima
    uint8_t   scales[12];// 8 x 6-bit scale + 8 x 6-bit min, packed
    uint8_t   qs[128];   // 256 x 4-bit codes
} block_q4_K;            // 144 bytes / 256 weights = 4.50 bits
```

Reconstruction, per sub-block of 32: `w = d * sc_i * q - dmin * m_i`, with
`sc_i`, `m_i` unsigned 6-bit.

Note the shape difference from affine: `bias = -dmin * m_i` is **subtracted**,
and both fields are unsigned. A port of the affine dequantizer that keeps its
sign conventions will be wrong in a way that still produces plausible text.

## Storage in a safetensors artifact

Keep three tensors per module rather than one packed blob, so the loader keeps
the shape it already has:

    <mod>.weight   uint32, [out, in * 4 / 32]      the 4-bit codes, as today
    <mod>.scales   uint8,  [out, in / 256, 12]     the packed 6-bit fields
    <mod>.dm       float16,[out, in / 256, 2]      d and dmin

`config.json`'s quantization block gets `"mode": "q4k"` with `group_size: 256`.
`check_rebuild.py` counts a module as quantized when it has more than one
tensor, so its 452/281 split survives -- the same way it survived nvfp4's
two-tensor modules.

## What to write, in order

1. **`tools/q4k.py`** -- **DONE.** The format in numpy, ported from
   `ggml-quants.c`. Its reader is exact against ggml (`max_abs == 0`); its
   writer is not and should not be used -- `tools/q4k_ggml.py` calls ggml's.
2. **The oracle harness -- built, and simpler than planned.** `vendor/ggml` is
   vendored; `tools/q4k_oracle.c` carries its own cmake line. `llama-quantize`
   turned out to be unnecessary: `quantize_q4_K` is exported from
   `libggml-base` and `tools/q4k_ggml.py` calls it through ctypes, so reference
   bytes come from three lines of Python. `tools/dump_q4k.py` now writes
   `refs/q4k/` from it -- the only reference case here whose oracle is not MLX.
3. **`kernels/q4k.metal` -- DONE, ported not invented.**
   `kernel_mul_mv_q4_K_f32_impl` and `dequantize_row_q4_K`, read off ggml's
   working code rather than re-derived from the format, and addressed for the
   three tensors instead of the packed block. **M = 1 only**, as planned.
   `src/ops/q4k.rs` is the dispatch side.
4. **The loader -- DONE.** `WeightStore::q4k_matrix` reads `.weight` /
   `.scales` / `.dm` and refuses a `.biases` or a `.global_scale`;
   `QuantMode::Q4k` exists so `config.json` can say the mode, and it names no
   mlx kernel -- every path that would have built one panics rather than
   resolving a name that does not exist. `dump_q4k.py` forges a one-module
   artifact so the loader is exercised rather than merely written.
5. **`tests/q4k.rs` -- DONE, four tests.** `q4k_dequantize` against ggml's own
   `dequantize_row_q4_K` at **`max_abs == 0.0`**, which is the contract this
   file was written to get. The matvec cannot have that contract -- its
   summation order is the kernel's, so no differently-computed reference can
   be bit-identical to it -- so it is checked against a float64 reference
   (measured `max_rel` 1.1e-7, bound 1e-5) and against a one-hot activation,
   where one term survives per output and an addressing error cannot hide
   inside a tolerance. That one measures exactly zero over 768 outputs. The
   fourth loads the forged artifact and dequantizes it back to the same bytes.

6. **`q4k_gather_qmv` -- DONE.** ggml's `kernel_mul_mv_id`, which turns out
   to be a *wrapper* rather than a kernel: it offsets the weights by the
   expert and the activations by the token and calls the matvec already
   ported. The body moved into an inline `qmv_impl` and both entry points
   resolve their own pointers. Checked over eleven rows whose experts repeat
   and whose tokens do not: 2.8e-7 against float64, and **exactly zero** with
   a one-hot activation, where each output row must be one column of one
   expert. That unlocks the 96 routed modules, 0.7172 GB of a decode step.

7. **Prefill** (`qmm_t`) -- **DONE, and it is one tile for both formats.**
   `kernels/kquant_mm.h` is the legacy (`simdgroup_matrix`) branch of ggml's
   `kernel_mul_mm`, 64 output features by 32 rows by 32 of K, with the format
   moved into a traits struct because the artifact stores three tensors where
   ggml stores one block. `kquant_gather_qmm_t` is the routed twin and it
   takes **mlx's** answer to the ragged part rather than ggml's: ggml builds a
   per-expert row list in a separate pass (`kernel_mul_mm_id_map0`) and
   dispatches a grid over experts, while mlx's `gather_qmm_rhs` walks the
   tile's own indices, finds each run of one expert and runs the K loop once
   per run. mlx's needs no second kernel and no scratch, keeps the port's own
   grid, and is *correct for any ordering* -- unsorted rows are slow, not
   wrong, which is a test rather than a comment.

   Three departures from ggml's kernel, and the middle one is the load-bearing
   one:

   - **the layout**, as above;
   - **the tile scalar is `float`, not ggml's `half`.** The gemm replaces a
     kernel that is already checked, so its contract is that it agrees with
     that kernel to within the difference in summation order. Rounding a
     dequantized weight to eleven significand bits is a 5e-4 perturbation --
     three orders above reassociation -- so it would not. At `float` the
     measured disagreement with the mat-vec is **2.0e-6**, which is what two
     float32 sums in different orders look like. (mlx's own affine gemm
     rounds its dequantized weights to *bfloat16*, so this tile is the more
     faithful of the two, and the probes show it.)
   - **the store** stages through threadgroup memory and converts, because
     `dst` here is the activation type where ggml's is float32. That also
     gives the routed variant its row slicing for free.

8. **Q6_K -- kernel DONE, artifact pending.** `kernels/q6k.metal` is ggml's
   `kernel_mul_mv_q6_K_f32_impl` and `dequantize_row_q6_K`, with `tests/q6k.rs`
   pinning the dequantize at `max_abs == 0.0`, the matvec at 7.9e-8, the
   one-hot at exactly zero and the routed gather at 9.0e-8. Two traps of its
   own, and neither is Q4_K's: the sub-block scale is **signed** and the code
   is **biased by 32**, so `w = d*sc*(q - 32)` with no minimum anywhere.

   The Rust sides of the two formats differed in three constants, so
   `ops::q4k` is now `ops::kquant` with a `KQuant` kind and `Linear` carries
   one k-quant variant rather than one per format.

   `forge_q4k.py --q6k` converts **128 of the 169** 8-bit modules: not
   `embed_tokens`, which is read by a gather and never by a matvec, and not
   `shared_expert_gate`, which is `[1, 2048]` where the kernel covers four
   rows at a time. 80 of the 128 get an importance vector.

9. **`in_proj_z`'s Hessian -- DONE, and free.** It reads the same tensor
   `in_proj_qkv` does, one line apart in both mlx-lm's `qwen3_5.py` and the
   port, so `X^T X` is the same matrix and the diagonal is the same vector.
   Borrowed on four layers it moves H-weighted error down 14-30% for 1-4% of
   unweighted, which is the shape every other module shows. The imatrix now
   covers all 91 rather than 61, at no calibration cost.

And between them and a verdict: a forge (`quantize_q4_K` over the 91 modules,
with the imatrix, writing the three tensors), the model wiring that routes a
q4k module through `ops::q4k::qmv`, and a probe capture. The instrument for
that capture exists and is checked -- `src/bin/probes.rs`, above.

Scope the first cut to the **91 modules** `int4-head-linattn` moves to 4 bits.
They are 57% of a decode step's traffic, they are what every number since
Phase 4 is measured on, and leaving the experts at affine means one new kernel
rather than two.

## A second reference, and it is tuned for this hardware

`vendor/accelerated-qwen-3.8` (Apache-2.0, upstream MIT in `licenses/`) is a
pinned llama.cpp patch that makes **Q4_K decode projections faster on Apple
Silicon**. `patches/metal-combination.patch` adds
`kernel_mul_mv_q4_K_f32_qwen_accel_{rows1,rows4,vector2}` -- direct derivatives
of ggml's own kernel, same `kmask1/2/3` and the same `ix/it/iq/ir`
decomposition, **specialised**: shape parameters moved into function constants
and narrow row counts given their own instantiations. That is exactly the
pattern this port already uses (`affine_qmv_fast` against `qmv_wide`,
`instantiate_quantized_batched`), so the tuning transfers conceptually even
though none of the code does.

It also pins the exact upstream to vendor: **llama.cpp b10434**, with a
published sha256 in `upstream.lock.json`. Take that revision for step 2.

**Do not read its headline as evidence for this phase.** Its 11.0% / 10.2%
decode gain on an M1 Pro is *its Q4_K kernel against ggml's Q4_K kernel* -- a
kernel-tuning result at a fixed format. Phase 14's question is the format
against affine at fewer bytes, and it is already answered. The two numbers
multiply rather than compete, and neither substitutes for the other.

## Q6_K for the 8-bit side, which this repo also priced without meaning to

The patch carries `q6_K` variants alongside `q4_K`, which is a reminder that
the other end of the artifact is unpriced. The shipping artifact's traffic:

    4-bit    1.5132 GB   72.9%
    8-bit    0.5527 GB   26.6%     <- 169 modules at affine int8 g64 = 8.50 bits
    bf16     0.0102 GB    0.5%

**Q6_K is 6.5625 bits.** Moving those 169 modules off 8.50 would cut them by
22.8%, which is **6.1% of the whole decode step** -- on top of Q4_K's 7.3%.
Phase 4 priced `mxfp8` for this slot and rejected it (power-of-two scale, no
zero point, 0.54% -> 2.94% error); Q6_K has neither defect.

**MEASURED 2026-09-18, `q4k_ggml.py --q6k`**, against the bf16 parent, with
ggml's own quantizer rather than a surrogate:

    module                ggml Q6_K   affine g64   ggml Q4_K
                            6.5625b        8.50b       4.50b
    self_attn.q_proj          1.853%       0.743%      7.387%
    self_attn.k_proj          1.864%       0.738%      7.432%
    self_attn.o_proj          1.891%       0.760%      7.509%
    mlp.gate (router)         2.139%       0.897%      8.304%
    shared_expert.gate        1.821%       0.727%      7.243%
    shared_expert.down        2.384%       1.024%      8.933%
    experts.gate_up           1.830%       0.725%      7.295%
    experts.down              1.814%       0.711%      7.250%
    embed_tokens              1.853%       0.737%      7.366%

**2.5x the error for 22.8% of the bytes**, and read on its own that sounds
like `mxfp8`'s 5.4x. It is not, and the difference is the whole result:

    step                     bits given up   error multiplier   doublings/bit
    affine 8.50 -> Q6_K            1.9375              2.49x           0.68
    Q6_K -> Q4_K                   2.0625              3.97x           0.96
    affine 5.00 -> Q4_K (Ph. 14)   0.5000              0.91x          -0.27

**The 8-bit side's bits are the artifact's most expensive.** Giving one up
there costs 0.68 error-doublings; giving one up on the 4-bit side costs 0.96,
and the artifact already pays that rate on 72.9% of its traffic. So if another
byte has to come from somewhere, it should come from here -- which is what
this measurement was for, and it is the opposite of `mxfp8`, which sat *off*
the curve rather than on it.

**AND IT IS POINTED AT THE WRONG STEP, which Phase 17 showed.** Built, the
8-bit side's bytes are the only ones that cost anything -- `q6k-all` is lower
than `q4k-all` on 22 of 64 probes, p = 0.017 -- while the 4-bit side's 187
modules were a coin flip across two phases. The reason is not that the
arithmetic above is wrong but that the two moves are different in kind:

    the 4-bit side   affine 5.00 bits, 8.45%  ->  Q4_K 4.50 bits, 7.48%
    the 8-bit side   affine 8.50 bits, 0.74%  ->  Q6_K 6.56 bits, 1.85%

Fewer bits and *less* error on one, fewer bits and 2.5x the error on the
other. The per-bit table compares two movements down the rate-distortion
curve; the 4-bit conversion is not on that curve, because a better grid buys
bytes and accuracy together. Rank steps of the same kind with it, and never a
grid change against a rate change.

What it does not say is that Q6_K keeps the quality: this is weight error, the
currency this rebuild has falsified twice as a selector (Phases 4 and 7).
Turning 6.1% of a step into a KL needs the second kernel, and the first one
has to pay first.

## What this does not need

- **No mlx change.** The artifact carries a mode mlx does not know, so
  `mlx_lm` cannot load it -- which costs the *MLX* oracle and nothing else. The
  golden tests for every other op are untouched, and `dump_probes.py` needs a
  port-side equivalent to capture distributions. That is the real hidden cost
  of this phase and it was to be priced before step 3.

  **PRICED, 2026-09-18, and it is small.** Three files, ~250 lines:
  `tools/export_probes.py` writes the set the port reads, `src/bin/probes.rs`
  runs the 64 contexts through the port at `dump_probes.py`'s own chunk width
  and writes raw logits, and `tools/probes_from_logits.py` applies mlx's
  `logsumexp` to them and emits an ordinary `probes.bin`. The normalisation is
  left to mlx deliberately: the port is then only ever asked for the thing it
  is bit-exact at, and `compare_probes.py` reads the result without knowing
  which instrument produced it.

  **And it was checked rather than assumed.** Run against an artifact mlx
  *can* load, the port's capture is bit-identical on **58 of 64 probes**. The
  other six are a bfloat16 rounding tie in one attention dot product -- see
  HANDOFF's "The divergence the probe set found" for the bisection; it is a
  toolchain difference, not a port bug, and it survives every flag tested.
  What it does to a verdict is nothing:

        KL(parent || gptq-head-linattn)   median      p90      max
        captured by mlx                   0.0297   0.2491   1.0122
        captured by the port              0.0297   0.2491   1.0122
        captured by mlx again             0.0297   0.2491   1.0122

  So the instrument transfers, the hidden cost is an afternoon rather than a
  phase, and **it is cheaper than the kernel it was blocking.** Proceed to
  step 3.
- **No change to `gptq.py`'s grid logic** for a first cut. GPTQ and the format
  are orthogonal; combine them only once the kernel is measured alone.

## The prize, and what it turned out to be

    int4-head-linattn + GPTQ    2.076 GB/token   median KL 0.0292   <-- ships today
    Q4_K twin                   1.925 (arith)    median KL 0.0396
    **Q4_K + GPTQ, unbuilt**    **1.925 real**   **~0.029 expected**

GPTQ is +27.1% of quality at fixed bytes; Q4_K is -7.3% of bytes at fixed
quality; they touch different things -- one picks the grid, the other picks the
point on it. Together: **-29.7% of a decode step against the control at the
quality GPTQ already delivers.**

**BUILT, 2026-09-18. `REBUILD.md` Phase 15 has the numbers; the summary is
that the bytes are real and the prize is not demonstrated.**

    artifact              bytes/token   decode    median KL, all 64 probes
    int4-head-linattn      2.0761 GB   102.6 t/s          0.0350
    q4k                    1.9814 GB   106.8 t/s          0.0303
    q4k + imatrix          1.9814 GB   104.8 t/s          0.0260
    gptq-head-linattn      2.0761 GB        --            0.0273

Three corrections to the paragraph above, and the first is the one that
matters:

- **-4.6%, not -7.3%** -- for the first cut. The twin re-encoded all 283
  four-bit modules; that cut converted the 91. **Phase 16 added the 96 routed
  experts and the number arrived: 1.9311 GB/token, -7.0% of a decode step and
  +5.2% of the rate, with the experts costing 31 of 64 probes against the same
  artifact without them, which is a coin flip.** One probe is not fine --
  `arith/ru-7x8` pays 4.65 nats against 0.25 without the experts -- and it is
  the probe that has broken every fragile format here.
- **The quality difference is not measurable.** Rank tests over the 64 probes:
  q4k lower than the affine int4 it replaces on 32 of 64 -- a coin flip --
  and q4k+imatrix lower than GPTQ on 37 of 64, p = 0.26. So *the bytes are
  free*, which is the result; "GPTQ's quality at Q4_K's bytes" is the best
  point estimate of the four and is not demonstrated.
- **The combination that paid was the cheap one.** Not GPTQ x Q4_K, with its
  super-block coordination problem, but `quantize_q4_K`'s own importance
  vector fed `diag(H)` -- one argument, no changes to `gptq.py`. It halves
  nothing on the median and it takes `arith/ru-7x8` from 1.9555 to 0.2470
  nats, which is a tail result rather than a central one.

And a fourth falsification of weight error, this time of a measure built not
to be falsifiable that way: Q4_K beats affine int4 on total weight error, on
weight error in every magnitude decile, on the error in the product against
random `x`, and on the **H-weighted** error in 59 of the 61 modules that have
a Hessian. The probes say the two are level.

The combination has one structural snag, and it is worth knowing before
starting: a super-scale needs all eight of a super-block's sub-block scales at
once, while `gptq.py` derives a group's scale from the weights it is about to
quantize, one group at a time. Either derive the super-scale from the
pre-compensation weights (an approximation) or make the walk's block the whole
256-column super-block (exact, but reintroduces the staleness that Phase 7's
"the block is the group" decision removed on purpose).

## Where the crossover is

`kquant::matmul` has to pick one of the two kernels at every call, and the
answer is not obvious in either direction. The mat-vec re-reads the whole
matrix once per row, so its cost is linear in M from the second row on. The
tile reads it once per 32 rows but pays for a 32-row tile whether or not the
rows are there. `src/bin/kquant_mm.rs` times both at M = 1..64 on the four
shapes the two formats are used at -- best of forty windows of thirty
dispatches, tile time over mat-vec time:

    shape                     M=1    M=4    M=6    M=8   M=16   M=32
    in_proj_qkv [8192, 2048]  3.91   1.65   1.13   0.86   0.44   0.22
    out_proj    [2048, 4096]  4.62   1.74   1.22   0.94   0.50   0.25
    q_proj      [2048, 2048]  2.89   1.11   0.80   0.63   0.33   0.18
    down_proj   [2048,  512]  1.81   0.93   0.84   0.67   0.40   0.22

**The crossover is 4 to 8 rows, and `MM_ROW_LIMIT` is 8** -- the conservative
end, right for the two widest shapes and leaving the two narrow ones a little
on the mat-vec. It also keeps the speculative verify window, `depth + 1` = 4
rows, exactly where it was; a prefill kernel has no business moving the MTP
path.

The shape of that table is worth reading rather than just its crossing point:
**the tile's own time is flat from M = 1 to M = 32** -- 98.4 to 99.1 us for
`in_proj_qkv` across the whole range -- and doubles at 64, where a second row
tile appears. So the ratio is the mat-vec's straight line divided by a
constant, and at M = 1 the tile is 1.8x to 4.6x worse. The guard is not
optional.

**And one measurement lesson, which is `HANDOFF.md`'s own rule one level up.**
The first run of that bench timed *one* dispatch between two `sync`s and every
shape came back at 120-190 us, crossover invisible. That is the submit-and-wait
round trip on this machine -- thirty times a small matrix's own kernel time.
Chaining thirty dispatches inside each window is what made the difference
measurable at all.

## The instrument moved, and a control says it is the tile, not the kernel

`Q4K.md` and `REBUILD.md` both say a probe capture has to come back with the
same KL table or the kernel is wrong. It did not, quite, and the first reading
looked bad: `q6k-full`'s median went **0.0327 -> 0.0354** and the head-to-head
printed "lower on 19 of 57", which reads like a significant loss.

Two things were wrong with that reading, and the second is the interesting one.

**The sign test was counting ties as losses.** 13 of 64 probes are bit-identical
between the two captures and many more are identical on the live set. Over the
probes that actually moved it is a coin flip:

    pair                              moved   gemm lower on      p   median
    q4k-all      -> gemm                 43       18    0.360   0.0320 -> 0.0315
    q6k-full     -> gemm                 44       19    0.451   0.0327 -> 0.0354
    int4 mat-vec -> mlx's own gemm       34       19    0.608   0.0447 -> 0.0427

**And the third row is the control.** `ORNITH_AFFINE_GEMM=0` (a measurement
escape hatch in `ops::quantized`, inert by default) runs an *all-affine*
artifact's prefill through mlx's mat-vec instead of mlx's tiled gemm -- the
same move, on a path nobody here wrote, with no k-quant anywhere in it. It
moves the capture by **more** than either k-quant does:

    pair                          max_abs   rows bit-identical   largest KL
    int4 mat-vec vs mlx's gemm      3.116        24 of 64          0.348
    q6k-full mat-vec vs this tile   3.062        13 of 64          0.152
    q4k-all  mat-vec vs this tile   2.671        13 of 64          0.062

So: **changing prefill from a mat-vec per token to a tile perturbs a capture,
on any quantization family, and this tile perturbs it less than mlx's own.**
That last part is not luck -- mlx's `qmm_t` dequantizes its weights into
threadgroup memory as *bfloat16* and this one uses float, which is the whole
reason the tile scalar was made a template parameter.

What it costs to know: the recorded verdicts of Phases 15-18 were all taken on
a mat-vec prefill, while every affine verdict in this rebuild was taken on a
tiled one. **The captures were never on the same instrument.** The orderings
survive it -- `q4k-all` 0.0301 < `q6k-full` 0.0344 < `int4` 0.0422, the same
order and the same `max` to four decimals on both k-quant artifacts, with
`arith/ru-7x8` landing on the same side of its cliff -- but that was worth
checking rather than assuming, and it is the kind of thing that would have sat
undetected under a table.

## How this goes wrong

- **The sign convention.** `w = d*sc*q - dmin*m` subtracts. Affine adds. Both
  produce fluent text when wrong.
- **The 6-bit packing.** ggml's 12-byte layout is not the obvious one; six of
  the bytes carry the low bits of all sixteen fields. Port it from the source,
  do not re-derive it.
- **Believing a number that is too bad.** Phase 14's first reading was
  18.8-25.4% and looked like a verdict on the format; it was `floor` where
  `ceil` belonged. A format the entire llama.cpp ecosystem ships cannot measure
  three times worse than plain affine.
