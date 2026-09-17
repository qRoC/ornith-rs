# What being allowed to design the artifact actually buys

`REBUILD.md` is the plan. This is the exploratory companion: what becomes
possible once the weights are ours to lay out and the algorithm is ours to
write, with the arithmetic done rather than asserted. Numbers here come from
`tools/quant_error.py` and short simulations; the model-dependent ones are
marked unmeasured.

## The ceiling on formats, measured

Relative RMS weight error against the rate-distortion floor for a Gaussian
source, which is the best any code can do at that rate:

All from `tools/quant_error.py`; 4-bit elements throughout except the first
row. Three weight distributions, because real weights are heavier-tailed than
a gaussian and the ranking moves with it.

| scheme | bits/w | gaussian | laplace | t(4) | floor |
|---|---|---|---|---|---|
| affine int8 g64 -- the 260 today | 8.50 | 0.54% | 0.67% | 0.70% | 0.28% |
| affine g32, bf16 scale+bias -- **the body today** | 5.00 | 7.82% | 9.18% | 9.37% | 3.12% |
| **affine g16, fp8 scale+bias** | **5.00** | **7.02%** | **7.77%** | **7.78%** | 3.12% |
| **Lloyd-Max codebook g16, fp8 scale** | **4.50** | **7.50%** | **8.09%** | **8.12%** | 4.42% |
| Lloyd-Max codebook g32, bf16 scale | 4.50 | 8.21% | 9.22% | 9.27% | 4.42% |
| affine g32, fp8 scale+bias | 4.50 | 8.14% | 9.51% | 9.72% | 4.42% |
| affine g16, fp8 scale, no zero-point | 4.50 | 8.35% | 9.97% | 10.43% | 4.42% |
| affine g64, bf16 scale+bias | 4.50 | 8.97% | 11.13% | 11.74% | 4.42% |
| nvfp4 g16 | 4.50 | 9.51% | 9.28% | 9.18% | 4.42% |
| mxfp4 g32 | 4.25 | 20.32% | 21.19% | 22.00% | 5.26% |
| VQ dim 2, 256 codes, g32 | 4.50 | 8.54% | -- | -- | 4.42% |
| VQ dim 4, 4096 codes, g32 | 3.50 | 14.97% | -- | -- | 8.84% |

Three readings:

- **Every scalar-with-group scheme sits about 2x above the floor.** There is
  roughly **one bit a weight** of theoretical headroom, and group-scaled
  scalar quantization does not reach it.
- **Vector quantization does not collect it either.** At equal rate VQ dim 2
  beats scalar by 6% relative (8.54 vs 9.07 symmetric), and larger dimensions
  lose badly at the codebook sizes that fit in a threadgroup. The gains the
  literature reports at 2-3 bits come from codebooks far larger than this, plus
  residual stages.
- **The cheap wins are in the group metadata, not the grid**, and they are
  bigger than they look on a gaussian. Moving from a bf16 scale+bias at g32 to
  an **fp8 scale+bias at g16** buys 10% less error on a gaussian and **15-17%
  on the heavy-tailed distributions real weights resemble** -- at the *same*
  5.00 bits, because the overhead fp8 frees pays for a group half the size.
  Same kernel shape, same 4-bit elements, same packing.
- **And a fitted codebook at g16 beats today's scheme while spending less.**
  Lloyd-Max at group 16 with an fp8 scale is 7.50 / 8.09 / 8.12% against
  today's 7.82 / 9.18 / 9.37% -- better on all three distributions at **0.50
  fewer bits a weight**. That is a 10% byte saving with a quality improvement
  attached, for a per-tensor 16-entry table and a kernel that indexes it.

## What voxel engines solved, and which of it transfers

They had the same problem -- more data than memory, a fixed budget per access
-- and the mapping is exact enough to be useful:

| their technique | what it exploits | ours | verdict |
|---|---|---|---|
| sparse octree | empty space | weight sparsity | dense-trained, nothing empty |
| **sparse voxel DAG** (dedup identical subtrees) | repeated structure | identical weight blocks | a 32-weight group at 4 bits carries ~122 bits of entropy; exact duplicates do not occur |
| **palette per chunk** | few distinct values locally | a codebook per tensor | measured above: +6% at equal rate |
| **BC / ASTC block compression** -- fixed budget, random access | local coherence | affine quantization with a group | **this is what we already run** |
| level of detail | distance from the camera | precision per module | **already run**: 4-bit body, 8-bit for 260 modules |
| **virtual texturing / streaming** | visibility -- fetch only what is needed | read only the weights that can affect the answer | **not done, and the one that transfers** |
| distance fields | computation instead of storage | low-rank factorisation | testable once bf16 weights land; transformer weight spectra decay slowly, so expect little |
| entropy coding | non-uniform symbol use | coded quantization levels | real (levels are not uniformly used) but it destroys the random access a matvec needs |

## The idea that comes out of that: a two-stage exact `lm_head`

`lm_head` is 248320 x 2048, **540 MB and 19.1% of every decode step**, and
greedy decoding needs exactly one number out of it: which row has the largest
inner product. Everything else read is thrown away. That is the virtual-texture
situation precisely.

It can be pruned **exactly**, not approximately:

1. Store a second copy of `lm_head` at 2 bits, and a per-row bound
   `d_v = ||w_v - w'_v||` from the build (248320 bytes in fp8, 0.25 MB).
2. Per token, compute approximate logits against the 2-bit copy: 143 MB.
3. `|y_v - y'_v| = |(w_v - w'_v).h| <= d_v.||h||`, so any row with
   `y'_v + d_v||h||` below `max_u (y'_u - d_u||h||)` **cannot** be the argmax.
4. Refine the survivors against the real rows and take the max.

| | MB a token | |
|---|---|---|
| `lm_head` today, 8-bit g64 | 540.3 | 19.1% of a step |
| 2-bit first pass | 143.0 | |
| + refining 1024 candidates at 8 bits | 145.5 | **saves 14.0% of a step** |
| + refining 4096 candidates | 152.2 | saves 13.7% |

The candidate count barely matters, which is what makes it robust: even a
pathological 50000 survivors would land near 253 MB, still half of today. It
degrades gracefully rather than failing.

**Unmeasured:** how many candidates actually survive, which depends on the gap
between the top logits and `2 d ||h||`. Measurable in one forward pass once the
artifact exists, and the table says the answer barely changes the verdict.

Caveats worth stating: the argument is exact for **greedy**, where
`argmax(log_softmax(x)) = argmax(x)` and no normalisation is needed. Sampled
decoding needs the softmax denominator over the whole vocabulary, so it needs
a bound on the discarded mass as well -- doable, more delicate, and not the
path the tests or the MTP verify take. And it composes with everything else:
the refine stage can read 4-bit rows, and FR-Spec already prunes the *draft*
head by a different mechanism.

## The highest-value conventional idea: activation-aware scaling

Not novel, but it fits the constraint exactly and costs nothing at runtime.
Weight error is not what matters; **output** error is, and the two differ
because activation magnitudes differ per input channel. Scaling salient
channels up before quantizing and folding the inverse into the preceding
RMSNorm weight (`W' = W diag(s)`, `x' = x diag(1/s)`, product unchanged) makes
the same 4 bits carry the channels that matter.

Why it suits this project better than anything else here:

- It produces an **ordinary affine-quantized artifact**. No new Metal, no new
  kernel family, and `max_abs == 0.0` against MLX survives untouched, because
  MLX simply reads a differently-quantized model.
- It attacks the actual question -- can the 260 modules live at 4 bits -- from
  the only side that has room, given that the format ceiling above says the
  grid itself has about one bit left in it.
- It needs a calibration pass over the parent, which is exactly what having
  the parent buys.

## Checked and rejected

- **Block deduplication.** Entropy per group is too high for exact matches.
- **Weight sparsity.** Dense training; nothing to skip.
- **Exact MIPS by clustering `lm_head`.** The bound `c.h + r||h||` is useless
  at 2048 dimensions with any tractable number of clusters -- the residual
  radius barely shrinks. The two-stage scheme above works because its bound is
  per row and comes from quantization error, not from geometry.
- **Low-rank factorisation of `in_proj`.** Break-even needs rank below 80% of
  full; transformer spectra rarely oblige. One `svd` once the weights land
  settles it.
- **A bespoke format.** Every gain above is available inside formats MLX
  already implements, and a bespoke one costs the oracle -- which is the thing
  that has caught every bug in this port.

## If it were ranked

1. **Activation-aware scaling** -- no kernels, keeps the oracle, attacks the
   binding question from the only side with room left.
2. **fp8 scale at group 16** instead of bf16 at 32 -- 10-17% less error at
   identical bits and an unchanged kernel shape. The closest thing here to
   free, and it needs mlx to accept an fp8 group scale in `affine` mode, which
   is the one thing to check before counting on it.
3. **Two-stage `lm_head`** -- 14% of a step, exact for greedy, independent of
   every other item on this list.
4. **Lloyd-Max codebook at g16** -- better than today on all three
   distributions at 0.50 fewer bits, so ~10% of bytes with quality attached.
   Costs a per-tensor table, a new kernel, and the oracle unless mlx grows the
   same format.
5. Vector quantization -- 6% at equal rate for a new kernel and a new format.
   Not worth it at codebook sizes that fit in a threadgroup.
