// Instantiations of mlx's block sort.
//
// `mx.argpartition` has no separate GPU kernel: `ArgPartition::eval_gpu` says
// "We direct arg partition to sort for now" and calls `gpu_merge_sort`. So the
// MoE router's `mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]` is a full
// ascending argsort followed by a slice, and the top-k indices come out in a
// fully determined order -- which matters, because that order fixes the
// summation order of `(y * scores).sum(axis=-2)` afterwards.
//
// For 256 experts mlx picks tn=4 and bn=64 (`gpu_merge_sort`: potential_bn =
// ceil(256/4) = 64, and the ladder takes the first bucket 64 is not greater
// than), giving n_per_block = 256 -- one block, so `single_block_sort`.
//
// uint32 is instantiated alongside so tests can check that this port's
// counting-sort permutation for the expert gather matches mlx's argsort.

// clang-format off
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"

#define instantiate_block_sort(                                             \
    name, itname, itype, otname, otype, arg_sort, bn, tn)                   \
  instantiate_kernel("c" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn,  \
                     block_sort, itype, otype, arg_sort, bn, tn)            \
  instantiate_kernel("nc" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn, \
                     block_sort_nc, itype, otype, arg_sort, bn, tn)

#define instantiate_arg_block_sort_base(itname, itype, bn, tn) \
  instantiate_block_sort(                                      \
      arg_block_sort, itname, itype, uint32, uint32_t, true, bn, tn)

instantiate_arg_block_sort_base(bfloat16, bfloat16_t, 64, 4)
instantiate_arg_block_sort_base(float32, float, 64, 4)
instantiate_arg_block_sort_base(uint32, uint32_t, 32, 4)
instantiate_arg_block_sort_base(uint32, uint32_t, 64, 4)
instantiate_arg_block_sort_base(uint32, uint32_t, 128, 4)
instantiate_arg_block_sort_base(uint32, uint32_t, 256, 4)
instantiate_arg_block_sort_base(uint32, uint32_t, 512, 4) // clang-format on

// --- multi-block sort --------------------------------------------------------
//
// The sampler argsorts a whole 248320-wide logit row. `gpu_merge_sort` picks
// tn = 4 and, since ceil(248320/4) = 62080 > 256, bn = 512 (bfloat16 is 2
// bytes, so the `size_of(dtype) > 4` demotion to 256 does not apply). That is
// 2048 per block and 122 blocks, so `multi_block_sort` runs: one blockwise
// sort, then ceil(log2(122)) = 7 partition/merge rounds.
//
// float32 is instantiated alongside for the tests.

#define instantiate_multi_block_sort(                                             \
    vtname, vtype, itname, itype, arg_sort, bn, tn)                               \
  instantiate_kernel("sort_mbsort_" #vtname "_" #itname "_bn" #bn "_tn" #tn,      \
                     mb_block_sort, vtype, itype, arg_sort, bn, tn)               \
  instantiate_kernel("partition_mbsort_" #vtname "_" #itname "_bn" #bn "_tn" #tn, \
                     mb_block_partition, vtype, itype, arg_sort, bn, tn)          \
  instantiate_kernel("merge_mbsort_" #vtname "_" #itname "_bn" #bn "_tn" #tn,     \
                     mb_block_merge, vtype, itype, arg_sort, bn, tn)

instantiate_multi_block_sort(bfloat16, bfloat16_t, uint32, uint32_t, true, 512, 4)
instantiate_multi_block_sort(float32, float, uint32, uint32_t, true, 512, 4)
