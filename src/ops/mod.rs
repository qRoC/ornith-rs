//! Kernel dispatch, one module per mlx primitive family. Each mirrors the
//! corresponding `mlx/backend/metal/*.cpp`.

pub mod conv;
pub mod elementwise;
pub mod embedding;
pub mod gdn;
pub mod moe;
pub mod norm;
pub mod quantized;
pub mod rope;
pub mod sample;
pub mod scan;
pub mod searchsorted;
pub mod sort;
pub mod indexing;
pub mod random;
pub mod reduce;
pub mod layout;
pub mod matmul;
pub mod sdpa;
pub mod softmax;
