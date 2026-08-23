# KNL methodology

Baseline: Flat + Quadrant. Cache + Quadrant is a separate whole-BIOS ablation. SNC is later work.

Use GCC <=14 because newer GCC removed KNL target support. Do not enable generic `GGML_AVX512` on KNL: later subsets such as BW/DQ/VL/VNNI are not available.

A successful KNL optimization requires numerical correctness, codegen evidence, runtime path evidence and an ON/OFF speed comparison. The lab intentionally keeps each prefetch/unroll choice separable.
