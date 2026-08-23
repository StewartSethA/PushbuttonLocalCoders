# NUMA methodology

The suite treats NUMA policies as ablations, never dogma. Mainline `--numa distribute` relies heavily on first-touch locality and mmap page state. Interleave maximizes aggregate channel participation but pays remote latency. Strict node-local runs establish the single-socket/node baseline.

Weights-only mirror, where supported by a tested build, is evaluated separately because it spends RAM capacity to give each compute node a local read-only copy of model weights. Independent node-local server/bench workers are an aggregate-throughput comparison, not a substitute for a single mirrored inference stream.

For multi-socket publication-quality runs record BIOS NPS/SNC mode, SMT state, NUMA balancing, page-cache protocol, DIMM population and STREAM bandwidth per node.
