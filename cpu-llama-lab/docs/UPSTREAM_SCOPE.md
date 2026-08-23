# Upstream contribution scope

## PR 1: Knights Landing CPU variant
Only the KNL backend/ISA bundle, dispatch/build plumbing, tests, and one demonstrably correct hot-loop optimization. No benchmark orchestration, distro installer, NUMA mirror, model downloader, or quality harness.

Acceptance:
- zero behavior change when KNL variant is not selected;
- correct runtime score / no illegal AVX-512BW/DQ/VL/VNNI dependency;
- `test-backend-ops` passes;
- compile/disassembly evidence;
- benchmark with optimization ON/OFF on physical KNL.

## Later PRs
Land q3/q4/q6 kernel work separately. Memory-tiering and NUMA-mirror work are independent architecture changes and should not be tied to initial KNL support.
