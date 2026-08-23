# Profiles
Profiles are generated from hardware detection rather than selected manually. The planner classifies KNL, Broadwell-EP, Skylake-SP, Cascade Lake, Ice Lake, Sapphire/Granite Rapids, Zen 2/3/4/5 EPYC, and Threadripper/Threadripper Pro. Unknown x86 CPUs fall back to `GGML_NATIVE=ON` plus the portable AVX2 ceiling when supported.
