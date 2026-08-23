# Toolchain, ABI and benchmark hardening

## Goals

1. Never use an illegal compiler target on KNL.
2. Never solve that problem by importing an old distro runtime.
3. Build against the target host's current libc and libraries.
4. Prevent user environment variables from silently selecting unrelated headers/libraries.
5. Detect unresolved libraries before benchmarking.
6. Record enough state to explain cross-host differences later.

## KNL compiler rule

A compiler is accepted only if C and C++ `-march=knl` probes compile. GCC 14 is preferred on modern Ubuntu because GCC 15 removed the KNL target. The host's default compiler remains untouched.

## ABI rule

Every produced ELF is inspected with `ldd` and `readelf`. Unresolved dependencies fail the audit. GLIBC/GLIBCXX requirements and RPATH/RUNPATH are retained in `abi/`.

## Package-manager rule

The installer uses only currently configured distro repositories. It never adds a PPA or copies glibc/libstdc++ from another distribution release. Apt hosts run `apt-get check` and record `dpkg --audit`/held packages.

## Cache/tool locality

Suite-managed uv, venvs, uv cache, XDG cache and ccache stay below `LAB_ROOT`.

## Benchmark policy

System-wide policies (CPU governor, turbo, THP, automatic NUMA balancing, mitigations) are recorded rather than silently changed. `LAB_STRICT=1` converts high-severity environmental conditions into a hard stop.
