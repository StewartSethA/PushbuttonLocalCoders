# Ubuntu notes

## Ubuntu 26.04 and KNL

Ubuntu 26.04's current compiler line can coexist with the distro `gcc-14` / `g++-14` packages. The lab does not replace the system compiler: it selects GCC <=14 only for the KNL build after compiling real `-march=knl` C and C++ probes.

This is preferable to installing an old Ubuntu userspace. The KNL binary is still compiled against the current host's libc, libstdc++, libnuma and other distro libraries.

## Kernel-specific perf

Ubuntu packages `perf` with kernel-specific linux-tools packages. The installer therefore first tries `linux-tools-$(uname -r)` and then generic/common meta-packages. It subsequently runs a real `perf stat` probe; package installation success alone is not accepted as evidence that counters work.

Custom/OEM kernels may have no matching Ubuntu tools package. This is a nonfatal instrumentation limitation, not a llama.cpp build failure.

## Library consistency

The lab does not add PPAs or foreign-release repositories. Apt hosts run `apt-get check` and `dpkg --audit`. Built ELF files are then checked independently with `ldd` and `readelf`.
