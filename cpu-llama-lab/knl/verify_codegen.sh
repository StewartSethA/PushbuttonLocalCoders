#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/out}"
mkdir -p "$OUT"
CC_BIN="${CC:-}"
if [ -z "$CC_BIN" ]; then
  for c in gcc-14 gcc-13 gcc-12 gcc-11 gcc; do
    command -v "$c" >/dev/null 2>&1 || continue
    probe="$OUT/.cc-probe.o"
    if printf 'int x;' | "$c" -x c -march=knl -c -o "$probe" - 2>/dev/null; then CC_BIN="$c"; rm -f "$probe"; break; fi
  done
fi
if [ -z "$CC_BIN" ]; then echo "SKIP: no GCC <=14 with -march=knl; run ./cpu-llama-lab.sh install on KNL"; exit 0; fi
echo "compiler=$($CC_BIN --version | head -1)"

# Native executable: correctness test can run on this host if AVX2/FMA exist.
$CC_BIN -O3 -mavx2 -mfma "$HERE/knl_q4_probe.c" -lm -o "$OUT/q4-native"
"$OUT/q4-native" | tee "$OUT/native-run.txt"

# KNL target: compile only here; executing a KNL-specific binary on a non-KNL host is not safe.
$CC_BIN -O3 -march=knl -mtune=knl -c "$HERE/knl_q4_probe.c" -o "$OUT/q4-knl.o" 2>"$OUT/knl-compile.stderr" || {
  cat "$OUT/knl-compile.stderr"; exit 2;
}
objdump -d "$OUT/q4-knl.o" > "$OUT/q4-knl.objdump"
$CC_BIN -march=knl -dM -E -x c /dev/null | sort > "$OUT/knl-macros.txt" 2>"$OUT/knl-macros.stderr"

need(){ grep -q "$1" "$OUT/knl-macros.txt" || { echo "MISSING macro $1"; exit 3; }; }
for m in __AVX2__ __FMA__ __AVX512F__ __AVX512CD__ __AVX512ER__ __AVX512PF__; do need "$m"; done
for m in __AVX512BW__ __AVX512DQ__ __AVX512VL__ __AVX512VNNI__; do
  if grep -q "$m" "$OUT/knl-macros.txt"; then echo "FORBIDDEN macro unexpectedly enabled: $m"; exit 4; fi
done

grep -Eq 'vpmaddubsw|vpmaddwd' "$OUT/q4-knl.objdump" || { echo "expected quant dot instructions absent"; exit 5; }
grep -Eq 'vfmadd|vfmadd231ps|vfmadd132ps|vfmadd213ps' "$OUT/q4-knl.objdump" || { echo "expected FMA absent"; exit 6; }
grep -Eq 'prefetch' "$OUT/q4-knl.objdump" || { echo "expected software prefetch absent"; exit 7; }

echo "PASS: native correctness + KNL cross-compile + macro ISA audit + objdump hot-loop audit"
