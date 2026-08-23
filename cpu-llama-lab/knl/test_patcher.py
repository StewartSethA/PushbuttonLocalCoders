#!/usr/bin/env python3
import subprocess, tempfile, os
from pathlib import Path
HERE=Path(__file__).resolve().parent
PATCHER=HERE/'patches'/'apply_knl_patch.py'

def w(root, rel, s):
    p=root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(s)

def main():
    base=Path(os.environ.get('LAB_ROOT', str(Path.cwd()/'.cpu-llama-lab'))); base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base) as td:
        r=Path(td)
        w(r,'ggml/CMakeLists.txt','option(GGML_AVX512           "ggml: enable AVX512F"          OFF)\n')
        w(r,'ggml/src/CMakeLists.txt','''function(ggml_add_cpu_backend_variant tag_name)\n                      AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16\nendfunction()\n        ggml_add_cpu_backend_variant(haswell            SSE42 AVX F16C FMA AVX2 BMI2)\n''')
        w(r,'ggml/src/ggml-cpu/CMakeLists.txt','''                if (GGML_AVX512)\n                    list(APPEND ARCH_FLAGS -mavx512f)\n                    list(APPEND ARCH_FLAGS -mavx512cd)\n                    list(APPEND ARCH_FLAGS -mavx512vl)\n                    list(APPEND ARCH_FLAGS -mavx512dq)\n                    list(APPEND ARCH_FLAGS -mavx512bw)\n                endif()\n''')
        w(r,'ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp','''int score(){ cpuid_x86 is; int score=1;\n#ifdef GGML_AVX512\n if (!is.AVX512F()) return 0;\n#endif\nreturn score;}\n''')
        q='''#include <immintrin.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <stddef.h>\n#define GGML_UNUSED(x) (void)(x)\n#define GGML_RESTRICT __restrict__\n#define UNUSED GGML_UNUSED\ntypedef struct { char z[64]; } block_q3_K; typedef struct { char z[64]; } block_q4_K; typedef struct { char z[64]; } block_q6_K;\ntypedef struct { float d; unsigned char qs[16]; } block_q4_0; typedef struct { float d; signed char qs[32]; } block_q8_0; typedef struct { char z[64]; } block_q8_K;\n#define MM256_SET_M128I(a,b) _mm256_insertf128_si256(_mm256_castsi128_si256(b),(a),1)\nstatic inline __m256i bytes_from_nibbles_32(const unsigned char*r){const __m128i x=_mm_loadu_si128((const __m128i*)r);return _mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(x,4),x),_mm256_set1_epi8(15));}\nstatic inline __m256 mul_sum_i8_pairs_float(__m256i x,__m256i y){__m256i ax=_mm256_sign_epi8(x,x),sy=_mm256_sign_epi8(y,x);return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1),_mm256_maddubs_epi16(ax,sy)));}\nstatic inline float hsum_float_8(__m256 x){float a[8];_mm256_storeu_ps(a,x);float s=0;for(int i=0;i<8;i++)s+=a[i];return s;}\nvoid ggml_vec_dot_q3_K_q8_K(int n, float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc) {\n    const int nb=n; const block_q3_K*x=vx; const block_q8_K*y=vy;\n#if defined __AVX2__\n    for (int i = 0; i < nb; ++i) { (void)x[i]; (void)y[i]; }\n#else\n    for (int i = 0; i < nb; ++i) { (void)x[i]; (void)y[i]; }\n#endif\n}\nvoid ggml_vec_dot_q4_K_q8_K(int n, float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc) {\n    const int nb=n; const block_q4_K*x=vx; const block_q8_K*y=vy;\n#if defined __AVX2__\n    for (int i = 0; i < nb; ++i) { (void)x[i]; (void)y[i]; }\n#else\n    for (int i = 0; i < nb; ++i) { (void)x[i]; (void)y[i]; }\n#endif\n}\nvoid ggml_vec_dot_q6_K_q8_K(int n, float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc) {\n    const int nb=n; const block_q6_K*x=vx; const block_q8_K*y=vy;\n#if defined __AVX2__\n    for (int i = 0; i < nb; ++i) { (void)x[i]; (void)y[i]; }\n#else\n    for (int i = 0; i < nb; ++i) { (void)x[i]; (void)y[i]; }\n#endif\n}\n// Regression for llama.cpp d775b896: unrelated global AVX/AVX2 data follows q6_K.\n#if defined (__AVX__) || defined (__AVX2__)\nstatic const int8_t post_q6_global_table[2] = { 1, -1 };\n#endif\nvoid ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {\n    const block_q4_0*x=vx; const block_q8_0*y=vy; int ib=0,nb=n; float sumf=0;\n#if defined(__AVX2__)\n    // Initialize accumulator with zeros\n\n    __m256 acc = _mm256_setzero_ps();\n\n    // Main loop\n\n    for (; ib < nb; ++ib) {\n        const __m256 d = _mm256_set1_ps( x[ib].d * y[ib].d );\n        __m256i qx = bytes_from_nibbles_32(x[ib].qs);\n        // Now we have bytes in [0..15]; offset them to signed q4 values.\n        const __m256i off = _mm256_set1_epi8( 8 );\n        qx = _mm256_sub_epi8( qx, off );\n        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);\n        const __m256 q = mul_sum_i8_pairs_float(qx, qy);\n        // Multiply q with scale and accumulate.\n        acc = _mm256_fmadd_ps( d, q, acc );\n    }\n\n    sumf = hsum_float_8(acc);\n#endif\n}\n'''
        w(r,'ggml/src/ggml-cpu/arch/x86/quants.c',q)
        subprocess.run(['python3',str(PATCHER),str(r),'--dry-run'],check=True)
        subprocess.run(['python3',str(PATCHER),str(r),'--apply'],check=True)
        subprocess.run(['python3',str(PATCHER),str(r),'--check'],check=True)
        # Idempotent reapply
        subprocess.run(['python3',str(PATCHER),str(r),'--apply'],check=True)
        assert 'AVX512_KNL' in (r/'ggml/src/CMakeLists.txt').read_text()
        assert 'GGML_KNL_PREFETCH_PAIR' in (r/'ggml/src/ggml-cpu/arch/x86/quants.c').read_text()
        subprocess.run(['gcc','-O2','-mavx2','-mfma','-DGGML_AVX512_KNL','-DGGML_KNL_PREFETCH','-DGGML_KNL_Q4_0_UNROLL','-c',str(r/'ggml/src/ggml-cpu/arch/x86/quants.c'),'-o',str(r/'patched-quants.o')],check=True)
        subprocess.run(['python3',str(PATCHER),str(r),'--revert'],check=True)
        assert 'KNL_LLAMA_LAB_V3' not in (r/'ggml/CMakeLists.txt').read_text()

        # Transactionality regression: break the final quant anchor. The patcher
        # must fail before modifying *any* of the other four upstream files.
        qp=r/'ggml/src/ggml-cpu/arch/x86/quants.c'
        qp.write_text(qp.read_text().replace('void ggml_vec_dot_q6_K_q8_K(int n,','void broken_q6_K_q8_K(int n,',1))
        before={rel:(r/rel).read_bytes() for rel in [
            'ggml/CMakeLists.txt','ggml/src/CMakeLists.txt','ggml/src/ggml-cpu/CMakeLists.txt',
            'ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp','ggml/src/ggml-cpu/arch/x86/quants.c']}
        bad=subprocess.run(['python3',str(PATCHER),str(r),'--apply'], capture_output=True, text=True)
        assert bad.returncode==2
        assert 'missing function signature' in bad.stderr
        after={rel:(r/rel).read_bytes() for rel in before}
        assert before==after, 'failed patch attempt modified upstream source'
    print('PASS: patch dry-run/apply/check/idempotence/revert/transactionality')
if __name__=='__main__': main()
