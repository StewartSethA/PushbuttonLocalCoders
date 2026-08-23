#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK 32
#define PF_DIST 2

typedef struct { float d; uint8_t qs[16]; } block_q4;
typedef struct { float d; int8_t qs[32]; } block_q8;

#define MM256_SET_M128I(a,b) _mm256_insertf128_si256(_mm256_castsi128_si256(b),(a),1)
static inline __m256i bytes_from_nibbles_32(const uint8_t *rsi) {
    const __m128i tmp=_mm_loadu_si128((const __m128i*)rsi);
    const __m256i bytes=MM256_SET_M128I(_mm_srli_epi16(tmp,4),tmp);
    return _mm256_and_si256(_mm256_set1_epi8(0xF),bytes);
}
static inline __m256 sum_i16_pairs_float(__m256i x) {
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1),x));
}
static inline __m256 mul_sum_i8_pairs_float(__m256i x,__m256i y) {
    const __m256i ax=_mm256_sign_epi8(x,x);
    const __m256i sy=_mm256_sign_epi8(y,x);
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax,sy));
}
static inline float hsum8(__m256 x) {
    __m128 r=_mm256_extractf128_ps(x,1); r=_mm_add_ps(r,_mm256_castps256_ps128(x));
    r=_mm_add_ps(r,_mm_movehl_ps(r,r)); r=_mm_add_ss(r,_mm_movehdup_ps(r)); return _mm_cvtss_f32(r);
}

float baseline(const block_q4*x,const block_q8*y,int nb) {
    __m256 acc=_mm256_setzero_ps();
    for(int i=0;i<nb;i++) {
        __m256 d=_mm256_set1_ps(x[i].d*y[i].d);
        __m256i qx=bytes_from_nibbles_32(x[i].qs);
        qx=_mm256_sub_epi8(qx,_mm256_set1_epi8(8));
        __m256i qy=_mm256_loadu_si256((const __m256i*)y[i].qs);
        acc=_mm256_fmadd_ps(d,mul_sum_i8_pairs_float(qx,qy),acc);
    }
    return hsum8(acc);
}

#define STEP(IDX,ACC) do { \
    __m256 d=_mm256_set1_ps(x[IDX].d*y[IDX].d); \
    __m256i qx=bytes_from_nibbles_32(x[IDX].qs); \
    qx=_mm256_sub_epi8(qx,_mm256_set1_epi8(8)); \
    __m256i qy=_mm256_loadu_si256((const __m256i*)y[IDX].qs); \
    ACC=_mm256_fmadd_ps(d,mul_sum_i8_pairs_float(qx,qy),ACC); \
} while(0)

float knl_unrolled(const block_q4*x,const block_q8*y,int nb) {
    int i=0; __m256 a0=_mm256_setzero_ps(),a1=a0,a2=a0,a3=a0;
    for(;i+3<nb;i+=4) {
        int p=i+PF_DIST;
        if(p<nb){_mm_prefetch((const char*)&x[p],_MM_HINT_T0);_mm_prefetch((const char*)&y[p],_MM_HINT_T0);}
        STEP(i+0,a0); STEP(i+1,a1); STEP(i+2,a2); STEP(i+3,a3);
    }
    __m256 acc=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
    for(;i<nb;i++) STEP(i,acc);
    return hsum8(acc);
}

static uint64_t rng=0x123456789abcdefULL;
static uint32_t r32(void){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return (uint32_t)rng;}
static double now_s(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(void){
    const int nb=8192; block_q4 *x=aligned_alloc(64,sizeof(*x)*nb); block_q8*y=aligned_alloc(64,sizeof(*y)*nb);
    if(!x||!y)return 2;
    for(int i=0;i<nb;i++){x[i].d=((int)(r32()%2001)-1000)/1000.0f;y[i].d=((int)(r32()%2001)-1000)/1000.0f;for(int j=0;j<16;j++)x[i].qs[j]=r32();for(int j=0;j<32;j++)y[i].qs[j]=(int8_t)r32();}
    float a=baseline(x,y,nb),b=knl_unrolled(x,y,nb); float err=fabsf(a-b),tol=1e-3f*fmaxf(1.0f,fabsf(a));
    printf("baseline=%.9g knl_unrolled=%.9g abs_err=%.6g tol=%.6g %s\n",a,b,err,tol,err<=tol?"PASS":"FAIL");
    volatile float sink=0; int reps=400; double t0=now_s();for(int r=0;r<reps;r++)sink+=baseline(x,y,nb);double t1=now_s();for(int r=0;r<reps;r++)sink+=knl_unrolled(x,y,nb);double t2=now_s();
    printf("native_probe baseline_ms=%.4f unrolled_ms=%.4f speedup=%.3fx sink=%g\n",1000*(t1-t0)/reps,1000*(t2-t1)/reps,(t1-t0)/(t2-t1),(double)sink);
    free(x);free(y);return err<=tol?0:1;
}
