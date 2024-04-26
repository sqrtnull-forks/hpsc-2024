#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  int iota[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    iota[i] = i;
  }
  for(int i=0; i<N; i++) {
    __m512 xv = _mm512_load_ps(x);
    __m512 yv = _mm512_load_ps(y);
    __m512 mv = _mm512_load_ps(m);
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 rx = _mm512_sub_ps(xi,xv);
    __m512 yi = _mm512_set1_ps(y[i]);
    __m512 ry = _mm512_sub_ps(yi,yv);
    __m512 r = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx),_mm512_mul_ps(ry, ry)));
    __mmask16 mask = _mm512_cmpneq_epi32_mask(_mm512_set1_epi32(i), _mm512_load_epi32(iota));
    __m512 rrr = _mm512_mul_ps(r, _mm512_mul_ps(r, r));
    fx[i]-=_mm512_mask_reduce_add_ps(mask,_mm512_mul_ps(_mm512_mul_ps(rx, mv),rrr));
    fy[i]-=_mm512_mask_reduce_add_ps(mask,_mm512_mul_ps(_mm512_mul_ps(ry, mv),rrr));

    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
