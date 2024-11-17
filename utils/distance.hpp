#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>

#include <omp.h>
#include <immintrin.h> 

inline int brute_euclidean(const std::vector<float>& vec_a, const std::vector<float>& vec_b, const int& vector_dim){
    int dist = 0;

    #pragma omp simd reduction(+:dist)   
    for(int i = 0; i < vector_dim; ++i){
        dist += (int(vec_a[i]) - int(vec_b[i]))*(int(vec_a[i]) - int(vec_b[i]));
    }
    return dist;
} 

// for unaligned data
inline int fast_euclidean(const float* __restrict vec_a, const float* __restrict vec_b, const int& vector_dim) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;

    // Process in chunks of 8 floats
    for (; i + 8 <= vector_dim; i += 8) {
        if (i + 16 < vector_dim) {
            _mm_prefetch((const char*)(vec_a + i + 16), _MM_HINT_T0);
            _mm_prefetch((const char*)(vec_b + i + 16), _MM_HINT_T0);
        }

        __m256 a = _mm256_loadu_ps(vec_a + i);
        __m256 b = _mm256_loadu_ps(vec_b + i);
        __m256 diff = _mm256_sub_ps(a, b);
        __m256 sq_diff = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, sq_diff);
    }

    // Sum the vector of squared differences
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_vec);
    float dist = 0;
    for (int j = 0; j < 8; ++j) {
        dist += sum_array[j];
    }

    // Handle remaining elements (if not divisible by 8)
    for (; i < vector_dim; ++i) {
        float diff = vec_a[i] - vec_b[i];
        dist += diff * diff;
    }

    return dist;
}

// for aligned data 
static inline float _mm256_reduce_add_ps(__m256 x) {
  /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
  const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
  /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  /* Conversion to float is a no-op on x86-64 */
  return _mm_cvtss_f32(x32);
}

inline float compute_distance_squared(int dim, const float* __restrict__ a, const float* __restrict__ b) {
  a = (const float *)__builtin_assume_aligned(a, 32);
  b = (const float *)__builtin_assume_aligned(b, 32);

  uint16_t niters = (uint16_t)(dim / 8);
  __m256 sum = _mm256_setzero_ps();
  for (uint16_t j = 0; j < niters; j++) {
    if (j+1 < niters) {
      _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
    }
    __m256 a_vec = _mm256_load_ps(a + 8 * j);
    __m256 b_vec = _mm256_load_ps(b + 8 * j);
    __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
    sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
  }
  // horizontal add sum
  return _mm256_reduce_add_ps(sum);
}

