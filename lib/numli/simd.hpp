// ****************************************************************************
/// @file numli.hpp
/// @author Kyle Webster
/// @version 0.3
/// @date 1 Dec 2025
/// @brief Numerics Library - SIMD @ref simd
/// @details
/// Collection of functions and utilities for SIMD use
// ****************************************************************************
#ifndef SIMD_HPP
#define SIMD_HPP
// ** Includes ****************************************************************
#include "immintrin.h"

#include "numli.hpp"
// ****************************************************************************

namespace nl
{ // ** nl ****************************

/// @namespace simd
/// @brief SIMD types and algorithms
/// @details
/// Contents:
//// * @ref data
///  * @ref functions
namespace simd
{ // ** nl::simd **********************

  // **************************************************************************
  /// @name data
  /// @brief simd data types

  // ** SIMD support ******************
  enum SIMD {NO_SIMD = 0, AVX2 = 1, AVX512 = 2};
  #if defined(__AVX512__)
    constexpr SIMD        SIMD_MODE  = AVX512;
    constexpr std::size_t SIMD_WIDTH = 64;
  #elif defined(__AVX2__)
    constexpr SIMD        SIMD_MODE  = AVX2;
    constexpr std::size_t SIMD_WIDTH = 32;
  #else
    constexpr SIMD        SIMD_MODE  = NO_SIMD;
    constexpr std::size_t SIMD_WIDTH = 0;
  #endif

  #define TARGET_AVX2 __attribute__((target("avx2")))
  // **********************************
  /// @struct simd
  /// @brief simd information
  /// @tparam T storage type for alignment if SIMD is unavailable
  template<typename T> struct simd
  {
    static constexpr size_t alignment      ///< simd data alignment
      = SIMD_MODE!=NO_SIMD ? SIMD_WIDTH : sizeof(T);
    static constexpr size_t alignmentFor(size_t n)
      { return n>lanes ? alignment : sizeof(T); }
    static constexpr std::size_t lanes     ///< number of lanes for type T
      = SIMD_WIDTH/sizeof(T);
  };
  // ** end of data ***********************************************************


  // **************************************************************************
  /// @name functions
  /// @brief simd functions
  /// @warning assumes alignment with SIMD register size
  /// @details
  /// Contents:
  ///  * @ref AVX2

  // **********************************
  /// @name AVX2
  /// @brief AVX2 operations on arrays
  /// @details
  /// Contents:
  ///  * @ref add_avx2_float
  ///  * @ref add_avx2_double
  ///  * @ref sub_avx2_float
  ///  * @ref mul_avx2_float
  ///  * @ref div_avx2_float

  /// @brief avx2 float array add
  TARGET_AVX2 inline void add_avx2_float(
    float       * restrict x,
    float const * restrict a,
    float const * restrict b,
    size_t                 n)
  {
    constexpr size_t l = 8;
    for (size_t i=0; i+l<=n; i+=l)
    {
      __m256 va = _mm256_load_ps(a+i);
      __m256 vb = _mm256_load_ps(b+i);
      __m256 vx = _mm256_add_ps(va, vb);
      _mm256_store_ps(x+i, vx);
    }
    for (size_t j=n-n%l; j<n; j++) { x[j] = a[j]+b[j]; }
  }
  /// @brief avx2 double array add
  TARGET_AVX2 inline void add_avx2_double(
    double       * restrict x,
    double const * restrict a,
    double const * restrict b,
    size_t                  n)
  {
    constexpr size_t l = 4;
    for (size_t i=0; i+l<=n; i+=l)
    {
      __m256d va = _mm256_load_pd(a+i);
      __m256d vb = _mm256_load_pd(b+i);
      __m256d vx = _mm256_add_pd(va, vb);
      _mm256_store_pd(x+i, vx);
    }
    for (size_t j=n-n%l; j<n; j++) { x[j] = a[j]+b[j]; }
  }
  /// @brief avx2 float array minus
  TARGET_AVX2 inline void sub_avx2_float(
    float       * restrict x,
    float const * restrict a,
    float const * restrict b,
    size_t                 n)
  {
    constexpr size_t l = 8;
    for (size_t i=0; i+l<=n; i+=l)
    {
      __m256 va = _mm256_load_ps(a+i);
      __m256 vb = _mm256_load_ps(b+i);
      __m256 vx = _mm256_sub_ps(va, vb);
      _mm256_store_ps(x+i, vx);
    }
    for (size_t j=n-n%l; j<n; j++) { x[j] = a[j]-b[j]; }
  }
  /// @brief avx2 float array multiply
  TARGET_AVX2 inline void mul_avx2_float(
    float       * restrict x,
    float const * restrict a,
    float const * restrict b,
    size_t                 n)
  {
    constexpr size_t l = 8;
    for (size_t i=0; i+l<=n; i+=l)
    {
      __m256 va = _mm256_load_ps(a+i);
      __m256 vb = _mm256_load_ps(b+i);
      __m256 vx = _mm256_mul_ps(va, vb);
      _mm256_store_ps(x+i, vx);
    }
    for (size_t j=n-n%l; j<n; j++) { x[j] = a[j]*b[j]; }
  }
  /// @brief avx2 float array divide
  TARGET_AVX2 inline void div_avx2_float(
    float       * restrict x,
    float const * restrict a,
    float const * restrict b,
    size_t                 n)
  {
    constexpr size_t l = 8;
    for (size_t i=0; i+l<=n; i+=l)
    {
      __m256 va = _mm256_load_ps(a+i);
      __m256 vb = _mm256_load_ps(b+i);
      __m256 vx = _mm256_div_ps(va, vb);
      _mm256_store_ps(x+i, vx);
    }
    for (size_t j=n-n%l; j<n; j++) { x[j] = a[j]/b[j]; }
  }
  // ** end of AVX2 *******************
}
}

// ****************************************************************************
#endif // #ifndef SIMD_HPP