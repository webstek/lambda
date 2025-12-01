// ****************************************************************************
/// @file bra.hpp
/// @author Kyle Webster
/// @version 0.5
/// @date 1 Dec 2025
/// @brief Numerics Library - Algebra @ref bra
/// @details
/// Collection of algebraic structures and algorithms
// ****************************************************************************
#ifndef BRA_HPP
#define BRA_HPP
// ** Includes ****************************************************************
#include <concepts>
#include <cstdint>
#include <stdfloat>

#include "numli.hpp"
#include "simd.hpp"
// ****************************************************************************

namespace nl 
{ // ** nl ****************************

/// @namespace bra
/// @brief Algebraic data structures and algorithms
/// @details
/// Contents:
///  * @ref operators
///  * @ref vector_spaces
namespace bra
{ // ** nl::bra ***********************

// ** concepts ************************
template <typename T> concept arithmetic = 
  std::integral<T> || std::floating_point<T>;

template <typename Op, typename T> concept binary_operator = 
  requires(Op op, T a, T b) { {op(a, b)} -> std::same_as<T>; };

template <typename Op, typename T> concept unary_operator = 
  requires(Op op, T a) { {op(a)} -> std::same_as<T>; };
// ** end of concepts ***************

// ****************************************************************************
/// @name operators
/// @brief structures representing operators
/// @warning For SIMD, only AVX2 +,-,*,/ for floats are defined

template<arithmetic T, simd::SIMD ISA>
struct Operator {};

// ** Binary Operators ****************
template<arithmetic T> struct op_add : Operator<T, simd::SIMD_MODE>
{ 
  static constexpr T serial(T a, T b) { return a+b; }
  static void SIMD(
    T       * restrict x, 
    T const * restrict a, 
    T const * restrict b, 
    size_t             n)
  {
    if constexpr (simd::SIMD_MODE == simd::AVX2)
    {
      if constexpr (std::same_as<T,float>) simd::add_avx2_float(x,a,b,n);
    }
  }
  constexpr T operator()(T a, T b) { serial(a,b); }
};

template<arithmetic T> struct op_sub : Operator<T, simd::SIMD_MODE>
{ 
  static constexpr T serial(T a, T b) { return a-b; }
  static void SIMD(
    T       * restrict x, 
    T const * restrict a, 
    T const * restrict b, 
    size_t             n)
  {
    if constexpr (simd::SIMD_MODE == simd::AVX2)
    {
      if constexpr (std::same_as<T,float>) simd::sub_avx2_float(x,a,b,n);
    }
  }
  constexpr T operator()(T a, T b) { serial(a,b); }
};

template<arithmetic T> struct op_mul : Operator<T, simd::SIMD_MODE>
{ 
  static constexpr T serial(T a, T b) { return a*b; }
  static void SIMD(
    T       * restrict x, 
    T const * restrict a, 
    T const * restrict b, 
    size_t             n)
  {
    if constexpr (simd::SIMD_MODE == simd::AVX2)
    {
      if constexpr (std::same_as<T,float>) simd::mul_avx2_float(x,a,b,n);
    }
  }
  constexpr T operator()(T a, T b) { serial(a,b); }
};

template<arithmetic T> struct op_div : Operator<T, simd::SIMD_MODE>
{ 
  static constexpr T serial(T a, T b) { return a/b; }
  static void SIMD(
    T       * restrict x, 
    T const * restrict a, 
    T const * restrict b, 
    size_t             n)
  {
    if constexpr (simd::SIMD_MODE == simd::AVX2)
    {
      if constexpr (std::same_as<T,float>) simd::div_avx2_float(x,a,b,n);
    }
  }
  constexpr T operator()(T a, T b) { serial(a,b); }
};

// ** Reductions **********************
template<arithmetic T, size_t n> 
struct op_inner_prod : Operator<T, simd::SIMD_MODE>
{
  static constexpr T serial(T const (&a)[n], T const (&b)[n])
    {T sum = T(0); for (int i=0; i<n; i++) sum += a[i]*b[i]; return sum;}
  /// @todo write SIMD inner product  
  static void SIMD(T &x, T const* restrict a, T const* restrict b);
  constexpr T operator()(T const (&a)[n], T (&b)[n]) { serial(a,b); }
};

// ****************************************************************************
/// @name vector spaces
/// @details
/// Contents:
///  * @ref ℝn - number in ℝⁿ
///  * @ref ℝnxm - number in ℝⁿˣᵐ, stored in row-major order

// ************************************
/// @name array operators

/// @brief A binary operation performed element-wise between arrays
/// @attention a, b, and x must not alias each other
template <uint32_t n, arithmetic T, binary_operator<T> Op>
constexpr void opBinaryArray(
  T       * restrict x,
  T const * restrict a,
  T const * restrict b)
{
  if constexpr (simd::SIMD_MODE!=simd::NO_SIMD && n > simd::simd<T>::lanes)
  {
    Op::SIMD(x, a, b, n);
  } else { for (size_t i=0;i<n;i++) {x[i] = Op::serial(a[i],b[i]);} }
}

/// @brief A unary operation performed element-wise on an array
/// @attention a and x must not alias each other
template <uint32_t n, arithmetic T, unary_operator<T> Op>
constexpr void opUnaryArray(T * restrict x, T const * restrict a)
{
  if constexpr (n > simd::simd<T>::lanes)
  {
    /// @todo SIMD unary operation
  } else { for (size_t i=0;i<n;i++) {x[i] = Op{}(a[i]);} }
}
// ** end of Array Operators **********

/// @struct ℝn
/// @brief a number in ℝⁿ
/// @tparam n number of basis elements that square to 1
/// @tparam T storage type, default float64_t
/// @attention T must be arithmetic
/// @details
/// Element of ℝⁿ with type T. Stored aligned to SIMD width if available.
template<uint32_t n, arithmetic T = std::float64_t> struct ℝn
{
  /// @name members
  alignas(simd::simd<T>::alignmentFor(n)) T elem[n]; ///< basis coefficients

  /// @name constructors
  constexpr explicit ℝn( T const restrict (&a)[n] ) { copy(elem, a); }
  constexpr explicit ℝn( ℝn<n,T> const &x ) { copy(elem, x.elem); }
  constexpr explicit ℝn( ℝn<n,T> const *x ) { copy(elem, x->elem); }
  constexpr explicit ℝn( T v ) { for (uint32_t i=0;i<n;i++) elem[i]=v; }
  constexpr explicit ℝn( ℝn<n-1,T> const &x, T v) 
    { for (uint32_t i=0;i<n-1;i++) elem[i]=x.elem[i]; elem[n-1]=v; }
  constexpr explicit ℝn( ℝn<n+1,T> const &x)
    { for (uint32_t i=0;i<n;i++) elem[i]=x.elem[i]; }
  constexpr ℝn(std::initializer_list<T> init)
  {
    auto it=init.begin(); 
    for (uint32_t i=0; i<n && it!=init.end(); i++,it++) {elem[i]=*it;}
  }
  template <binary_operator<T> Op> constexpr explicit ℝn(
    ℝn<n,T> const &x, ℝn<n,T> const &y, std::type_identity<Op>)
    { opBinaryArray<n,T,Op>(elem, x.elem, y.elem); }
  constexpr ℝn() {}

  /// @name member functions
  constexpr T l2() const 
  {
    T sum=T(0); 
    for(uint32_t i=0;i<n;i++) { sum+=elem[i]*elem[i]; }
    return std::sqrt(sum);
  }
  constexpr void normalize()
    { T len=l2(); assert(len!=T(0)); for (uint32_t i=0;i<n;i++) elem[i]/=len; }
  constexpr ℝn normalized() const {ℝn x(this);x.normalize();return ℝn(x.elem);}
  constexpr void negate() { for (uint32_t i=0;i<n;i++) elem[i]*=-1; }
  constexpr ℝn negated() const { ℝn x(this); x.negate(); return ℝn(x.elem); }

  /// @name operators
  constexpr T& operator[](size_t i)       noexcept { return elem[i]; }
  constexpr T  operator[](size_t i) const noexcept { return elem[i]; }

  constexpr ℝn& operator=(ℝn const &x) noexcept
  { if (&x!=this) copy(elem, x.elem); return *this; }
  constexpr ℝn& operator=(T const (&a)[n]) noexcept
    { copy(elem, a); return *this; }
  constexpr ℝn& operator=(T v) noexcept
    { for (uint32_t i=0;i<n;i++) elem[i]=v; return *this; }
};
// ** end ℝn **************************

/// @struct ℝnxm
/// @brief a number in ℝⁿˣᵐ
/// @tparam n number of rows
/// @tparam m number of columns
/// @tparam T storage type, default float64_t
/// @attention T must be arithmetic
/// @details
/// Element of ℝⁿˣᵐ with type T. Stored aligned to SIMD width if available.
template<uint32_t n, uint32_t m, arithmetic T = std::float64_t> 
struct ℝnxm
{
  /// @name members
  static constexpr uint64_t N = n*m;                 ///< number of elements
  alignas(simd::simd<T>::alignmentFor(N)) T elem[N]; ///< basis coefficients

  /// @name constructors
  constexpr explicit ℝnxm( T const restrict (&a)[N] ) { copy(elem, a); }
  constexpr explicit ℝnxm( ℝnxm<n,m,T> const &x ) { copy(elem,x.elem); }
  constexpr explicit ℝnxm( ℝnxm<n,m,T> const *x ) { copy(elem,x->elem); }
  constexpr explicit ℝnxm( T v ) { for (uint64_t k=0;k<N;k++) elem[k]=v; }
  constexpr ℝnxm(std::initializer_list<T> init)
  {
    auto it=init.begin(); 
    for (uint32_t k=0; k<N && it!=init.end(); k++,it++) {elem[k]=*it;}
  }
  ℝnxm() {}

  /// @name member functions
  constexpr void identity() 
  {
    for (uint64_t i=0;i<n;i++) for (uint64_t j=0;j<m;j++) 
      if (i==j) {elem[i*m+j]=T(1);} else {elem[i*m+j]=T(0);}
  }
  constexpr void negate() { for (uint64_t k=0;k<N;k++) elem[k]*=-1; }
  constexpr ℝnxm negated() const {ℝnxm x(this);x.negate();return ℝnxm(x.elem);}

  /// @name operators
  constexpr T& operator()(size_t i, size_t j) noexcept 
    { return elem[i*m+j]; }
  constexpr T  operator()(size_t i, size_t j) const noexcept
    { return elem[i*m+j]; }
  constexpr ℝnxm& operator=(ℝnxm const &A) noexcept
    { if (&A!=this) copy(elem, A.elem); return *this; }
  constexpr ℝnxm& operator=(T const (&a)[N]) noexcept
    { copy(elem, a); return *this; }
  constexpr ℝnxm& operator=(T v) noexcept
    { for (uint64_t k=0;k<N;k++) elem[k]=v; return *this; }
};
// ** end ℝnxm ************************

// ** Operators on ℝn and ℝnxm ********

/// @name unary operators
template<uint32_t n, uint32_t m, arithmetic T>
constexpr ℝnxm<n,m,T> operator-(ℝnxm<n,m,T> const &x) { return x.negated(); }
template<uint32_t n, arithmetic T>
constexpr ℝn<n,T> operator-(ℝn<n,T> const &x) { return x.negated(); }

/// @name binary operators
template<uint32_t n, arithmetic T>
constexpr ℝn<n,T> operator+(ℝn<n,T> const &x, ℝn<n,T> const &y)
  { return ℝn<n,T>(x,y,std::type_identity<op_add<T>>{}); }
template<uint32_t n, arithmetic T>
constexpr ℝn<n,T> operator-(ℝn<n,T> const &x, ℝn<n,T> const &y)
  { return ℝn<n,T>(x,y,std::type_identity<op_sub<T>>{}); }
template<uint32_t n, arithmetic T>
constexpr ℝn<n,T> operator*(ℝn<n,T> const &x, ℝn<n,T> const &y)
  { return ℝn<n,T>(x,y,std::type_identity<op_mul<T>>{}); }
template<uint32_t n, arithmetic T>
constexpr ℝn<n,T> operator/(ℝn<n,T> const &x, ℝn<n,T> const &y)
  { return ℝn<n,T>(x,y,std::type_identity<op_div<T>>{}); }

/// @brief Matrix-Matrix multiplication
/// @todo simd matrix-matrix multiplication
template<uint32_t n, uint32_t K, uint32_t m, arithmetic T>
constexpr ℝnxm<n,m,T> operator*(ℝnxm<n,K,T> const &A, ℝnxm<K,m,T> const &B)
{
  T vals[n*m];
  for (uint32_t i=0;i<n;i++) for (uint32_t j=0;j<m;j++) 
  {
    T sum=0.f;
    for (uint32_t k=0;k<K;k++) { sum += A.elem[i*K+k]*B.elem[k*m+j]; }
    vals[i*m+j] = sum;
  }
  return ℝnxm<n,m,T>(vals);
}

/// @brief Matrix-Vector multiplication
/// @todo simd matrix-vector multiplication
template<uint32_t n, uint32_t m, arithmetic T>
constexpr ℝn<n,T> operator*(ℝnxm<n,m,T> const &A, ℝn<m,T> const &b)
{
  T vals[n];
  for (uint32_t i=0;i<n;i++)
  {
    T sum=0.f;
    for (uint32_t k=0;k<m;k++) { sum += A.elem[i*m+k]*b[k]; }
    vals[i] = sum;
  }
  return ℝn<n,T>(vals);
}

/// @brief Vector-Scalar multiplication
template<uint32_t n, arithmetic T>
constexpr ℝn<n,T> operator*(T s, ℝn<n,T> const &a)
{
  T vals[n]; 
  for (uint32_t i=0;i<n;i++) { vals[i]=a.elem[i]*s; }
  return ℝn<n,T>(vals);
}
template<uint32_t n, arithmetic T>
constexpr ℝn<n,T> operator*(ℝn<n,T> const &a, T s)
{
  T vals[n]; 
  for (uint32_t i=0;i<n;i++) { vals[i]=a.elem[i]*s; }
  return ℝn<n,T>(vals);
}

/// @brief cross product in ℝ3
template<arithmetic T>
constexpr ℝn<3,T> operator^(ℝn<3,T> const &x, ℝn<3,T> const &y)
  { return {x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]}; }
/// @brief dot product in ℝ3
template<arithmetic T>
constexpr T operator|(ℝn<3,T> const &x, ℝn<3,T> const &y)
  { return x[0]*y[0]+x[1]*y[1]+x[2]*y[2]; }


// ** end operators on ℝn and ℝnxm ****

// ************************************
/// @name functions on ℝn and ℝnxm

/// @brief Subset of column j of length l of matrix A starting at i0
template<uint32_t l, uint32_t n, uint32_t m, arithmetic T>
constexpr ℝn<l,T> column(ℝnxm<n,m,T> const &A, uint32_t j, uint32_t i0=0)
{
  assert(j<m); assert(i0<n); assert(l<=n-i0);
  T col[l];
  for (uint32_t i=i0;i<i0+l;i++) { col[i] = A(i,j); }
  return ℝn<l,T>(col);
}

/// @todo row subset
/// @todo sub matrix

/// @brief Transpose of sub matrix of A of size (l,k) starting at (i,j)
template<uint32_t l, uint32_t k, uint32_t n, uint32_t m, arithmetic T>
constexpr ℝnxm<l,k,T> subMatT(ℝnxm<n,m,T> const &A,uint32_t i0=0,uint32_t j0=0)
{
  assert(i0<n); assert(j0<m); assert(l<=n-i0); assert(k<=m-j0);
  T vals[l*k];
  uint64_t idx=0;
  for (uint32_t i=0;i<l;i++) for (uint32_t j=0;j<k;j++,idx++)
    { vals[idx] = A(i0+i,j0+j); }
  return ℝnxm<l,k,T>(vals);
}

/// @brief square matrix inverse
template<uint32_t n, arithmetic T> 
constexpr ℝnxm<n,n,T> inverse(ℝnxm<n,n,T> const &A);

/// @brief 3x3 matrix inverse specialization
template<arithmetic T>
constexpr ℝnxm<3,3,T> inverse(ℝnxm<3,3,T> const &A)
{
  // Compute 2x2 minors
  T m00 = A(1,1)*A(2,2) - A(1,2)*A(2,1);
  T m01 = A(1,0)*A(2,2) - A(1,2)*A(2,0);
  T m02 = A(1,0)*A(2,1) - A(1,1)*A(2,0);
  T m10 = A(0,1)*A(2,2) - A(0,2)*A(2,1);
  T m11 = A(0,0)*A(2,2) - A(0,2)*A(2,0);
  T m12 = A(0,0)*A(2,1) - A(0,1)*A(2,0);
  T m20 = A(0,1)*A(1,2) - A(0,2)*A(1,1);
  T m21 = A(0,0)*A(1,2) - A(0,2)*A(1,0);
  T m22 = A(0,0)*A(1,1) - A(0,1)*A(1,0);
  
  // Compute determinant
  T det = A(0,0)*m00 - A(0,1)*m01 + A(0,2)*m02;
  
  // Compute inverse using adjugate/determinant
  T vals[9];
  T inv_det = T(1) / det;
  vals[0] =  m00 * inv_det;
  vals[1] = -m10 * inv_det;
  vals[2] =  m20 * inv_det;
  vals[3] = -m01 * inv_det;
  vals[4] =  m11 * inv_det;
  vals[5] = -m21 * inv_det;
  vals[6] =  m02 * inv_det;
  vals[7] = -m12 * inv_det;
  vals[8] =  m22 * inv_det;
  return ℝnxm<3,3,T>(vals);
}

// ** end of Vector Spaces ****************************************************
} // ** end of namespace bra **********

// ************************************
/// @name aliases
using ℤ2p  = bra::ℝn<2,uint64_t>;
using ℝ2   = bra::ℝn<2,float>;
using ℝ3   = bra::ℝn<3,float>;
using ℝ4   = bra::ℝn<4,float>;
using ℝ3x3 = bra::ℝnxm<3,3,float>;
using ℝ4x4 = bra::ℝnxm<4,4,float>;

} // ** end of namespace nl ***********

// ****************************************************************************
#endif // #ifndef BRA_HPP