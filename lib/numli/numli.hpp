// ****************************************************************************
/// @file numli.hpp
/// @author Kyle Webster
/// @version 0.4
/// @date 6 Dec 2025
/// @brief Numerics Library - @ref nl
/// @details
/// Collection of constants and core numerics
// ****************************************************************************
#ifndef NUMLI_HPP
#define NUMLI_HPP
// ** Includes ****************************************************************
#include <numbers>
#include <stdfloat>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <concepts>
// ****************************************************************************

// ************************************
/// @namespace nl
/// @brief Numerics Library main module
/// @details
/// Contents:
///  * @ref compiler - compatibility for MSC and GCC
///  * @ref utilities - language utilities
///  * @ref math - basic math constants and functions
namespace nl
{
  // **************************************************************************
  /// @name compiler
  /// @brief defines macros depending on which compiler is being used

  // ** restrict **********************
  #if defined(__GNUC__)
  #define restrict __restrict__
  #elif defined(_MSC_VER)
  #define restrict __restrict
  #else
  #define restrict
  #endif
  // ** end of compiler *******************************************************

  // **************************************************************************
  /// @name math
  /// @brief basic mathematical constants and functions

  // ** constants *********************
  template<typename T> constexpr T π     = std::numbers::pi_v<T>;
  template<typename T> constexpr T π²    = 
    std::numbers::pi_v<T>*std::numbers::pi_v<T>;
  template<typename T> constexpr T π_2   = std::numbers::pi_v<T>/2;
  template<typename T> constexpr T inv_π = 1/std::numbers::pi_v<T>; 
  template<typename T> constexpr T e     = std::numbers::e_v<T>;
  template<typename T> constexpr T ε     = 
    std::numeric_limits<T>::epsilon();
  // **********************************

  
  // ** functions *********************
  template<std::totally_ordered T> constexpr T min(T a, T b) {return a<=b?a:b;}
  template<std::totally_ordered T> constexpr T max(T a, T b) {return a>=b?a:b;}
  template<std::totally_ordered T> constexpr T clamp(T a, T inf, T sup)
    { return min(max(inf,a),sup); }
  template<typename T> constexpr T deg2rad(T deg) {return deg*π<T>/T(180);}
  template<typename T> constexpr T UB = std::numeric_limits<T>::max();
  constexpr float constexprSqrt(float x, float x1=1., float x0=0.f)
  { return x1 == x0 ? x1 : constexprSqrt(x, 0.5f*(x1+x/x1), x1); }
  // **********************************
  // ** end of math ***********************************************************

  // **************************************************************************
  /// @name utilities
  /// @brief common shorthand functions and misc items

  // ** copy *********************
  template <std::size_t n, typename T>
  constexpr void copy_impl(T (&dst)[n], T const (&src)[n], std::true_type)
    { for (std::size_t i=0; i<n; ++i) {dst[i] = src[i];} }
  template <std::size_t n, typename T>
  void copy_impl(T (&dst)[n], T const (&src)[n], std::false_type)
    { std::memcpy(dst, src, n * sizeof(T)); }
  /// @brief copies from src to dst optimized for constexpr or runtime call
  /// @tparam T type to copy
  /// @tparam n number of elements of T to copy
  /// @param dst destination array
  /// @param src source array
  template <std::size_t n, typename T>
  constexpr void copy(T (&dst)[n], T const (&src)[n])
  {
    copy_impl(dst, src, 
      std::integral_constant<bool, std::is_constant_evaluated()>{});
  }
  // **********************************

  // ** overload **********************
  /// @brief pass to std::visit with functions to resolve the std::variant
  template<class... Fs> struct Overload : Fs... { using Fs::operator()...; };
  template<class... Fs> Overload(Fs...) -> Overload<Fs...>;
  // **********************************

  // ** conversions *******************
  constexpr uint8_t float2byte(float f) 
    { return static_cast<uint8_t>(clamp(int(255*f+0.5f),0,255)); }
  // **********************************

  // ** end of utilities ******************************************************
} // ** end of namespace nl ***********

// ****************************************************************************
#endif // #ifndef NUMLI_HPP