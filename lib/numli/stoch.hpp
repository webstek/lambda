// ****************************************************************************
/// @file stoch.hpp
/// @author Kyle Webster
/// @version 0.2
/// @date 6 Dec 2025
/// @brief Numerics Library - Stochastic @ref stoch
/// @details
/// Collection of utilities related to stochastic processes
// ****************************************************************************
#ifndef STOCH_HPP
#define STOCH_HPP
// ************************************
#include <cstdint>
#include <stdfloat>
#include <random>

#include "numli.hpp"
#include "bra.hpp"
// ************************************
namespace nl
{

/// @namespace stoch
/// @brief Stochastic process utilities
namespace stoch
{

/// @todo replace <random> with PCG rng
/// @todo Halton sequences
class RNG 
{
public:
  /// @name constructors
  RNG() : RNG(std::random_device{}()) {}
  RNG(uint64_t seed) : 
    state(seed), engine(seed), rng_float(0.f,1.f), rng_uint(0,UB<uint64_t>) {}
  
  /// @name member functions
  constexpr uint64_t uint64() {return rng_uint(engine);}
  constexpr float flt() {return rng_float(engine);}
private:
  /// @name members
  uint64_t state;
  std::mt19937 engine;
  std::uniform_real_distribution<float> rng_float;
  std::uniform_int_distribution<uint64_t> rng_uint;
};


// ************************************
/// @name sampling

constexpr ℝ3 UnifHemi(float x0, float x1)
{
  float const cosθ = 1.f-x0;
  float const sinθ = std::sqrtf(1.f-cosθ*cosθ);
  float const φ    = 2.f*π<float>*x1;
  return {sinθ*cosf(φ), sinθ*sinf(φ), cosθ};
}

constexpr ℝ3 CosHemi(float x0, float x1)
{
  float const cosθcosθ = 1.f-x0;
  float const cosθ = std::sqrtf(cosθcosθ);
  float const sinθ = std::sqrtf(1.f-cosθcosθ);
  float const φ    = 2.f*π<float>*x1;
  return {sinθ*cosf(φ), sinθ*sinf(φ), cosθ};
}
// ** end of sampling *****************

} // ** end of namespace nl::stoch ****

using RNG = stoch::RNG;
} // ** end of namespace nl ***********

// ****************************************************************************
#endif // #ifndef STOCH_HPP