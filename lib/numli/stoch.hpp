// ****************************************************************************
/// @file stoch.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 1 Dec 2025
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
} // ** end of namespace nl::stoch ****

using RNG = stoch::RNG;
} // ** end of namespace nl ***********

// ****************************************************************************
#endif // #ifndef STOCH_HPP