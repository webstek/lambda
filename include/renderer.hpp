// ****************************************************************************
/// @file renderer.hpp
/// @author Kyle Webster
/// @version 0.2
/// @date 02 Feb 2026
/// @brief Definition of renderer
// ****************************************************************************
#pragma once
// ** Includes ************************
#include <string>
#include <omp.h>
#include "LodePNG/lodepng.h"
#include "cg.hpp"
// ************************************

class Renderer
{
public:
  uint64_t MAX_SCATTERINGS = 64;
  uint64_t SPP   = 64;
  float SAMPLE_P = 0.95;
  std::string scene_path;
  nl::cg::image<nl::cg::rgb24> image;
  nl::cg::scene scene;

  void render();
  nl::cg::heroλ tracePath(
    nl::cg::heroλ const &λ, 
    nl::cg::ray const &ray, 
    nl::RNG &rng, 
    uint64_t scatters) const;

  void loadScene(std::string fpath);
  void saveImage() const;
};

// ****************************************************************************