// ****************************************************************************
/// @file renderer.hpp
/// @author Kyle Webster
/// @version 0.1
/// @date 30 Nov 2025
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
  nl::cg::image<nl::cg::rgb24> image;
  nl::cg::scene scene;

  void render();
  nl::cg::linRGB tracePath(
    nl::cg::ray const &ray, nl::RNG &rng, uint64_t scatters) const;

  void loadScene(std::string fpath);
  void saveImage(std::string fname) const;
};

// ****************************************************************************