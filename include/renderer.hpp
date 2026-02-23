// ****************************************************************************
/// @file renderer.hpp
/// @author Kyle Webster
/// @version 0.3
/// @date 22 Feb 2026
/// @brief Definition of renderer
// ****************************************************************************
#pragma once
// ** Includes ************************
#include <string>
#include <omp.h>
#include "LodePNG/lodepng.h"
#include "cg.hpp"
// ************************************

struct rendering
{
  nl::cg::image<nl::cg::linRGB> img;

  std::vector<nl::cg::rgb24> rgb24() const
  { // outputs flat vector in 24bit rgb format with gamma correction
    std::vector<nl::cg::rgb24> out;
    for (size_t j=0; j<img.height; j++) for (size_t i=0; i<img.width; i++)
    {
      out.push_back(
        nl::cg::sRGB2rgb24(nl::cg::linRGB2sRGB(img.data[j*img.width+i])));
    }
    return out;
  }
};

class Renderer
{
public:
  uint64_t MAX_SCATTERINGS = 4;
  uint64_t SPP   = 4;
  float SAMPLE_P = 0.5;

  void render(nl::cg::scene const &scene, rendering &buffer);
  void toneMap(
    rendering const &buffer, rendering &tm_buffer, float Y_MID=0.12f) const;

  nl::cg::heroλ tracePath(
    nl::cg::scene const &scene,
    nl::cg::heroλ const &λ, 
    nl::cg::ray const &ray, 
    nl::RNG &rng, 
    uint64_t scatters) const;

  void saveImage(rendering const &buffer, std::string fpath) const;
};

// ****************************************************************************