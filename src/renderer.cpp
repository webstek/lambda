// ****************************************************************************
/// @file renderer.cpp
/// @author Kyle Webster
/// @version 0.1
/// @date 30 Nov 2025
/// @brief Renderer implementation
// ****************************************************************************
#include "renderer.hpp"
// ************************************
using namespace nl::cg;

void Renderer::render()
{
  const uint w = scene.cam.width; const uint h = scene.cam.height;
  image.init(w,h);

  uint px_idx = 0;
  for (uint i=0; i<h; i++) for (uint j=0; j<w; j++, px_idx++)
  {
    auto ray = sample::camera(scene.cam, {float(i+.5f),float(j+.5f),0.f,0.f});
    linRGB radiance = tracePath(ray);
    image.data[px_idx] = tosRGB(radiance);
  }
}

linRGB Renderer::tracePath(ray const &ray) const
{
  hitinfo h_info;
  if (intersect::scene(scene, ray, h_info))
  { // scattering
    return {1.f,1.f,1.f}; // white placeholder
  }
  return {0.f,0.f,0.f}; // scene not hit
}

void Renderer::loadScene(std::string fpath)
{
  load::loadNLS(scene, fpath);
}
void Renderer::saveImage(std::string fname) const
{
  lodepng::encode(
    fname, image.data[0].c.elem, image.width, image.height, LCT_RGB, 8);
}

// ****************************************************************************