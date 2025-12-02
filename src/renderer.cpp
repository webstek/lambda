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
using ℝ3 = nl::ℝ3;
// ************************************

void Renderer::render()
{
  const uint64_t w = scene.cam.width; const uint64_t h = scene.cam.height;
  image.init(w,h);

  #pragma omp parallel for collapse(2) schedule(runtime)
  for (uint64_t i=0; i<h; i++) for (uint64_t j=0; j<w; j++)
  {
    nl::RNG rng(i*w+j);
    linRGB radiance = 0.f;
    for (int k=0; k<128; k++)
    { // multiple samples per pixel
      sample::info<ray> si;
      nl::ℝ2 uv = {float(j+rng.flt()), float(i+rng.flt())};
      sample::camera(scene.cam, uv, si, rng);
      radiance += tracePath(si.val, rng);
    }
    radiance /= 128.f;
    image.data[(h-i-1)*w+j] = sRGB2rgb24(linRGB2sRGB(radiance));
  }
}
// ****************************************************************************

/// @todo Only handles direct lighting paths at the moment (DL by NEE) 
linRGB Renderer::tracePath(ray const &ray, nl::RNG &rng) const
{
  hitinfo hinfo;
  if (intersect::scene(scene, ray, hinfo))
  { // scattering
    // check if path hit a light (emitter material)
    Material const mat = scene.materials[hinfo.mat];
    if (std::holds_alternative<emitter>(mat)) 
      { return std::get<emitter>(mat).radiance; }

    // hit object, perform NEE
    ℝ3 const ωo = -ray.u.normalized();

    // sample lights in scene: 
    //   prob = p(li) - prob of choosing light
    //   val  = pointer to chosen light
    sample::info<Light const*> si_l;
    sample::lights(scene.lights, si_l, rng);

    // sample chosen light:
    //   prob = p(ωi|light)
    //   val  = ωi
    //   mult = L(ωi)
    //   weight = L(ωi)/p(ωi|li)
    sample::info<ℝ3,linRGB> si_ωi;
    sample::light(si_l.val, hinfo, scene, si_ωi, rng);

    // evaluate BSDFcosθ for ωi
    linRGB coef = BxDFcosθ(mat, si_ωi.val, ωo, hinfo.n);
    return si_ωi.mult * coef / (si_ωi.prob*si_l.prob);
  }
  return {0.f,0.f,0.f}; // scene not hit
}
// ****************************************************************************


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