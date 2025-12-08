// ****************************************************************************
/// @file renderer.cpp
/// @author Kyle Webster
/// @version 0.3
/// @date 3 Dec 2025
/// @brief Renderer implementation
// ****************************************************************************
#include "renderer.hpp"
// ************************************
using namespace nl::cg;
using ℝ3 = nl::ℝ3;
// ************************************
constexpr uint64_t MAX_SCATTERINGS = 128;
constexpr uint64_t SPP   = 8192;
constexpr float SAMPLE_P = 0.95;
// ************************************

void Renderer::render()
{
  const uint64_t w = scene.cam.width; const uint64_t h = scene.cam.height;
  image.init(w,h);

  std::vector<linRGB> irradiance(w*h,0.f);

  #ifndef DEBUG 
  #pragma omp parallel
  #endif
  { // ** begin parallel region ***************************

  #ifndef DEBUG
  #pragma omp for collapse(2) schedule(runtime)
  #endif
  for (uint64_t i=0;i<h;i++) for (uint64_t j=0;j<w;j++)
  { // compute pixel
    nl::RNG rng(i*w+j);
    linRGB irrad_acc = 0.f;
    for (uint64_t k=0;k<SPP;k++)
    { // trace a single path
      sample::info<ray> si;
      nl::ℝ2 const uv = {float(j+rng.flt()), float(i+rng.flt())};
      sample::camera(scene.cam, uv, si, rng);
      irrad_acc += tracePath(si.val, rng, 0);
    }
    irradiance[i*w+j] = irrad_acc;
  }

  #ifndef DEBUG
  #pragma omp for schedule(static, 16)
  #endif
  for (uint64_t i=0;i<h;i++) for (uint64_t j=0;j<w;j++)
  { // write irradiance to image buffer
    image.data[(h-i-1)*w+j] = sRGB2rgb24(linRGB2sRGB(irradiance[i*w+j]/SPP));
  }
  } // ** end of parallel region **************************
}
// ****************************************************************************

/// @todo Only handles direct lighting paths at the moment (DL by NEE) 
linRGB Renderer::tracePath(ray const &r, nl::RNG &rng, uint64_t scatters) const
{
  hitinfo hinfo;
  if (!intersect::scene(scene, r, hinfo)) { return {0.f,0.f,0.f}; } // misses
  
  ℝ3 const o = -r.u.normalized();
  ℝ3 const n = hinfo.n();

  // check if path hit a light (emitter material)
  Material const &mat = scene.materials[hinfo.mat];
  if (std::holds_alternative<emitter>(mat)) 
    { return std::get<emitter>(mat).Radiance(o,hinfo); }

  // hit an object, scatter if less than max scattering
  if (scatters > MAX_SCATTERINGS) return {0.f,0.f,0.f};

  // ** Light IS estimate *******************************
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
  sample::info<ℝ3,linRGB> si_i_L;
  sample::light(si_l.val, hinfo, scene, si_i_L, rng);

  // evaluate BSDFcosθ for ωi
  linRGB const coef = BxDFcosθ(mat, si_i_L.val, o, n, hinfo.front);

  // Light IS estimate
  linRGB const L_IS = si_i_L.mult * coef / (si_i_L.prob*si_l.prob);
  // ** end of L IS estimate ****************************

  // ** Material IS estimate ****************************
  // sample material:
  //   prob = p(ωi)
  //   val  = ωi
  //   mult = BxDFcosθ
  //   weight = BxDFcosθ/p(ωi)
  sample::info<ℝ3,linRGB> si_i_mat;
  bool const sample = sample::materiali(&mat,hinfo,o,si_i_mat,rng,SAMPLE_P);
  
  // no material sample generated, use light IS estimate
  if (!sample) { return L_IS; }

  // evaluate L(ωi)
  ray const i_ray = {hinfo.p, si_i_mat.val};
  linRGB Li = tracePath(i_ray, rng, scatters+1);
  
  // Material IS estimate
  // #ifdef DEBUG
    linRGB const M_IS = Li*si_i_mat.mult / si_i_mat.prob;
  // #else
  //   linRGB const M_IS = Li*si_i_mat.weight;
  // #endif
  // ** end of Material IS estimate *********************

  // ** MIS estimate ************************************
  float const p_L_mati = sample::probForLight(si_l.val, i_ray)*si_l.prob;
  float const p_mat_Li = 
    sample::probForMateriali(&mat, hinfo, si_i_L.val, o, SAMPLE_P);

  // power heuristic weights, β=2
  float const p_L_i = si_i_L.prob*si_l.prob;
  float const w_L = p_L_i*p_L_i / (p_L_i*p_L_i + p_mat_Li*p_mat_Li);
  float const w_mat = si_i_mat.prob*si_i_mat.prob 
    / (si_i_mat.prob*si_i_mat.prob + p_L_mati*p_L_mati);

  return M_IS*w_mat + L_IS*w_L;
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