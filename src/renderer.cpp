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
    for (int k=0; k<64; k++)
    { // multiple samples per pixel
      sample::info<ray> si;
      nl::ℝ2 uv = {float(j+rng.flt()), float(i+rng.flt())};
      sample::camera(scene.cam, uv, si, rng);
      radiance += tracePath(si.val, rng, 0);
    }
    radiance /= 64.f;
    image.data[(h-i-1)*w+j] = sRGB2rgb24(linRGB2sRGB(radiance));
  }
}
// ****************************************************************************

/// @todo Only handles direct lighting paths at the moment (DL by NEE) 
linRGB Renderer::tracePath(ray const &r, nl::RNG &rng, int scatters) const
{
  hitinfo hinfo;
  if (intersect::scene(scene, r, hinfo))
  {
    // check if path hit a light (emitter material)
    Material const &mat = scene.materials[hinfo.mat];
    if (std::holds_alternative<emitter>(mat)) 
      { return std::get<emitter>(mat).radiance; }

    // hit an object, scatter if less than max scattering
    if (scatters > 16) return {0.f,0.f,0.f};
    ℝ3 const o = -r.u.normalized();

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
    linRGB const coef = BxDFcosθ(mat, si_i_L.val, o, hinfo.n);

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
    bool const mat_sample = sample::materiali(&mat, hinfo, o, si_i_mat, rng);
    
    // no material sample generated, use light IS estimate
    if (!mat_sample) { return L_IS; }

    // evaluate L(ωi)
    ray const i_ray = {hinfo.p, si_i_mat.val};
    linRGB Li = tracePath(i_ray, rng, scatters+1);
    
    // Material IS estimate
    #ifdef DEBUG
      linRGB const M_IS = Li*si_i_mat.mult / si_i_mat.prob;
    #else
      linRGB const M_IS = Li*si_i_mat.weight;
    #endif
    // ** end of Material IS estimate *********************

    // ** MIS estimate ************************************
    float const p_L_mati = sample::probForLight(si_l.val, i_ray)*si_l.prob;
    float const p_mat_Li = sample::probForMateriali(&mat, hinfo, si_i_L.val);

    // power heuristic weights, β=2
    float const p_L_i = si_i_L.prob*si_l.prob;
    float const w_L = p_L_i*p_L_i / (p_L_i*p_L_i + p_mat_Li*p_mat_Li);
    float const w_mat = si_i_mat.prob*si_i_mat.prob 
      / (si_i_mat.prob*si_i_mat.prob + p_L_mati*p_L_mati);

    return M_IS*w_mat + L_IS*w_L;
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