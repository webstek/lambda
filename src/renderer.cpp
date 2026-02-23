// ****************************************************************************
/// @file renderer.cpp
/// @author Kyle Webster
/// @version 0.6
/// @date 22 Feb 2026
/// @brief Renderer implementation
// ****************************************************************************
#include "renderer.hpp"
// ************************************
using namespace nl::cg;
using ℝ3 = nl::ℝ3;
// ************************************

void Renderer::render(scene const &scene, rendering &buffer)
{
  uint64_t const w = scene.cam.width; const uint64_t h = scene.cam.height;
  buffer.img.init(w,h);

  std::vector<nl::cg::coefficientλ<nl::cg::Nλ>> irradiance(w*h,0.f);
  nl::cg::coefficientλ<nl::cg::Nλ> sample_spec(0.f);

  #ifndef DEBUG
  #pragma omp parallel
  { // ** begin parallel region ***************************
  #pragma omp for collapse(2) schedule(runtime)
  #endif
  for (uint64_t i=0;i<h;i++) for (uint64_t j=0;j<w;j++)
  { // compute pixel
    nl::RNG rng(i*w+j);
    nl::cg::coefficientλ<nl::cg::Nλ> irrad_acc(0.f);
    for (uint64_t k=0;k<SPP;k++)
    { // trace a single path
      // ray sample info
      sample::info<ray> si;
      nl::ℝ2 const uv = {float(j+rng.flt()), float(i+rng.flt())};
      sample::camera(scene.cam, uv, si, rng);

      // wavelength sample info
      sample::info<heroλ> si_λ;
      sample::spectrum(sample_spec, si_λ, rng);

      // compute Li along generated ray, add to irradiance
      heroλ const Li = tracePath(scene, si_λ.val, si.val, rng, 0);
      irrad_acc.addHeroλ(si_λ.val, Li/si_λ.prob);
    }
    irradiance[i*w+j].replaceWith(irrad_acc);
  }

  #ifndef DEBUG
  #pragma omp for schedule(static, 16)
  #endif
  for (uint64_t i=0;i<h;i++) for (uint64_t j=0;j<w;j++)
  { // Convert irradiance to linear RGB, write to buffer
    buffer.img.data[(h-i-1)*w+j] = 
      coefλ2linRGB(irradiance[i*w+j]/float(SPP*HERO_SAMPLES));
  }

  #ifndef DEBUG
  } // ** end of parallel region **************************
  #endif
}
// ****************************************************************************

heroλ Renderer::tracePath(
  scene const &scene,
  heroλ const &λ, 
  ray const &r, 
  nl::RNG &rng, 
  uint64_t scatters) const
{
  hitinfo hinfo;
  if (!intersect::scene(scene, r, hinfo)) { return heroλ(0.f); } // misses
  
  ℝ3 const o = -r.u.normalized();
  ℝ3 const n = hinfo.n();

  // check if path hit a light (emitter material)
  Material const &mat = scene.materials[hinfo.mat];
  if (std::holds_alternative<emitter>(mat)) 
    { return std::get<emitter>(mat).Radiance(λ); }
  if (std::holds_alternative<diremitter>(mat))
    { return std::get<diremitter>(mat).Radiance(λ,o,hinfo.F.z); }

  // hit an object, scatter if less than max scattering
  if (scatters > MAX_SCATTERINGS) return heroλ(0.f);

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
  sample::info<ℝ3,heroλ> si_i_L;
  sample::light(λ, si_l.val, hinfo, scene, si_i_L, rng);

  // evaluate BSDFcosθ for ωi
  heroλ const coef = BxDFcosθ(λ, mat, si_i_L.val, o, n, hinfo.front);

  // Light IS estimate
  heroλ const L_IS = si_i_L.mult * coef / (si_i_L.prob*si_l.prob);

  // ** end of L IS estimate ****************************

  // ** Material IS estimate ****************************
  // sample material:
  //   prob = p(ωi)
  //   val  = ωi
  //   mult = BxDFcosθ
  //   weight = BxDFcosθ/p(ωi)
  sample::info<ℝ3,heroλ> si_i_mat;
  bool const sample = sample::materiali(λ,&mat,hinfo,o,si_i_mat,rng,SAMPLE_P);
  
  // no material sample generated, use light IS estimate
  if (!sample) { return heroλ(L_IS); }

  // evaluate L(ωi)
  ray const i_ray = {hinfo.p, si_i_mat.val};
  heroλ Li = tracePath(scene, λ, i_ray, rng, scatters+1);
  
  // Material IS estimate
  // #ifdef DEBUG
  heroλ const M_IS = Li*si_i_mat.mult / si_i_mat.prob;
  // #else
  //   linRGB const M_IS = Li*si_i_mat.weight;
  // #endif
  // ** end of Material IS estimate *********************

  // ** MIS estimate ************************************
  float const p_L_mati = sample::probForLight(si_l.val, i_ray)*si_l.prob;
  float const p_mat_Li = 
    sample::probForMateriali(λ[0], &mat, hinfo, si_i_L.val, o, SAMPLE_P);

  // power heuristic weights, β=2
  float const p_L_i = si_i_L.prob*si_l.prob;
  float const w_L = p_L_i*p_L_i / (p_L_i*p_L_i + p_mat_Li*p_mat_Li);
  float const w_mat = si_i_mat.prob*si_i_mat.prob 
    / (si_i_mat.prob*si_i_mat.prob + p_L_mati*p_L_mati);

  return M_IS*w_mat + L_IS*w_L;
}
// ****************************************************************************

void Renderer::toneMap(
  rendering const &in_buffer, rendering &tm_buffer, float Y_MID) const
{
  size_t const w = in_buffer.img.width;
  size_t const h = in_buffer.img.height;
  size_t const N = w*h;
  tm_buffer.img.init(w,h);
  // get log-averaged luminance and max luminance
  float maxY = 0.f;
  float sumlogY = 0.f;
  for (size_t i=0; i<N; i++)
  { 
    float const Y = in_buffer.img[i].luma();
    if (Y > maxY) [[unlikely]] { maxY = Y; }
    sumlogY += std::log(nl::max(Y,nl::ε<float>));
  }
  sumlogY /= float(N);
  float const Ybar = std::exp(sumlogY);
  float const exposure = Y_MID/Ybar;

  // Write to tone-mapped buffer
  for (size_t i=0; i<N; i++)
  {
    linRGB c = exposure*in_buffer.img[i];
    linRGB c_tonemapped = tonemap<tonemapping::koiFilmic>(c);
    tm_buffer.img[i] = c_tonemapped;
  }
}
// ****************************************************************************

void Renderer::saveImage(rendering const &buffer, std::string fpath) const
{
  std::string fname = "render"+
    fpath.substr(6,fpath.length()-10)+"-"+std::to_string(SPP)+"spp-"
    +std::to_string(MAX_SCATTERINGS)+"b-"
    +std::format("{:.2f}", SAMPLE_P)+"p.png";
  
  std::vector<rgb24> const display = buffer.rgb24();
  lodepng::encode(
    fname, reinterpret_cast<unsigned char const*>(display.data()), 
    buffer.img.width, buffer.img.height, LCT_RGB, 8);
}
// ****************************************************************************