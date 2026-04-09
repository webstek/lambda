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

  /// @todo add adaptive sampling
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

      // compute Li along generated ray, add to irradiance
      sample::info<heroλ,heroλ> si_path;
      tracePath(scene, si.val, si_path, rng);
      irrad_acc.addHeroλ(si_path.val, si_path.mult);
    }
    irradiance[i*w+j].replaceWith(irrad_acc);
  }

  /// @todo add density-based outlier rejection
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

/// @todo finish converting to sample::info instead of spectrum return
void Renderer::tracePath(
  scene const &scene, 
  ray const &r, 
  sample::info<heroλ,heroλ> pinfo, 
  nl::RNG &rng) const
{
  // sample hero wavelength from light to connect to
  sample::info<Light const*> si_light;
  sample::lights(scene.lights, si_light, rng);
  sample::info<heroλ> si_λ; 
  sample::radiance(si_light.val, si_λ, rng);
  pinfo.val = si_λ.val;
  pinfo.prob = si_λ.prob;

  ray wi = r;
  std::vector<hitinfo> cam_subpath;
  for (int k=0; k<MAX_CAMERA_SCATTERS; k++)
  { // generate camera subpath
    hitinfo hinfo;
    if (!intersect::scene(scene, wi, hinfo)) break;
    Material const &mat = scene.materials[hinfo.mat];
    
    if (std::holds_alternative<emitter>(mat))
    { // directly connects to light
      return std::get<emitter>(mat).Radiance(si_λ.val);
    }

    // scatters, add vertex to subpath
    cam_subpath.push_back(hinfo);
    ℝ3 const o = -wi.u.normalized();
    ℝ3 const n = hinfo.n();
    sample::info<ℝ3, heroλ> si_f;
    bool sample = sample::materiali(si_λ.val, &mat, hinfo, o, si_f, rng, 1.f);
    if (!sample) break;

    // update for next loop
    wi = {hinfo.p, si_f.val};
  }
  uint64_t n = cam_subpath.size();
  if (n==0) return heroλ(0.f);

  sample::info<ray,heroλ> si_wo;
  sample::lighto(si_λ.val, si_light.val, si_wo, rng);
  ray wo = si_wo.val;
  std::vector<hitinfo> light_subpath;
  for (int k=0; k<MAX_LIGHT_SCATTERS; k++)
  { // generate light subpath
    hitinfo hinfo;
    if (!intersect::scene(scene, wo, hinfo)) break;
    Material const &mat = scene.materials[hinfo.mat];
    
    if (std::holds_alternative<emitter>(mat)) break;

    // scatters, add vertex to subpath
    light_subpath.push_back(hinfo);
    ℝ3 const i = -wo.u.normalized();
    ℝ3 const n = hinfo.n();
    sample::info<ℝ3, heroλ> si_f;
    bool sample = sample::materiali(si_λ.val, &mat, hinfo, i, si_f, rng, 1.f);
    if (!sample) break;

    // update for next loop
    wo = {hinfo.p, si_f.val};
  }
  uint64_t m = light_subpath.size();

  /// @todo path contribution
  // evaluate bidirectional path contribution
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

void Renderer::saveImage(
  rendering const &buffer, std::string fpath, std::string suffix) const
{
  std::string fname = "render"+
    fpath.substr(6,fpath.length()-10)+"-"+std::to_string(SPP)+"spp-"
    +std::to_string(MAX_CAMERA_SCATTERS)+"b-"
    +std::format("{:.2f}", SAMPLE_P)+"p"+suffix+".png";
  
  std::vector<rgb24> const display = buffer.rgb24();
  lodepng::encode(
    fname, reinterpret_cast<unsigned char const*>(display.data()), 
    buffer.img.width, buffer.img.height, LCT_RGB, 8);
}
// ****************************************************************************