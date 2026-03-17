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
      heroλ const Li = tracePath(scene, si_λ.val, si.val, rng);
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
  nl::RNG &rng) const
{
  heroλ L(0.f); // radiance accumulation
  heroλ β(1.f); // throughput
  ray current = r;
  float p_light = 0.f; // previous light chosen prob
  float p_b = 0.f;     // BxDF i sample prob
  Light const* prev_light = nullptr;
  uint64_t k=0;
  for (; k<MAX_CAMERA_SCATTERS; k++)
  { // accumulate radiance along path
    hitinfo hinfo;
    if (!intersect::scene(scene, current, hinfo)) { break; } // exits scene
    ℝ3 const o = -current.u.normalized();
    ℝ3 const n = hinfo.n();

    // hits emitter material
    Material const &mat = scene.materials[hinfo.mat];
    if (std::holds_alternative<emitter>(mat)) 
    { 
      heroλ const Le = std::get<emitter>(mat).Radiance(λ);
      if (k==0) { L += β*Le; } // no bounce to compute MIS for
      else 
      { // compute MIS weight for emitter
        float const p_l = p_light*sample::probForLight(prev_light, current); 
        float const w_b = nl::stoch::PowerHeuristic(1,p_b,1,p_l);
        L += β*w_b*Le;
      }
      break;
    }

    // add direct illumination
    sample::info<Light const*> si_light;
    sample::lights(scene.lights, si_light, rng);
    sample::info<ℝ3,heroλ> si_Li;
    sample::light(λ, si_light.val, hinfo, scene, si_Li, rng);
    heroλ const coef = BxDFcosθ(λ,mat,si_Li.val, o, n, hinfo.front);
    float const p_l = si_Li.prob*si_light.prob;
    heroλ const Lo_light = si_Li.mult*coef/p_l;
    float const mat_pdf = 
      sample::probForMateriali(λ[0],&mat,hinfo,si_Li.val,o,1.f);
    float const w_l = nl::stoch::PowerHeuristic(1,p_l,1,mat_pdf);
    L += β*w_l*Lo_light;
    
    // check for scattering
    sample::info<ℝ3,heroλ> si_f;
    bool const sample = sample::materiali(λ,&mat,hinfo,o,si_f,rng,SAMPLE_P);
    if (!sample) { break; }

    // update throughput, prep for next loop
    β *= si_f.mult / si_f.prob;
    current = {hinfo.p, si_f.val};
    p_b = si_f.prob;
    p_light = si_light.prob;
    prev_light = si_light.val;
  }
  return heroλ(L);
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