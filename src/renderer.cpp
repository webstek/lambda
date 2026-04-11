// ****************************************************************************
/// @file renderer.cpp
/// @author Kyle Webster
/// @version 0.7
/// @date 10 Apr 2026
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
      irrad_acc.addHeroλ(si_path.val, si_path.weight);
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

void Renderer::tracePath(
  scene const &scene, 
  ray const &r, 
  sample::info<heroλ,heroλ> &pinfo, 
  nl::RNG &rng) const
{
  // sample hero wavelength from light to connect to
  sample::info<Light const*> si_light;
  sample::lights(scene.lights, si_light, rng);
  sample::info<heroλ> si_λ; 
  sample::radiance(si_light.val, si_λ, rng);
  pinfo.val = si_λ.val;
  pinfo.prob = si_λ.prob*si_light.prob;

  ray wi = r;
  std::vector<pathvertex> cam_subpath;
  for (uint64_t k=0; k<MAX_CAMERA_SCATTERS; k++)
  { // generate camera subpath
    hitinfo hinfo;
    if (!intersect::scene(scene, wi, hinfo)) break;
    Material const &mat = scene.materials[hinfo.mat];
    
    if (std::holds_alternative<emitter>(mat))
    {
      if (k!=0) { break; } 
      pinfo.weight = std::get<emitter>(mat).Radiance(si_λ.val)/pinfo.prob;
      return;
    }

    // scatters, add vertex to subpath
    ℝ3 const o = -wi.u.normalized();
    sample::info<ℝ3, heroλ> si_f;
    bool sample = sample::materiali(si_λ.val, &mat, hinfo, o, si_f, rng, 1.f);
    cam_subpath.push_back({hinfo, ℝ3(o), si_f});
    if (!sample) break;

    // update for next loop
    wi = {hinfo.p, si_f.val};
  }
  uint64_t n = cam_subpath.size();
  if (n==0) { pinfo.weight = heroλ(0.f); return; }

  sample::info<ray,heroλ> si_wo;
  sample::lighto(si_λ.val, si_light.val, si_wo, rng);
  pinfo.prob *= si_wo.prob;
  pinfo.mult = si_wo.mult;
  ray wo = si_wo.val;
  std::vector<pathvertex> light_subpath;
  for (uint64_t k=0; k<MAX_LIGHT_SCATTERS; k++)
  { // generate light subpath
    hitinfo hinfo;
    if (!intersect::scene(scene, wo, hinfo)) break;
    Material const &mat = scene.materials[hinfo.mat];
    if (std::holds_alternative<emitter>(mat)) break;

    // scatters, add vertex to subpath
    ℝ3 const i = -wo.u.normalized();
    sample::info<ℝ3, heroλ> si_f;
    bool sample = sample::materiali(si_λ.val, &mat, hinfo, i, si_f, rng, 1.f);
    light_subpath.push_back({hinfo, ℝ3(i), si_f});
    if (!sample) break;

    // update for next loop
    wo = {hinfo.p, si_f.val};
  }
  uint64_t m = light_subpath.size();
  if (m==0) { pinfo.weight = heroλ(0.f); return; }

  /// @todo all MIS path contribution
  // evaluate bidirectional path contribution
  // check visibility between subpaths
  pathvertex const v_n = cam_subpath[n-1];
  pathvertex const v_m = light_subpath[m-1];
  ℝ3 const p_n = ℝ3(v_n.hinfo.p);
  ℝ3 const p_m = ℝ3(v_m.hinfo.p);
  ℝ3 const u = (p_m-p_n).normalized();
  float L = (p_m-p_n).l2();
  hitinfo hinfo;
  hinfo.z = L-nl::ε<float>;
  float full_prob = 0.f;
  heroλ full_mult = heroλ(0.f);
  if (!intersect::scene(scene, {p_n,u}, hinfo)) 
  {
    // connection factor
    heroλ const f_n = BxDFcosθ(
      si_λ.val, 
      scene.materials[v_n.hinfo.mat], 
      u, 
      v_n.ω_prev, 
      v_n.hinfo.n(), 
      v_n.hinfo.front);
    heroλ const f_m = BxDFcosθ(
      si_λ.val,
      scene.materials[v_m.hinfo.mat],
      v_m.ω_prev,
      -u,
      v_m.hinfo.n(),
      v_m.hinfo.front);
    float G = (abs(v_m.hinfo.n()|u))/(p_m-p_n).len2();

    heroλ bxdfcos = heroλ(1.f);
    float prob = 1.f;
    for (uint64_t k=0; k<n-1; k++) 
    { 
      bxdfcos *= cam_subpath[k].si_f.mult;
      prob *= cam_subpath[k].si_f.prob;
    }
    for (uint64_t k=0; k<m-1; k++)
    { 
      bxdfcos *= light_subpath[k].si_f.mult;
      prob *= light_subpath[k].si_f.prob;
    }
    full_prob = pinfo.prob*prob;
    full_mult = pinfo.mult*f_n*f_m*G*bxdfcos;
  }
  
  // direct lighting
  // visibility
  auto const v_0 = cam_subpath[0];
  ℝ3 v = si_wo.val.p-v_0.hinfo.p;
  L = v.l2();
  v.normalize();
  hinfo.z = L-nl::ε<float>;
  float dir_prob = 0.f;
  heroλ dir_mult = heroλ(0.f);
  sample::info<ℝ3,heroλ> si_dirL;
  sample::light(si_λ.val, si_light.val, v_0.hinfo, scene, si_dirL, rng);
  {
    heroλ const f_0 = BxDFcosθ(
      si_λ.val,
      scene.materials[v_0.hinfo.mat],
      si_dirL.val,
      v_0.ω_prev,
      v_0.hinfo.n(),
      v_0.hinfo.front);
      dir_prob = si_λ.prob*si_light.prob*si_dirL.prob;
      dir_mult = si_dirL.mult*f_0;
    }
  pinfo.prob = full_prob+dir_prob;
  if (full_prob!=0.f) pinfo.mult = full_mult;
  pinfo.mult += dir_mult;
  pinfo.weight = pinfo.mult/pinfo.prob;
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
  std::filesystem::create_directories("render");

  std::string stem = std::filesystem::path(fpath).stem().string();
  std::string fname = "render/" + stem
    + "-" + std::to_string(SPP) + "spp-"
    + std::to_string(MAX_CAMERA_SCATTERS) + "b-"
    + std::to_string(MAX_LIGHT_SCATTERS) + "l-"
    + std::format("{:.2f}", SAMPLE_P) + "p" + suffix + ".png";
  
  std::vector<rgb24> const display = buffer.rgb24();
  lodepng::encode(
    fname, reinterpret_cast<unsigned char const*>(display.data()), 
    buffer.img.width, buffer.img.height, LCT_RGB, 8);
}
// ****************************************************************************