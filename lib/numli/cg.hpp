// ****************************************************************************
/// @file cg.hpp
/// @author Kyle Webster
/// @version 0.2
/// @date 22 Nov 2025
/// @brief Numerics Library - Computer Graphics - @ref cg
/// @details
/// Collection of computer graphics structures and algorithms
// ****************************************************************************
#ifndef CG_HPP
#define CG_HPP
// ** Includes ****************************************************************
#include <variant>
#include <vector>
#include <string>
#include <fstream>

#include "json.hpp"

#include "bra.hpp"
// ****************************************************************************

namespace nl 
{ // ** nl ****************************

/// @namespace cg
/// @brief Computer graphics data structures and algorithms
/// @details
/// Contains:
///  * @ref data_structures
namespace cg 
{ // ** nl::cg ************************

// ************************************
/// @name aliases
using materialidx = uint32_t;
using objectidx   = uint32_t;
using meshidx     = uint32_t;

// ****************************************************************************
/// @name data structures

template<typename T>
concept named = requires(T t) {{t.name}->std::convertible_to<std::string>;};
inline std::string getName(const named auto &x) { return x.name; }
template<typename... Ts>
inline std::string getName(const std::variant<Ts...> &v) {
  return std::visit([](auto const &x) { return x.name; }, v);
}

/// @brief named item for other structs to inherit
struct item {std::string name;};

/// @brief list class
/// @warning requires elements to have a public name member
template <typename T>
struct list : std::vector<T>
{
  /// @brief returns index of named element, returns size if not found 
  size_t idxOf(std::string const &name) const
  {
    size_t N = this->size();
    size_t i=0;
    for (; i<N; i++) { if (getName(this->at(i)) == name) break; }
    return i;
  }
  T* find(std::string const &name) 
  { 
    size_t idx = idxOf(name);
    if (idx!=this->size()) { return &this->at(idxOf(name)); }
    return nullptr;
  }
  T const* find(std::string const &name) const 
  {
    size_t idx=idxOf(name);
    if (idx!=this->size()) { return &this->at(idxOf(name)); }
    return nullptr;
  }
};


// ****************************************************************************
/// @name spectrums

// ************************************
/// @name RGB

template <bra::arithmetic T> struct rgb
{
  bra::ℝn<3,T> c;
  constexpr rgb() {}
  constexpr rgb(T r, T g, T b) {c[0]=r; c[1]=g; c[2]=b;}
  constexpr rgb(T v) {c[0]=v; c[1]=v; c[2]=v;}
  constexpr rgb& operator=(T v) {c[0]=v; c[1]=v; c[2]=v; return *this;}
  constexpr std::string toString()
    {return std::to_string(c[0])+std::to_string(c[1])+std::to_string(c[2]);}
};

using sRGB   = rgb<uint8_t>;
using linRGB = rgb<float>;

constexpr sRGB tosRGB(linRGB const &x) 
  { return sRGB(float2byte(x.c[0]), float2byte(x.c[1]), float2byte(x.c[2])); }

// ** end of RGB **********************

// ** end of spectrums ********************************************************


// ****************************************************************************
/// @name spatial

/// @brief computes the inverse transformation matrix of M
constexpr ℝ4x4 inverseTransform(ℝ4x4 const &M)
{
  ℝ3x3 R = {M(0,0),M(0,1),M(0,2),M(1,0),M(1,1),M(1,2),M(2,0),M(2,1),M(2,2)};
  ℝ3x3 R_inv = bra::inverse(R);
  ℝ4x4 M_inv;
  ℝ3 t = -R_inv*bra::column<3>(M,3);
  for (int i=0;i<3;i++) 
  {
    for (int j=0;j<3;j++) { M_inv(i,j)=R_inv(i,j); }
    M_inv(i,3) = t[i];
  }
  M_inv(3,0)=0.f; M_inv(3,1)=0.f; M_inv(3,2)=0.f; M_inv(3,3)=1.f;
  return ℝ4x4(M_inv.elem);
}

struct vec 
{
  ℝ4 dir;
  constexpr vec() {dir[3]=0.f;}
  constexpr vec(float (&x)[3]) {for(size_t i=0;i<3;i++)dir[i]=x[i];dir[3]=0.f;}
  constexpr vec(vec const &v)  {for(size_t i=0;i<3;i++)dir[i]=v.dir[i];}
  constexpr vec(ℝ3 const &x)   {for(size_t i=0;i<3;i++)dir[i]=x[i];dir[3]=0.f;}
  constexpr vec(ℝ4 const &x)   {for(size_t i=0;i<3;i++)dir[i]=x[i];dir[3]=0.f;}
};
constexpr vec operator*(float s, vec const &v) { return vec(v.dir*s); }
constexpr vec operator*(vec const &v, float s) { return s*v; }

struct pnt
{
  ℝ4 pos;
  constexpr pnt() {pos[3]=1.f;}
  constexpr pnt(float (&x)[3]) {for(size_t i=0;i<3;i++)pos[i]=x[i];pos[3]=1.f;}
  constexpr pnt(pnt const &v)  {for(size_t i=0;i<3;i++)pos[i]=v.pos[i];}
  constexpr pnt(ℝ3 const &x)   {for(size_t i=0;i<3;i++)pos[i]=x[i];pos[3]=1.f;}
  constexpr pnt(ℝ4 const &x)   {for(size_t i=0;i<3;i++)pos[i]=x[i];pos[3]=1.f;}
};
constexpr pnt operator+(pnt const &p, vec const &v) {return pnt(p.pos+v.dir);}
constexpr pnt operator+(vec const &v, pnt const &p) {return p+v;}

struct ray 
{
  ℝ3 p;
  ℝ3 u;
  ray(ℝ3 const &p, ℝ3 const &u) : p(p), u(u) {}
  ray(ℝ4 const &p, ℝ4 const &u) : 
    p({p.elem[0],p.elem[1],p.elem[2]}), u({u.elem[0],u.elem[1],u.elem[2]}) {}
  constexpr pnt operator()(float t) { return p+t*u; }
};
struct basis
{
  ℝ3 x, y, z;
  basis() = default;
  basis(ℝ3 const &e0, ℝ3 const &e1, ℝ3 const &e2) : x(e0), y(e1), z(e2) {}
};
struct transform
{
  ℝ4x4 M;
  ℝ4x4 M_inv;

  /// @name constructors
  constexpr transform() { M.identity(); M_inv.identity(); }
  constexpr transform(ℝ3 const &x, ℝ3 const &y, ℝ3 const &z, ℝ3 const &p)
  {
    for (int i=0;i<3;i++) {M(i,0)=x[i]; M(i,1)=y[i]; M(i,2)=z[i]; M(i,3)=p[i];}
    M(3,0)=0.f; M(3,1)=0.f; M(3,2)=0.f; M(3,3)=1.f;
    M_inv = inverseTransform(M);
  }

  /// @name member functions
  constexpr ray toLocal(ray const &_ray) const
    {return ray(M*ℝ4(_ray.p,1.f), M*ℝ4(_ray.u,0.f)); }
  static constexpr transform scale(ℝ3 const &x)
  {
    transform T;
    for (int i=0;i<3;i++) { T.M(i,i)=x[i]; T.M_inv(i,i)=1.f/x[i]; }
    return T;
  }
  static constexpr transform rotate(ℝ3 const &u, float degrees)
  { // using Rodrigues' formula R(u,θ) = cosθI+(1-cosθ)*outer(u)+sinθ*skew(u)
    transform T;
    const float θ = radians(degrees);
    const float cosθ = cosf32(θ);
    const float sinθ = sinf32(θ);
    T.M(0,0) = cosθ + (1-cosθ)*u[0]*u[0];
    T.M(0,1) = (1-cosθ)*u[0]*u[1]-sinθ*u[2];
    T.M(0,2) = (1-cosθ)*u[0]*u[2]+sinθ*u[1];
    T.M(1,0) = (1-cosθ)*u[1]*u[0]+sinθ*u[2];
    T.M(1,1) = cosθ + (1-cosθ)*u[1]*u[1];
    T.M(1,2) = (1-cosθ)*u[1]*u[2]-sinθ*u[0];
    T.M(2,0) = (1-cosθ)*u[2]*u[0]-sinθ*u[1];
    T.M(2,1) = (1-cosθ)*u[2]*u[1]+sinθ*u[0];
    T.M(2,2) = cosθ + (1-cosθ)*u[2]*u[2];

    // replace θ with -θ for inverse
    T.M_inv(0,0) = cosθ + (1-cosθ)*u[0]*u[0];
    T.M_inv(0,1) = (1-cosθ)*u[0]*u[1]+sinθ*u[2];
    T.M_inv(0,2) = (1-cosθ)*u[0]*u[2]-sinθ*u[1];
    T.M_inv(1,0) = (1-cosθ)*u[1]*u[0]-sinθ*u[2];
    T.M_inv(1,1) = cosθ + (1-cosθ)*u[1]*u[1];
    T.M_inv(1,2) = (1-cosθ)*u[1]*u[2]+sinθ*u[0];
    T.M_inv(2,0) = (1-cosθ)*u[2]*u[0]+sinθ*u[1];
    T.M_inv(2,1) = (1-cosθ)*u[2]*u[1]-sinθ*u[0];
    T.M_inv(2,2) = cosθ + (1-cosθ)*u[2]*u[2];
    return T;
  }
  static constexpr transform translate(ℝ3 const &x) 
  { 
    transform T; 
    for (int i=0;i<3;i++) { T.M(i,3)=x[i]; T.M_inv(i,3)=-x[i]; } 
    return T;
  }
};

/// @brief compose transform T2 after T1
/// @todo composition operator
constexpr transform operator<<(transform const &T1, transform const &T2)
{
  transform T;
  T.M = T2.M*T1.M;
  T.M_inv = T1.M_inv*T2.M_inv;
  return T;
}
// ** end of spatial **********************************************************


// ****************************************************************************
/// @name infos

struct hitinfo {};
struct sampleinfo {};
// ****************************************************************************


// ****************************************************************************
/// @name objects

// ************************************
/// @name structures

struct aabb
{
  ℝ3 inf, sup;
};
struct triangle
{
  ℝ3 x0, x1, x2;
};
struct vertex 
{
  vec n;
};
template <typename T> struct bvh {};
struct trimeshdata
{
  bvh<triangle>         _bvh;
  std::vector<ℝ3>       V;
  std::vector<uint32_t> F;
};
// ** end of structures ***************

// ************************************
/// @name object instances
enum class ObjectType {GROUP, SPHERE, PLANE, TRIMESH};
constexpr ObjectType str2obj(std::string s)
{
  if (s=="group")   return ObjectType::GROUP;
  if (s=="sphere")  return ObjectType::SPHERE;
  if (s=="plane")   return ObjectType::PLANE;
  if (s=="trimesh") return ObjectType::TRIMESH;
  throw std::domain_error("No corresponding ObjectType.");
}

struct sphere : item
{
  transform   T;
  materialidx mat;
};
struct plane : item
{
  transform T;
  materialidx mat;
};
struct trimesh : item
{
  transform   T;
  materialidx mat;
  meshidx     mesh;
};
// ** end of instances ****************
// ** end of objects **********************************************************


// ****************************************************************************
/// @name lights
enum class LightType {AMBIENT, POINT, DIR, SPHERE};
constexpr LightType str2light(std::string s)
{
  if (s=="ambient")   return LightType::AMBIENT;
  if (s=="point")     return LightType::POINT;
  if (s=="direction") return LightType::DIR;
  if (s=="sphere")    return LightType::SPHERE;
  throw std::domain_error("No corresponding LightType.");
}

struct ambientlight : item
{
  linRGB irradiance;
};
struct pointlight : item
{
  linRGB radiant_intensity;
  pnt   pos;
};
struct dirlight : item
{
  linRGB radiant_intensity;
  vec   dir;
};
struct spherelight : item
{
  linRGB radiance;
  sphere _sphere;
};
// ** end of lights ***********************************************************


// ****************************************************************************
/// @name materials
enum class MaterialType {NONE, LAMBERTIAN, BLINN, MICROFACET};
constexpr MaterialType str2mat(std::string const &s)
{
  if (s=="lambertian") return MaterialType::LAMBERTIAN;
  if (s=="blinn")      return MaterialType::BLINN;
  if (s=="microfacet") return MaterialType::MICROFACET;
  return MaterialType::NONE;
}

struct lambertian : item
{
  linRGB albedo;
};
struct blinn : item
{
  linRGB Kd, Ks, Kt, Le, reflect, transmit;
  float α, ior;
};
struct microfacet : item
{

};
struct emitter : item
{
  linRGB radiance;
};
// ** end of materials ********************************************************


// ****************************************************************************
/// @name imaging

template <typename T> struct image 
{
  size_t width, height;
  std::vector<T> data;
  void init(size_t w, size_t h) {width=w; height=h; data.resize(width*height);}
};

template <typename T>
struct texture {};
using valuetex  = texture<float>;
using colourtex = texture<linRGB>;
// ** end of imaging **********************************************************


// ****************************************************************************
/// @name variants

using Light    = std::variant<ambientlight, pointlight, dirlight, spherelight>;
using Object   = std::variant<sphere, plane, trimesh>;
using Mesh     = std::variant<trimeshdata>;
using Material = std::variant<lambertian, blinn, microfacet, emitter>;
using Texture  = std::variant<valuetex, colourtex>;
// ****************************************************************************


// ****************************************************************************
/// @name rendering
/// @brief rendering related structures

struct camera 
{
  basis base;
  ℝ3 pos;
  float fov, dof, D, Δ, h, w;
  uint width, height;
  
  void init()
  {
    // assumes focal distance 1
    Δ = 2*D*tanf32(fov*π<float>/360)/height;
    h = Δ*height;
    w = Δ*width;
  }
};

struct scene 
{
  // main interface components
  camera      cam;
  bvh<Object> obvh;

  // shared storage
  list<Object>   objects;
  list<Material> materials;
  list<Light>    ideal_lights;
  list<Mesh>     meshes;
  list<Texture>  textures;
  std::vector<Light*> lights;
};
// ** end of data structures **************************************************


/// @namespace sample
/// @brief sampling code
namespace sample
{ // ** nl::cg::sample ********************************************************

/// @brief ray from camera c through SS (s[0],s[1]) from disk (s[2],s[3])
constexpr ray camera(cg::camera const &c, ℝ4 const &s)
{
  const basis F = c.base;
  ℝ3 worldij = c.pos + F.x*(-c.w/2+c.Δ*s[0])+F.y*(-c.h/2+c.Δ*s[1])-c.D*F.z;
  ℝ3 worldkl = c.pos + c.dof*(F.x*s[2] + F.y*s[3]);
  return ray(worldkl, (worldij-worldkl).normalized());
}
} // ** end of namespace sample ***********************************************

/// @namespace intersect
/// @brief intersection code for all objects
namespace intersect
{ // ** nl::cg::intersect *****************************************************

constexpr float BIAS = ε<float>;

constexpr bool sphere(cg::sphere const &s, ray const &w_ray, hitinfo &h_info)
{ 
  // convert ray to local space
  ray l_ray = s.T.toLocal(w_ray);

  // descriminant of ray-sphere intersection equation
  float const a = l_ray.u|l_ray.u;
  float const b = 2*(l_ray.p|l_ray.u);
  float const c = l_ray.p|l_ray.p;
  float const Δ = b*b - 4*a*c;
  if (Δ < BIAS) [[likely]] { return false; }

  // otherwise return closest non-negative t
  float const inv_2a = 1.f/(2.f*a);
  float const tp = (-b + sqrt(Δ))*inv_2a;
  float const tm = (-b - sqrt(Δ))*inv_2a;
  float t = tm;
  if (tm < BIAS)   [[unlikely]] { t=tp; }        // check for hit too close
  if (t  < BIAS)   [[unlikely]] { return false; }
  // if (h.z < t) [[unlikely]] { return false; } // check for closest hit

  // ray hits
  // ℝ3  const p = l_ray.p+t*l_ray.u;
  // ℝ3  const n(p);
  // bool const front = (n|l_ray.u) < 0.f;

  // populate hitinfo
  /// @todo
  return true;
}

constexpr bool scene(cg::scene const &sc, ray const &r, hitinfo &h)
{
  bool hit_any = false;
  for (auto const &obj : sc.objects)
  {
    hit_any |= std::visit(Overload{
      [&](cg::sphere const &s){return sphere(s, r, h);},
      [] (auto const &object) {(void)object; return false;}},
      obj);
  }
  return hit_any;
}

} // ** end of namespace intersect ********************************************


/// @namespace load
/// @brief loading utilities
namespace load
{ // nl::cg::load *************************************************************
using json = nlohmann::json;

// ************************************
/// @name data loading
template <bra::arithmetic T> inline void load(T &x, json const &j) 
  {x=j.get<T>();}
inline void loadℝ3(ℝ3 &x, json const &j) 
  {for(int i=0;i<3;i++)x[i]=j[i].get<float>();}
inline void loadTransform(transform &T, json const &j)
{
  ℝ3 _scale, _translate, _axis;
  float _deg;
  try { loadℝ3(_scale, j.at("scale")); } catch(...) { _scale=1.f; }
  try { loadℝ3(_translate, j.at("translate")); } catch(...) { _translate=0.f; }
  try 
  { // try to load rotation
    json rot = j.at("rotate"); 
    loadℝ3(_axis, rot.at("axis")); 
    load(_deg, rot.at("degrees")); 
  } catch(...) { _axis={0.f,0.f,1.f}; _deg=0.f; }
  auto scaling = transform::scale(_scale);
  auto rotation = transform::rotate(_axis, _deg);
  auto translation = transform::translate(_translate);
  T = scaling << rotation << translation;
}
// ************************************

// ************************************
/// @brief loads a camera
inline void loadCamera(camera &cam, json const &j) 
{
  ℝ3 pos, look_at, up;
  float fov, dof, ar, focal_dist;
  uint width;
  loadℝ3(pos, j.at("pos"));
  loadℝ3(look_at, j.at("look_at"));
  loadℝ3(up, j.at("up"));
  load(width, j.at("width"));
  load(fov, j.at("fov"));
  try { load(ar, j.at("ar")); } catch(...) { ar=1.7778f; } // default 16:9
  try { load(dof, j.at("f")); } catch(...) { dof=0.f; }
  try { load(focal_dist, j.at("f")); } catch(...) { focal_dist=1.f; }
  cam.fov = fov;
  cam.dof = dof;
  cam.D = focal_dist;
  cam.width = width;
  cam.height = std::ceil(width/ar);
  ℝ3 z = (pos-look_at).normalized();
  ℝ3 x = up^z;
  ℝ3 y = z^x;
  cam.base = {x,y,z};
  cam.pos  = pos;
  cam.init();
}
// ************************************

// ************************************
/// @name light loading

inline void loadAmbientLight(ambientlight &l_ray, json const &j) 
  {linRGB irrad; loadℝ3(irrad.c, j.at("irradiance")); l_ray.irradiance = irrad;}
/// @todo
inline void loadPointLight(pointlight &l_ray, json const &j);
/// @todo
inline void loadDirLight(dirlight &l_ray, json const &j);
inline void loadSphereLight(
  spherelight &l_ray, 
  json const &j, 
  list<Material> &mats)
{
  sphere s;
  linRGB radiance;
  loadℝ3(radiance.c, j.at("radiance"));
  loadTransform(s.T, j.at("transform"));
  std::string name = "emitter_"+radiance.toString();
  materialidx mat = mats.idxOf(name);
  if (mat==mats.size()) {emitter m = {name, radiance}; mats.push_back(m);}
  s.mat = mat;
  l_ray.radiance = radiance;
  l_ray._sphere = s;
}
/// @todo point and direction light loading
inline void loadLights(
  scene &scene,
  json const &j) 
{
  size_t n_lights = j.size();
  for (size_t i=0; i<n_lights; i++)
  {
    auto j_light = j[i];
    auto name = j_light.at("name").get<std::string>();
    auto type = str2light(j_light.at("type").get<std::string>());
    switch (type)
    {
    case LightType::AMBIENT:
      {
        ambientlight l_ray; 
        l_ray.name=name; 
        loadAmbientLight(l_ray, j_light); 
        scene.ideal_lights.push_back(l_ray);
        scene.lights.push_back(&scene.ideal_lights.back());
        break;
      }
    case LightType::POINT: break;
    case LightType::DIR: break;
    case LightType::SPHERE:
    {
      spherelight l_ray;
      l_ray.name=name;
      loadSphereLight(l_ray, j_light, scene.materials);
      scene.objects.push_back(l_ray._sphere);
      break;
    }
    }
  }
}
// ************************************

// ************************************
/// @name object loading

/// @todo
inline void loadGroup() {}
inline void loadSphere(sphere &s, json const &j, list<Material> const &mats)
{
  transform T;
  loadTransform(T, j.at("transform"));
  s.T = T;
  s.mat = mats.idxOf(j.at("material").get<std::string>());
}
inline void loadPlane(plane &p, json const &j, list<Material> const &mats) 
{
  transform T;
  loadTransform(T, j.at("transform"));
  p.T = T;
  p.mat = mats.idxOf(j.at("material").get<std::string>());
}
/// @todo
inline void loadTriMesh(trimesh &m, json const &j, list<Material> const &mats);
inline void loadObjects(
  list<Object> &objs, 
  json const &j, 
  list<Material> const &mats)
{
  size_t n_objs = j.size();
  for (size_t i=0; i<n_objs;i++)
  {
    auto j_obj = j[i];
    auto name = j_obj.at("name").get<std::string>();
    auto type = str2obj(j_obj.at("type").get<std::string>());
    switch (type)
    {
    case ObjectType::GROUP:
      /// @bug does not apply transformation of group to children
      {loadObjects(objs, j_obj, mats); break;}
    case ObjectType::SPHERE: 
    {
      sphere s; 
      s.name=name; 
      loadSphere(s, j_obj, mats); 
      objs.push_back(s); 
      break;
    }
    case ObjectType::PLANE:
    {
      plane p; 
      p.name=name; 
      loadPlane(p, j_obj, mats); 
      objs.push_back(p); 
      break;
    }
    case ObjectType::TRIMESH: break;
    }
  }
}
// ************************************

// ************************************
/// @name material loading

inline void loadLambertian(lambertian &m, json const &j)
{
  linRGB albedo;
  loadℝ3(albedo.c, j.at("albedo"));
  m.albedo = albedo;
}
inline void loadBlinn(blinn &m, json const &j)
{
  linRGB Kd, Ks, Kt, Le, reflect, transmit;
  float alpha, ior;
  try {loadℝ3(Kd.c, j.at("Kd"));} catch(...) {Kd.c = 0.f;}
  try {loadℝ3(Ks.c, j.at("Ks"));} catch(...) {Ks.c = 0.f;}
  try {loadℝ3(Kt.c, j.at("Kt"));} catch(...) {Kt.c = 0.f;}
  try {loadℝ3(Le.c, j.at("Le"));} catch(...) {Le.c = 0.f;}
  try {loadℝ3(reflect.c, j.at("reflect"));} catch(...) {reflect.c = 0.f;}
  try {loadℝ3(transmit.c, j.at("transmit"));} catch(...) {transmit.c = 0.f;}
  try {load(alpha, j.at("glossiness"));} catch(...) {alpha = 1024;}
  try {load(ior, j.at("ior"));} catch(...) {ior=1.54;}
  m.Kd = Kd;
  m.Ks = Ks;
  m.Kt = Kt;
  m.Le = Le;
  m.reflect = reflect;
  m.transmit = transmit;
  m.α = alpha;
  m.ior = ior;
}
/// @todo
inline void loadMicrofacet(microfacet &m, json const &j) {(void)m; (void)j;}

inline void loadMaterial(Material &mat, json const &j)
{
  std::visit(
    Overload {
      [&](lambertian &m){ loadLambertian(m, j); },
      [&](blinn &m){ loadBlinn(m, j); },
      [&](microfacet &m){ loadMicrofacet(m, j); },
      [](auto &m){(void)m;}}, // no behaviour fallback
    mat);
}
inline void loadMaterials(list<Material> &mats, json const &j)
{
  size_t n_mats = j.size();
  for (size_t i=0; i<n_mats; i++)
  {
    auto j_mat = j[i];
    auto name = j_mat.at("name").get<std::string>();
    auto type = str2mat(j_mat.at("type").get<std::string>());
    switch (type)
    {
    case MaterialType::NONE: break;
    case MaterialType::LAMBERTIAN:
    {
      lambertian l_ray; 
      l_ray.name=name; 
      loadLambertian(l_ray,j_mat); 
      mats.push_back(l_ray); 
      break;
    }
    case MaterialType::BLINN:
      {blinn b; b.name=name; loadBlinn(b,j_mat); mats.push_back(b); break;}
    case MaterialType::MICROFACET:
    {
      microfacet m; 
      m.name=name; 
      loadMicrofacet(m,j_mat); 
      mats.push_back(m); 
      break;
    }
    } // end of switch
  }
}
// ************************************

/// @todo load gltf node 
inline void loadGLTFNode(scene &scene, std::string fpath);

/// @brief load scene from file
/// @param scene scene to put data in
/// @param fpath path to file to load
/// @return true on successful loading, false otherwise
/// @todo Texture support
inline bool loadNLS(scene &scene, std::string fpath)
{
  std::ifstream file(fpath);
  if (!file.is_open()) 
    {throw std::runtime_error("Could no open file."); return false;}
  json j;
  file >> j;

  loadCamera(scene.cam, j.at("camera"));
  loadMaterials(scene.materials, j.at("materials"));
  loadObjects(scene.objects, j.at("objects"), scene.materials);
  loadLights(scene, j.at("lights"));
  return true;
}



} // ** end of namespace load *************************************************

} // ** end of namespace cg ***********
} // ** end of namespace nl ***********

// ****************************************************************************
#endif // #ifndef CG_HPP