// ****************************************************************************
/// @file cg.hpp
/// @author Kyle Webster
/// @version 0.4
/// @date 1 Dec 2025
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
#include "stoch.hpp"
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
  constexpr rgb(ℝ3 const &x) : c(x) {}
  constexpr rgb(T r, T g, T b) {c[0]=r; c[1]=g; c[2]=b;}
  constexpr rgb(T v) {c[0]=v; c[1]=v; c[2]=v;}
  constexpr rgb& operator=(T v) {c[0]=v; c[1]=v; c[2]=v; return *this;}
  constexpr std::string toString()
    {return std::to_string(c[0])+std::to_string(c[1])+std::to_string(c[2]);}
};

template<bra::arithmetic T> 
constexpr rgb<T> operator*(rgb<T> const &C, float s) {return rgb<T>(C.c*s);}
template<bra::arithmetic T> 
constexpr rgb<T> operator*(float s, rgb<T> const &C) {return C*s;}
template<bra::arithmetic T> 
constexpr rgb<T> operator*(rgb<T> const &C1, rgb<T> const &C2) 
  { return rgb<T>(C1.c*C2.c); }
template<bra::arithmetic T> 
constexpr rgb<T> operator/(rgb<T> const &C, float s) {return rgb<T>(C.c*(1/s));}

/// @name colour representations
using rgb24  = rgb<uint8_t>;
using linRGB = rgb<float>;
using sRGB   = rgb<float>;


constexpr sRGB linRGB2sRGB(linRGB const &x)
{
  auto f=[](float cl){ return cl>0.0031308f ? 
    1.055f*std::pow(cl,1.f/2.4f)-0.055f : 12.92f*cl; };
  return sRGB(f(x.c[0]), f(x.c[1]), f(x.c[2]));
}
constexpr rgb24 sRGB2rgb24(sRGB const &x) 
  { return rgb24(float2byte(x.c[0]),float2byte(x.c[1]),float2byte(x.c[2])); }

// ** end of RGB **********************
// ** end of spectrums ********************************************************


// ****************************************************************************
/// @name spatial

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

struct normal
{
  ℝ3 dir;
  constexpr normal(float const (&x)[3]) {dir = x;}
  constexpr normal(ℝ3 const &x) {dir = x;}
};

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
  ray() {}
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


	// void GetOrthonormals ( Vec3 &v0, Vec3 &v1 ) const	//!< Returns two orthogonal vectors to this vector, forming an orthonormal basis
	// {
	// 	if ( z >= y ) {
	// 		T const a = T(1)/(1 + z);
	// 		T const b = -x*y*a;
	// 		v0.Set( 1 - x*x*a, b, -x );
	// 		v1.Set( b, 1 - y*y*a, -y );
	// 	} else {
	// 		T const a = T(1)/(1 + y);
	// 		T const b = -x*z*a;
	// 		v0.Set( b, -z, 1 - z*z*a );
	// 		v1.Set( 1 - x*x*a, -x, b );
	// 	}
	// }

/// @brief returns an orthonormal basis with the e0^e1 = e2 = v.normalized()
/// @details
/// Algorithm from Cem Yuksel's cyCodeBase's Vec3 class
constexpr basis orthonormalBasisOf(ℝ3 const &v)
{
  ℝ3 const e2 = v.normalized();
  float const x=e2[0];
  float const y=e2[1];
  float const z=e2[2];
  ℝ3 e0, e1;
  if ( z >= y ) 
  {
    float const a = 1.f/(1.f + z);
    float const b = -x*y*a;
    e0 = { 1 - x*x*a, b, -x };
    e1 = { b, 1 - y*y*a, -y };
  } else 
  {
    float const a = 1.f/(1.f + y);
    float const b = -x*z*a;
    e0 = { b, -z, 1 - z*z*a };
    e1 = { 1 - x*x*a, -x, b };
  }
  return basis(e0,e1,e2);
}

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
  constexpr ℝ3 pos() const {return bra::column<3>(M,3);}
  constexpr ray toLocal(ray const &_ray) const
    {return ray(M_inv*ℝ4(_ray.p,1.f), M_inv*ℝ4(_ray.u,0.f)); }
  constexpr ℝ3 toWorld(pnt p) const { return ℝ3(M*p.pos); }
  constexpr ℝ3 toWorld(vec v) const { return ℝ3(M*v.dir); }
  constexpr ℝ3 toWorld(normal n) const 
    { return ℝ3(bra::subMatT<3,3>(M_inv)*n.dir); }

  // ** transform generation **********    
  static constexpr transform scale(ℝ3 const &x)
  {
    transform T;
    for (int i=0;i<3;i++) { T.M(i,i)=x[i]; T.M_inv(i,i)=1.f/x[i]; }
    return T;
  }
  static constexpr transform rotate(ℝ3 const &u, float degrees)
  { // using Rodrigues' formula R(u,θ) = cosθI+(1-cosθ)*outer(u)+sinθ*skew(u)
    transform T;
    const float θ = deg2rad(degrees);
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
  objectidx   obj;
};
struct plane : item
{
  transform   T;
  materialidx mat;
  objectidx   obj;
};
struct trimesh : item
{
  transform   T;
  materialidx mat;
  objectidx   obj;
  meshidx     mesh;
};
// ** end of instances ****************

/// @brief Object interface
using Object = std::variant<sphere, plane, trimesh>;

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

/// @brief Light interface
using Light = std::variant<ambientlight, pointlight, dirlight, spherelight>;
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
  constexpr linRGB BRDFcosθ(ℝ3 const &i, ℝ3 const &n) const 
    { return albedo*inv_π<float>*max(i|n,0.f); }
};
struct blinn : item
{
  linRGB Kd, Ks, Kt, Le, reflect, transmit;
  float α, ior;
  /// @todo add specular lobe
  constexpr linRGB BSDFcosθ(ℝ3 const &i, ℝ3 const &o, ℝ3 const &n) const 
    { return Kd*inv_π<float>*max(i|n,0.f); }
};
struct microfacet : item
{

};
struct emitter : item
{
  linRGB radiance;
};

/// @brief Material interface
using Material = std::variant<lambertian, blinn, microfacet, emitter>;

/// @brief Evaluation of BSDF times geometry term
constexpr linRGB BxDFcosθ(
  Material const &mat, 
  ℝ3 const &i, 
  ℝ3 const &o, 
  ℝ3 const &n)
{
  return std::visit(Overload{
    [&](lambertian const &mat){return mat.BRDFcosθ(i,n);},
    [&](blinn const &mat)     {return mat.BSDFcosθ(i,o,n);},
    [](auto const &){return linRGB(1.f,0.f,0.f);}},
    mat);
}

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

using Mesh     = std::variant<trimeshdata>;
using Texture  = std::variant<valuetex, colourtex>;
// ****************************************************************************


// ****************************************************************************
/// @name rendering
/// @brief rendering related structures

struct hitinfo 
{
  float z = UB<float>;
  ℝ3 p;
  ℝ3 n;
  ℝ3 tangent;
  bool front;
  materialidx mat;
  objectidx obj;
};

struct camera 
{
  basis base;
  ℝ3 pos;
  float fov, dof, D, Δ, h, w;
  uint64_t width, height;
  
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
  list<Light>    lights;
  list<Mesh>     meshes;
  list<Texture>  textures;
};
// ** end of data structures **************************************************


/// @namespace intersect
/// @brief intersection code for all objects
namespace intersect
{ // ** nl::cg::intersect *****************************************************

constexpr float BIAS = ε<float>;

/// @brief Sphere-Ray intersection
/// @param s sphere to intersect
/// @param w_ray world ray
/// @param hinfo hitinfo to populate
/// @return true on intersection, otherwise false
constexpr bool sphere(cg::sphere const &s, ray const &w_ray, hitinfo &hinfo)
{ 
  // convert ray to local space
  ray const l_ray = s.T.toLocal(w_ray);

  // descriminant of ray-sphere intersection equation
  float const a = l_ray.u|l_ray.u;
  float const b = 2*(l_ray.p|l_ray.u);
  float const c = (l_ray.p|l_ray.p)-1.f;
  float const Δ = b*b - 4*a*c;
  if (Δ < BIAS) [[likely]] { return false; }

  // otherwise return closest non-negative t
  float const inv_2a = 1.f/(2.f*a);
  float const tp = (-b + std::sqrtf(Δ))*inv_2a;
  float const tm = (-b - std::sqrtf(Δ))*inv_2a;
  float t = tm;
  if (tm < BIAS)   [[unlikely]] { t=tp; }        // check for hit too close
  if (t  < BIAS)   [[unlikely]] { return false; }
  if (hinfo.z < t) [[unlikely]] { return false; } // check for closest hit

  // ray hits
  ℝ3   const p = l_ray.p+t*l_ray.u;
  ℝ3   const n(p);
  bool const front = (n|l_ray.u) < 0.f;

  // (θ,φ) parameterization for tangent (and bitangent)
  float const sinθ = std::sqrtf(1.f-p[2]*p[2]);
  float const φ = atan2f32(p[1],p[0]);
  ℝ3 const t_vec = {-sinθ*sinf(φ),sinθ*cosf(φ),0.f};

  // populate hitinfo in world space
  hinfo.z = t;
  hinfo.p = s.T.toWorld(pnt(p));
  hinfo.n = s.T.toWorld(normal(n));
  hinfo.tangent = s.T.toWorld(vec(t_vec));
  hinfo.front = front;
  hinfo.mat = s.mat;
  hinfo.obj = s.obj;
  return true;
}
// ************************************

/// @brief finds the intersection closest to the ray origin in the scene
/// @param sc scene to search for intersection in
/// @param r ray to intersect
/// @param h hitinfo to populate
/// @param not object index to skip intersections for
/// @return true on intersection, false otherwise
constexpr bool scene(
  cg::scene const &sc, 
  ray const &r, 
  hitinfo &h, 
  objectidx skip=UB<objectidx>)
{
  bool hit_any = false;
  uint32_t n_objs = sc.objects.size();
  for (objectidx i=0; i<n_objs; i++)
  {
    if (i==skip) continue;
    const bool hit = std::visit(Overload{
      [&](cg::sphere const &s){return sphere(s, r, h);},
      [] (auto const &) {return false;}},
      sc.objects[i]);
    hit_any |= hit;
  }
  return hit_any;
}
// ************************************

} // ** end of namespace intersect ********************************************


/// @namespace sample
/// @brief sampling code
namespace sample
{ // ** nl::cg::sample ********************************************************

template<typename T, typename S=T> struct info 
{
  float prob; ///< sample probability
  T val;      ///< sample value
  S mult;     ///< function evaluated with sample val
  S weight;   ///< mult/prob
};

/// @brief ray from camera c through SS (s[0],s[1]) from disk (s[2],s[3])
inline void camera(cg::camera const &c, ℝ2 const &uv, info<ray> &info, RNG &rng)
{
  const basis F = c.base;
  ℝ3 worldij = c.pos + F.x*(-c.w/2+c.Δ*uv[0])+F.y*(-c.h/2+c.Δ*uv[1])-c.D*F.z;
  ℝ3 worldkl = c.pos + c.dof*(F.x*rng.flt() + F.y*rng.flt());
  info.val = {worldkl, (worldij-worldkl).normalized()};
}

/// @brief uniform random light from a scene
inline void lights(
  list<Light> const &lights, 
  info<Light const*> &info, 
  RNG &rng)
{
  size_t const n = lights.size();
  uint64_t i = rng.uint64()%n;
  info.prob = 1.f/float(n);
  info.val = &lights[i];
}

/// @brief uniformly samples the solid angle of a sphere light from a point
inline void spherelight(
  cg::spherelight const &sl, 
  hitinfo const &hinfo, 
  scene const &sc,
  info<ℝ3,linRGB> &info, 
  RNG &rng) 
{
  // compute probability for ωi
  ℝ3 const L = sl._sphere.T.pos()-hinfo.p;
  float const dist2 = L|L;
  float const size = bra::column<3>(sl._sphere.T.M,0).l2();
  float const sr = (1.f-std::sqrtf(1.f-size*size/dist2));
  float const Ω = 2*π<float>*sr;
  info.prob = 1.f/Ω;

  // sample direction in projection of sphere light onto sphere
  basis const base = orthonormalBasisOf(L);
  float const cosθ = 1.f-rng.flt()*sr;
  float const sinθ = std::sqrtf(1.f-cosθ*cosθ);
  float const φ    = 2*π<float>*rng.flt();
  ℝ3 const ωi = base.x*sinθ*cosf(φ)+base.y*sinθ*sinf(φ)+base.z*cosθ;
  info.val = ωi;

  // get L(ωi)
  hitinfo unused;
  float const shadowing = 
    intersect::scene(sc,{hinfo.p, ωi},unused,sl._sphere.obj) ? 0.f : 1.f;
  info.mult = sl.radiance*shadowing;
}

/// @brief light sampling dispatch
constexpr void light(
  Light const *l, 
  hitinfo const &hinfo, 
  scene const &sc,
  info<ℝ3,linRGB> &info,
  RNG &rng)
{
  std::visit(Overload{
    [&](cg::spherelight const &sl){spherelight(sl,hinfo,sc,info,rng);},
    [](auto const &){}
  }, *l);
}

} // ** end of namespace sample ***********************************************


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
  uint64_t width;
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

inline void loadAmbientLight(ambientlight &light, json const &j) 
  {linRGB irrad; loadℝ3(irrad.c, j.at("irradiance")); light.irradiance = irrad;}
/// @todo
inline void loadPointLight(pointlight &light, json const &j);
/// @todo
inline void loadDirLight(dirlight &light, json const &j);
inline void loadSphereLight(
  spherelight &light, 
  json const &j, 
  list<Material> &mats)
{
  sphere s;
  linRGB radiance;
  loadℝ3(radiance.c, j.at("radiance"));
  loadTransform(s.T, j.at("transform"));
  std::string name = "emitter_"+radiance.toString();
  materialidx mat = mats.idxOf(name);
  if (mat==mats.size()) 
  {
    emitter m = {name, radiance}; 
    mats.emplace_back(std::in_place_type<emitter>, m);
  }
  s.mat = mat;
  light.radiance = radiance;
  light._sphere = s;
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
        ambientlight l; 
        l.name=name; 
        loadAmbientLight(l, j_light); 
        scene.lights.emplace_back(std::in_place_type<ambientlight>,l);
        break;
      }
    case LightType::POINT: break;
    case LightType::DIR: break;
    case LightType::SPHERE:
    {
      spherelight light;
      light.name=name;
      loadSphereLight(light, j_light, scene.materials);
      scene.objects.emplace_back(std::in_place_type<sphere>,light._sphere);
      scene.lights.emplace_back(std::in_place_type<spherelight>,light);
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
      s.obj = objs.size();
      objs.emplace_back(std::in_place_type<sphere>, s); 
      break;
    }
    case ObjectType::PLANE:
    {
      plane p; 
      p.name=name; 
      loadPlane(p, j_obj, mats);
      p.obj = objs.size();
      objs.emplace_back(std::in_place_type<plane>, p); 
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
      lambertian m; 
      m.name=name; 
      loadLambertian(m,j_mat); 
      mats.emplace_back(std::in_place_type<lambertian>, m); 
      break;
    }
    case MaterialType::BLINN:
    {
      blinn b; 
      b.name=name; 
      loadBlinn(b,j_mat); 
      mats.emplace_back(std::in_place_type<blinn>, b); 
      break;
    }
    case MaterialType::MICROFACET:
    {
      microfacet m; 
      m.name=name; 
      loadMicrofacet(m,j_mat); 
      mats.emplace_back(std::in_place_type<microfacet>, m); 
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