// ****************************************************************************
/// @file cg.hpp
/// @author Kyle Webster
/// @version 0.10
/// @date 6 Dec 2025
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
  constexpr float r() const {return c[0];}
  constexpr float g() const {return c[1];}
  constexpr float b() const {return c[2];}
  constexpr std::string toString()
    {return std::to_string(c[0])+std::to_string(c[1])+std::to_string(c[2]);}
  constexpr float luma() const {return 0.2126f*c[0]+0.7152f*c[1]+0.0722f*c[2];}
};

template<bra::arithmetic T>
constexpr rgb<T> operator+(rgb<T> const &C1, rgb<T> const &C2) 
  { return rgb<T>(C1.c+C2.c); }
template<bra::arithmetic T> 
constexpr rgb<T> operator*(rgb<T> const &C, float s) {return rgb<T>(C.c*s);}
template<bra::arithmetic T> 
constexpr rgb<T> operator*(float s, rgb<T> const &C) {return C*s;}
template<bra::arithmetic T> 
constexpr rgb<T> operator*(rgb<T> const &C1, rgb<T> const &C2) 
  { return rgb<T>(C1.c*C2.c); }
template<bra::arithmetic T> 
constexpr rgb<T> operator/(rgb<T> const &C,float s) {return rgb<T>(C.c*(1/s));}
template<bra::arithmetic T>
constexpr rgb<T>& operator+=(rgb<T> &C1, rgb<T> const &C2) 
  { C1.c[0]+=C2.c[0]; C1.c[1]+=C2.c[1]; C1.c[2]+=C2.c[2]; return C1; }
template<bra::arithmetic T>
constexpr rgb<T>& operator/=(rgb<T> &C1, float s)
  { C1.c[0]/=s; C1.c[1]/=s; C1.c[2]/=s; return C1; }

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
  constexpr vec(float const (&x)[3]) 
    { for (size_t i=0;i<3;i++) dir[i]=x[i]; dir[3]=0.f; }
  constexpr vec(vec const &v)  {for(size_t i=0;i<4;i++)dir[i]=v.dir[i];}
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
  constexpr pnt(float const (&x)[3]) 
    { for (size_t i=0;i<3;i++) pos[i]=x[i]; pos[3]=1.f; }
  constexpr pnt(pnt const &v)  {for(size_t i=0;i<4;i++)pos[i]=v.pos[i];}
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
  constexpr ℝ3 operator()(float t) const { return p+t*u; }
  constexpr std::array<float,6> plucker() const
    { ℝ3 const m = p^u; return {u[0],u[1],u[2],m[0],m[1],m[2]}; }
};

struct basis
{
  ℝ3 x, y, z;
  basis() = default;
  basis(ℝ3 const &e0, ℝ3 const &e1, ℝ3 const &e2) : x(e0), y(e1), z(e2) {}
  constexpr ℝ3 toBasis(ℝ3 const &v) const { return v[0]*x+v[1]*y+v[2]*z; }
};

/// @brief returns an orthonormal basis with the e0^e1 = e2 = v.normalized()
/// @details
/// Algorithm from Cem Yuksel's cyCodeBase's Vec3 class
constexpr basis orthonormalBasisOf(ℝ3 const &v)
{
  ℝ3 const e2 = v.normalized();
  float const &x=e2[0];
  float const &y=e2[1];
  float const &z=e2[2];
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
    { return ray(M_inv*ℝ4(_ray.p,1.f), M_inv*ℝ4(_ray.u,0.f)); }
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
  { // using Rodrigues' formula R(u,θ) = cosθI+(1-cosθ)*outer(u)-sinθ*skew(u)
    transform T;
    const float θ = deg2rad(degrees);
    const float cosθ = cosf32(θ);
    const float sinθ = sinf32(θ);
    T.M(0,0) = cosθ + (1-cosθ)*u[0]*u[0];
    T.M(0,1) = (1-cosθ)*u[0]*u[1]+sinθ*u[2];
    T.M(0,2) = (1-cosθ)*u[0]*u[2]-sinθ*u[1];
    T.M(1,0) = (1-cosθ)*u[1]*u[0]-sinθ*u[2];
    T.M(1,1) = cosθ + (1-cosθ)*u[1]*u[1];
    T.M(1,2) = (1-cosθ)*u[1]*u[2]+sinθ*u[0];
    T.M(2,0) = (1-cosθ)*u[2]*u[0]+sinθ*u[1];
    T.M(2,1) = (1-cosθ)*u[2]*u[1]-sinθ*u[0];
    T.M(2,2) = cosθ + (1-cosθ)*u[2]*u[2];

    // replace θ with -θ for inverse
    T.M_inv(0,0) = cosθ + (1-cosθ)*u[0]*u[0];
    T.M_inv(0,1) = (1-cosθ)*u[0]*u[1]-sinθ*u[2];
    T.M_inv(0,2) = (1-cosθ)*u[0]*u[2]+sinθ*u[1];
    T.M_inv(1,0) = (1-cosθ)*u[1]*u[0]+sinθ*u[2];
    T.M_inv(1,1) = cosθ + (1-cosθ)*u[1]*u[1];
    T.M_inv(1,2) = (1-cosθ)*u[1]*u[2]-sinθ*u[0];
    T.M_inv(2,0) = (1-cosθ)*u[2]*u[0]-sinθ*u[1];
    T.M_inv(2,1) = (1-cosθ)*u[2]*u[1]+sinθ*u[0];
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

/// @brief transmission direction for relative ior (ηo/ηi), normal n
/// @warning ASSUMES o and n are unit vectors in the same hemisphere
constexpr ℝ3 transmit(ℝ3 const &o, ℝ3 const &n, float ior)
{
  float const cos_i = o|n;
  float const cos_t2 = 1.f-ior*ior*(1.f-cos_i*cos_i);
  if (cos_t2 < 0.f) { return ℝ3(0.f); } // TIR
  return -ior*o-(std::sqrtf(cos_t2)-ior*cos_i)*n;
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
  uint64_t V[3], E[3], GN;
};

template <typename T> struct bvh {};

/// @note All vertex data vectors are indexed by the same indices in F
struct trimeshdata
{
  aabb bounds;
  bvh<triangle>         _bvh;  ///< bvh over triangles F
  std::vector<ℝ3>       V;     ///< vertices
  std::vector<ℝ3>       N;     ///< vertex normals
  std::vector<ℝ3>       T;     ///< vertex tangent vectors
  std::vector<ℝ2>       u;     ///< vertex texture coords
  std::vector<ℝ3>       E;     ///< edges
  std::vector<ℝ3>       GN;    ///< face normals
  std::vector<triangle> F;     ///< faces (triangles)
  constexpr ℝ3 const& v(uint64_t face, int i) const {return V[F[face].V[i]];}
  constexpr ℝ3 const& t(uint64_t face, int i) const {return T[F[face].V[i]];}
  constexpr ℝ2 const& uv(uint64_t face, int i) const {return u[F[face].V[i]];}
  constexpr ℝ3 const& e(uint64_t face, int i) const {return E[F[face].E[i]];}
  constexpr ℝ3 const& gn(uint64_t face) const {return GN[face];}
  constexpr ℝ3 n(uint64_t face, ℝ3 const &b) const
  { 
    triangle const &tri = F[face]; 
    return b[0]*N[tri.V[0]]+b[1]*V[tri.V[1]]+b[2]*V[tri.V[2]]; 
  }
  constexpr ℝ3 t(uint64_t face, ℝ3 const &b) const
  {
    triangle const &tri = F[face];
    return b[0]*T[tri.V[0]]+b[1]*T[tri.V[1]]+b[2]*T[tri.V[2]];
  }
  constexpr ℝ2 uv(uint64_t face, ℝ3 const &b) const
  {
    triangle const &tri = F[face];
    return b[0]*u[tri.V[0]]+b[1]*u[tri.V[1]]+b[2]*u[tri.V[2]];
  }
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
  float size;
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
    { return albedo*inv_π<float>*nl::max(0.f,i|n); }
};
struct blinn : item
{
  linRGB Kd, Ks, Kt, Le;
  float α, ior, p_d, p_spec, w_r, w_t, F0;
  constexpr void init() 
  { 
    p_d    = Kd.luma();
    w_r    = Ks.luma();
    w_t    = Kt.luma();
    p_spec = 1.f-p_d;
    F0 = (ior*ior-2.f*ior+1.f)/(ior*ior+2.f*ior+1.f);
  }
  constexpr float fresnel(float F0, float cos_i) const 
    { return F0+(1.f-F0)*std::pow(1.f-cos_i,5); }
  constexpr std::pair<float,float> fresnelSplit(float F) const
  {
    float p_r = w_r+F*w_t;
    float p_t = (1.f-F)*w_t;
    float const tot = p_r+p_t;
    p_r/=tot;
    p_t/=tot;
    return {p_r, p_t};
  }
  constexpr linRGB fdcosθ(float cos_i) const 
    { return Kd*inv_π<float>*nl::max(0.f,cos_i); }
  constexpr linRGB frcosθ(float cos_h, float F) const
    { return (Ks+F*Kt)*(α+2)*.125f*inv_π<float>*std::pow(cos_h,α); }
  constexpr linRGB ftcosθ(float cos_h_t, float F) const
    { return Kt*(1-F)*(α+2)*.125f*inv_π<float>*std::pow(cos_h_t,α); }
  constexpr linRGB BSDFcosθ(
    ℝ3 const &i, ℝ3 const &o, ℝ3 const &n, bool front) const 
  { 
    float const η = front ? 1.f/ior : ior;
    float const cos_i = i|n;
    float const cos_o = o|n;
    float const F = fresnel(F0, cos_o);
    if (cos_i>0.f)
      { return fdcosθ(cos_i) + frcosθ((i+o).normalized()|n,F); } 
    else { return ftcosθ(-(η*i+o).normalized()|n,F); }
  }
  /// @brief returns p(i|diffuse sampled)
  constexpr float fdiProb(float cos_i) const 
    { return std::abs(cos_i)*inv_π<float>; }
  /// @brief returns p(h|conditions)
  constexpr float blinnhProb(float cos_h) const 
    { return std::pow(std::abs(cos_h),α+1)*(α+1)*.5f*inv_π<float>; }
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
  ℝ3 const &n,
  bool front)
{
  return std::visit(Overload{
    [&](lambertian const &mat){return mat.BRDFcosθ(i,n);},
    [&](blinn const &mat)     {return mat.BSDFcosθ(i,o,n,front);},
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
  basis F; ///< coordinate frame at hit point
  bool front;
  materialidx mat;
  objectidx obj;
  constexpr ℝ3 n() const { return front ? F.z : -F.z; }
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
    Δ = 2*D*tanf32(.5f*deg2rad(fov))/height;
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

constexpr float BIAS = constexprSqrt(ε<float>);

/// @brief Sphere-Ray intersection
/// @param s sphere to intersect
/// @param w_ray world ray
/// @param hinfo hitinfo to populate
/// @return true on intersection, otherwise false
constexpr bool sphere(cg::sphere const &s, ray const &w_ray, hitinfo &hinfo)
{ 
  ray const l_ray = s.T.toLocal(w_ray);

  // descriminant of ray-sphere intersection equation
  float const a = l_ray.u|l_ray.u;
  float const b = 2*(l_ray.p|w_ray.u);
  float const c = (l_ray.p|l_ray.p)-1.f;
  float const Δ = b*b - 4*a*c;
  if (Δ < .1f*BIAS) [[likely]] { return false; }

  // otherwise return closest non-negative t
  float const inv_2a = .5f/a;
  float const tp = (-b + std::sqrtf(Δ))*inv_2a;
  float const tm = (-b - std::sqrtf(Δ))*inv_2a;
  float t = tm;
  if (tm < BIAS) [[unlikely]] { t=tp; } // hit too close
  if (t  < BIAS)   { return false; }    // hit behind ray origin
  if (hinfo.z < t) { return false; }    // closer hit

  // ray hits
  ℝ3 const p = l_ray(t);
  ℝ3 const n = s.T.toWorld(normal(p));
  bool const front = (n|w_ray.u) < 0.f;

  // (θ,φ) parameterization for tangent (and bitangent)
  float const cosθ = n[2];
  float const sinθ = std::sqrtf(1.f-cosθ*cosθ);
  float const φ = atan2f32(n[1],n[0]);
  float const cosφ = cosf(φ);
  float const sinφ = sinf(φ);
  ℝ3 const t_vec = {-sinθ*sinφ,sinθ*cosφ,0.f};
  ℝ3 const b_vec = {cosθ*cosφ,cosθ*sinφ,-sinθ};

  // populate hinfo in world space
  hinfo.z = t;
  hinfo.p = s.T.toWorld(pnt(p));
  hinfo.F.x = t_vec;
  hinfo.F.y = b_vec;
  hinfo.F.z = n;
  hinfo.front = front;
  hinfo.mat = s.mat;
  hinfo.obj = s.obj;
  return true;
}
// ************************************

/// @brief Plane-Ray intersection
constexpr bool plane(cg::plane const &p, ray const &w_ray, hitinfo &hinfo)
{
  ray const l_ray = p.T.toLocal(w_ray);
  float const t = -l_ray.p[2] / l_ray.u[2];
  ℝ3 const x = l_ray(t);
  if (x[0]<-1.f || x[0]>1.f || x[1]<-1.f || x[1]>1.f || t<BIAS || t>hinfo.z) 
    [[likely]] { return false; } // ray misses
  
  // populate hinfo in world space
  hinfo.z = t;
  hinfo.p = p.T.toWorld(pnt(x));
  hinfo.F.z = p.T.toWorld(normal({0.f,0.f,1.f})).normalized();
  hinfo.F.x = p.T.toWorld(vec({0.f,1.f,0.f})).normalized();
  hinfo.F.y = hinfo.F.z^hinfo.F.x;
  hinfo.front = (w_ray.u|hinfo.F.z) < 0.f;
  hinfo.mat = p.mat;
  hinfo.obj = p.obj;
  return true;
}

/// @brief aabb intersection
/// @todo

/// @brief Triangle-Ray intersection
/// @warning hinfo.mat and hinfo.obj are NOT set
constexpr bool trimeshdata(
  uint64_t face, cg::trimeshdata const &mesh, ray const &l_ray, hitinfo &hinfo)
{
  ℝ3 const &v0 = mesh.v(face,0);
  ℝ3 const &v1 = mesh.v(face,1);
  ℝ3 const &v2 = mesh.v(face,2);
  ℝ3 const &e0 = mesh.e(face,0);
  ℝ3 const &e1 = mesh.e(face,1);
  ℝ3 const &e2 = mesh.e(face,2);
  ℝ3 const &gn = mesh.gn(face);
  ℝ3 const m(l_ray.p^l_ray.u);
  float const D = l_ray.u|gn;
  if (std::abs(D)<.1f*BIAS) [[unlikely]] { return false; } // parallel
  float const s0 = (l_ray.u|(v0^v1)) + (m|e0);
  float const s1 = (l_ray.u|(v1^v2)) + (m|e1);
  float const s2 = (l_ray.u|(v2^v0)) + (m|e2);
  if (s0*D<-BIAS || s1*D<-BIAS || s2*D<-BIAS) [[likely]] 
    { return false; } // not in triangle
  float const t = ((v0-l_ray.p)|gn)/D;
  if (t<0.f || hinfo.z < t) { return false; } // behind ray or not closest hit

  // barycentric coordinates
  float const inv_D = 1.f/D;
  ℝ3 const b({s0*inv_D, s1*inv_D, s2*inv_D});

  // populate hinfo
  hinfo.z = t;
  hinfo.p = l_ray(t);
  hinfo.F.z = mesh.n(face,b);
  hinfo.F.x = mesh.t(face,b);
  hinfo.front = (gn|l_ray.u)<0.f;
  return true;
}


/// @brief Mesh-Ray Intersection
constexpr bool trimesh(
  cg::trimesh const tmesh, 
  ray const &w_ray, 
  cg::scene const &sc, 
  hitinfo &hinfo)
{
  bool hit_any = false;
  auto const &mesh = std::get<cg::trimeshdata>(sc.meshes[tmesh.mesh]);
  int n_faces = mesh.F.size();
  ray const l_ray = tmesh.T.toLocal(w_ray);

  /// @todo aabb check


  /// @todo bvh acceleration
  for (int i=0;i<n_faces;i++) { hit_any|=trimeshdata(i, mesh, l_ray, hinfo); }

  // convert hinfo to world-space
  hinfo.p   = tmesh.T.toWorld(pnt(hinfo.p));
  hinfo.F.z = tmesh.T.toWorld(normal(hinfo.F.z));
  hinfo.F.x = tmesh.T.toWorld(vec(hinfo.F.x));
  hinfo.F.y = hinfo.F.z^hinfo.F.x;
  hinfo.mat = tmesh.mat;
  hinfo.obj = tmesh.obj;
  return hit_any;
}


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
      [&](cg::sphere const &s){return sphere(s,r,h);},
      [&](cg::plane const &p){return plane(p,r,h);},
      [&](cg::trimesh const &m){return trimesh(m,r,sc,h);},
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

// ************************************
/// @name light sampling

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
  float const sr = (1.f-std::sqrtf(1.f-sl.size*sl.size/dist2));
  info.prob = 0.5f/(π<float>*sr);

  // sample direction in projection of sphere light onto sphere
  basis const base = orthonormalBasisOf(L);
  float const cosθ = 1.f-rng.flt()*sr;
  float const sinθ = std::sqrtf(1.f-cosθ*cosθ);
  float const φ    = 2*π<float>*rng.flt();
  ℝ3 const ωi = base.x*sinθ*cosf(φ)+base.y*sinθ*sinf(φ)+base.z*cosθ;
  info.val = ωi;

  // get L(ωi)
  hitinfo temp;
  temp.z = std::sqrtf(dist2);
  float const shadowing = 
    intersect::scene(sc,{hinfo.p, ωi},temp,sl._sphere.obj) ? 0.f : 1.f;
  info.mult = sl.radiance*shadowing;
}
/// @brief probability that a ray (direction from a hit point) could be sampled
inline float probForSphereLight(cg::spherelight const &l, ray const &r)
{
  // compute probability if ray intersects
  ℝ3 const L = l._sphere.T.pos()-r.p;
  float const dist2 = L|L;
  float const sr = (1.f-std::sqrtf(1.f-l.size*l.size/dist2));
  float const prob = .5f/(π<float>*sr);

  hitinfo unused;
  return intersect::sphere(l._sphere, r, unused) ? prob : 0.f;
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

/// @brief probability of a light generating a direction as a sample
constexpr float probForLight(
  Light const *l,
  ray const &r)
{
  return std::visit(Overload{
      [&](cg::spherelight const &sl){return probForSphereLight(sl,r);},
      [](auto const &){return 0.f;}
    }, *l);
}
// ** end of light sampling ***********


// ************************************
/// @name material sampling

/// @brief cosine weighted sample of lambertian brdf
inline bool lambertiani(
  cg::lambertian const &mat,
  hitinfo const &hinfo,
  info<ℝ3,linRGB> &info,
  RNG &rng,
  float p)
{
  // russian roulette chance
  float const ξ = rng.flt();
  if (ξ>p) { return false; }

  // generate outgoing direction and fill probability
  ℝ3 dir = stoch::CosHemi(rng.flt(),rng.flt());
  if (!hinfo.front) { dir[2]*=-1; } // flip to correct hemisphere if needed
  ℝ3 i = hinfo.F.toBasis(dir);
  info.prob = p*std::abs(dir[2])*inv_π<float>;
  info.val = i;

  // evaluate BRDFcosθ
  info.mult = mat.BRDFcosθ(i,hinfo.n());
  info.weight = mat.albedo/p;
  return true;
}
/// @brief probability of lambertian generating the sample direction
inline float probForLambertian(
  hitinfo const &hinfo, ℝ3 const &i, ℝ3 const &o, float p)
{ 
  ℝ3 const n = hinfo.n();
  float const cos_i = i|n;
  if ((i|n)<0.f || (o|n)<0.f) return 0.f;
  return p*cos_i*inv_π<float>;
}

/// @brief Blinn half-vector sample
constexpr ℝ3 blinnh(float α, float x0, float x1)
{
  float const cosθ = std::pow(1.f-x0, 1.f/(α+1.f));
  float const sinθ = std::sqrtf(1.f-cosθ*cosθ);
  const float φ    = 2.f*π<float>*x1;
  return {sinθ*cosf(φ), sinθ*sinf(φ), cosθ};
}

/// @brief samples blinn material for an incoming direction
inline bool blinni(
  cg::blinn const &b, 
  hitinfo const &hinfo,
  ℝ3 const &o,
  info<ℝ3,linRGB> &info, 
  RNG &rng, 
  float p)
{
  float const lobe = rng.flt();
  if (lobe>=p) { return false; }
  
  float const pp_d = p*b.p_d; float const pp_spec = p*b.p_spec;

  if (lobe<pp_spec)
  { // specular sample
    // sample half vector, split on Fresnel
    float const η = hinfo.front ? 1.f/b.ior : b.ior;
    ℝ3 ω, h, i_R, i_T;
    ω = blinnh(b.α,rng.flt(),rng.flt());
    if (!hinfo.front) { ω[2]*=-1; } // flip half vector to outgoing hemisphere
    h = hinfo.F.toBasis(ω);
    i_R = bra::reflect(o,h);
    i_T = transmit(o,h,η); // returns {0,0,0} if TIR
    
    /// @todo, compare with transmitted Fresnel split
    float const cos_o = o|h;
    float F = b.fresnel(b.F0,cos_o);
    if (i_T[0]==0.f) { F=1.f; } // TIR, all reflection
    auto [p_r, p_t] = b.fresnelSplit(F);

    if (lobe<pp_spec*p_r)
    { // specular sample
      // p(sample)p(spec|sample)p(r|spec)p(i|r)
      info.prob = pp_spec*p_r*b.blinnhProb(ω[2])*.25f;
      info.val  = i_R;
      info.mult = b.frcosθ(ω[2],F);
      // frcosθ/p(i)
      info.weight = (b.Ks+b.Kt*F)*(b.α+2)/(pp_spec*p_r*std::abs(ω[2])*(b.α+1));
      return true;
    } else
    { // transmissive sample
      // float const cos_it = -h|i_T;
      // float const J = cos_o/((cos_it+η*cos_o)*(cos_it+η*cos_o));
      info.prob = pp_spec*p_t*b.blinnhProb(ω[2])*.25f;
      info.val  = i_T;
      info.mult = b.ftcosθ(ω[2], F);
      info.weight = b.Kt*(1-F)*(b.α+2)/(pp_spec*p_t*std::abs(ω[2])*(b.α+1));
      return true;
    }
  } else if (lobe<pp_spec+pp_d)
  { // diffuse lobe sample
    ℝ3 dir = stoch::CosHemi(rng.flt(), rng.flt());
    if (!hinfo.front) { dir[2]*=-1; }
    ℝ3 const i = hinfo.F.toBasis(dir);
    info.prob = pp_d*b.fdiProb(dir[2]);
    info.val = i;
    info.mult = b.fdcosθ(dir[2]);
    info.weight = b.Kd/pp_d;
    return true;
  }
  return false;
}
/// @brief probability of blinn generating the sample direction
inline float probForBlinn(
  blinn const &b, 
  hitinfo const &hinfo, 
  ℝ3 const &o,
  ℝ3 const &i, 
  float p)
{
  ℝ3 const n = hinfo.n();
  float const cos_i = i|n;
  float const cos_o = o|n;
  float const F = b.fresnel(b.F0, cos_o);
  auto [p_r, p_t] = b.fresnelSplit(F);
  if (cos_i>0.f)
  { // reflection/diffuse
    ℝ3 const h = (i+o).normalized();
    // p(i) = p*(p(d)p(i|d)+p(spec)*p(r|spec)*p(i|r))
    return p*(b.p_d*b.fdiProb(cos_i)+b.p_spec*p_r*b.blinnhProb(h|n)*.25f);
  } else
  { // transmition
    float const ior = hinfo.front ? 1.f/b.ior : b.ior;
    if ((1.f-ior*ior*(1.f-cos_i*cos_i)) < 0.f) { return 0.f; } // should be tir
    // p(i) = pp(spec)p(t|spec)p(i|t)
    ℝ3 const h = (ior*i+o).normalized();
    return p*b.p_spec*p_t*b.blinnhProb(h|n)*.25f;
  }
}


/// @brief material incoming light direction sampling dispatch
constexpr bool materiali(
  Material const *mat,
  hitinfo const &hinfo,
  ℝ3 const &o,
  info<ℝ3,linRGB> &info,
  RNG &rng,
  float p)
{
  return std::visit(Overload{
      [&](cg::lambertian const &l){return lambertiani(l,hinfo,info,rng,p);},
      [&](cg::blinn const &b){return blinni(b,hinfo,o,info,rng,p);},
      [](auto const &){return false;}
    }, *mat);
}

/// @brief material sample evaluation dispatch 
constexpr float probForMateriali(
  Material const *mat, hitinfo const &hinfo, ℝ3 const &i, ℝ3 const &o, float p)
{
  return std::visit(Overload{
      [&](cg::lambertian const &){return probForLambertian(hinfo,i,o,p);},
      [&](cg::blinn const &m){return probForBlinn(m,hinfo,i,o,p);},
      [](auto const&){return 0.f;}
    }, *mat);
}
// ** end of material sampling ********

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
  float const sx = bra::column<3>(s.T.M,0).l2();
  float const sy = bra::column<3>(s.T.M,1).l2();
  float const sz = bra::column<3>(s.T.M,2).l2();
  assert(std::abs(sx-sy)<0.01);
  assert(std::abs(sx-sz)<0.01); // check for uniform scaling
  std::string name = "emitter_"+radiance.toString();
  materialidx mat = mats.idxOf(name);
  if (mat==mats.size()) 
  {
    emitter m = {name, radiance}; 
    mats.emplace_back(std::in_place_type<emitter>, m);
  }
  s.mat = mat;
  light.size = sx;
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
      light._sphere.obj = scene.objects.size();
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

/// @brief loads all scene objects
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
  m.α = alpha;
  m.ior = ior;
  m.init();
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