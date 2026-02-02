// ****************************************************************************
/// @file cg.hpp
/// @author Kyle Webster
/// @version 0.17
/// @date 02 Feb 2026 
/// @brief Numerics Library - Computer Graphics - @ref cg
/// @details
/// Collection of computer graphics structures and algorithms
// ****************************************************************************
#ifndef CG_HPP
#define CG_HPP
// ** Includes ****************************************************************
#include <variant>
#include <vector>
#include <map>
#include <queue>
#include <string>
#include <fstream>

#include "fast_obj.h"
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
/// @name constants
constexpr uint32_t Nλ = 64;
constexpr uint8_t HERO_SAMPLES = 8;

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
/// @name light

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
using XYZ    = rgb<float>;

constexpr XYZ linRGB2XYZ( linRGB const &C )
{ return XYZ(
    0.4124564f*C.r() + 0.3575761f*C.g() + 0.1804375f*C.b(),
    0.2126729f*C.r() + 0.7151522f*C.g() + 0.0721750f*C.b(),
    0.0193339f*C.r() + 0.1191920f*C.g() + 0.9503041f*C.b()); }
constexpr linRGB XYZ2linRGB( XYZ const &C )
{ return linRGB(
     3.2404542f*C.c[0] - 1.5371385f*C.c[1] - 0.4985314f*C.c[2],
    -0.9692660f*C.c[0] + 1.8760108f*C.c[1] + 0.0415560f*C.c[2],
     0.0556434f*C.c[0] - 0.2040259f*C.c[1] + 1.0572252f*C.c[2]); }
constexpr sRGB linRGB2sRGB(linRGB const &x)
{
  auto f=[](float cl){ return cl>0.0031308f ? 
    1.055f*std::pow(cl,1.f/2.4f)-0.055f : 12.92f*cl; };
  return sRGB(f(x.c[0]), f(x.c[1]), f(x.c[2]));
}
constexpr linRGB sRGB2linRGB(sRGB const &x)
{
  auto f=[](float c){ return c > 0.04045f ? 
    std::pow( (c+0.055f)/1.055f, 2.4f ) : c / 12.92f;};
  return linRGB(f(x.c[0]), f(x.c[1]), f(x.c[2]));
}
constexpr rgb24 sRGB2rgb24(sRGB const &x) 
  { return rgb24(float2byte(x.c[0]),float2byte(x.c[1]),float2byte(x.c[2])); }

// ** end of RGB **********************


// ************************************
/// @name spectrums

/// @brief Sampled Spectrum - values (λ,v) 
struct sampledλ 
{
  std::vector<float> l; ///< wavelengths
  std::vector<float> v; ///< values
  sampledλ(std::string fpath)
  { // loads from csv file
    std::ifstream file(fpath);
    std::string line;
    while (std::getline(file, line))
    {
      std::stringstream ss(line);
      std::string λstr, vstr;
      std::getline(ss, λstr, ',');
      std::getline(ss, vstr, ',');
      l.emplace_back(std::stof(λstr));
      v.emplace_back(std::stof(vstr));
    }
  }
  sampledλ(std::string fpath, float minλ, float stepλ, uint8_t i=0)
  { // loads spectrums from csv without wavelength values
    std::ifstream file(fpath);
    std::string line;
    uint64_t n = 0;
    while (std::getline(file,line))
    {
      l.emplace_back(minλ+n*stepλ);
      n++;
      std::stringstream ss(line);
      std::string vstr;
      for (int skips=0; skips<=i; skips++) { std::getline(ss, vstr, ','); }
      v.emplace_back(std::stof(vstr));
    }
  }
};


/// @brief Coeffiecient Spectrum - values spaced every Δ in nm
template <uint32_t n> struct coefficientλ 
{
  float const inf=400;
  float const sup=720;
  float const Δ=(sup-inf)/(n-1);
  bra::ℝn<n,float> v;
  constexpr coefficientλ() : v(0.f) {}
  constexpr coefficientλ(float x) {for (uint32_t i=0;i<n;i++) {v[i]=x;}}
  constexpr coefficientλ(coefficientλ const& s) : v(s.v) {}
  /// @warning assumes samples are in sorted order
  coefficientλ(sampledλ const &s)
  {
    size_t j=0;
    for (uint32_t i=0;i<n;i++)
    { // v[i] is the average of sampledλ <= l[i] but > l[i-1]
      float const l0=inf+i*Δ;
      float const l1=l0+Δ;
      float sum=0.f;
      while (j+1<s.l.size() && s.l[j+1]<=l0) j++;
      size_t k=j;
      while (k+1<s.l.size() && s.l[k]<l1)
      {
        float seg0 = max(l0, s.l[k]);
        float seg1 = min(l1, s.l[k+1]);
        float seg  = seg1-seg0;
        if (seg>0) { sum += .5f*(s.v[k]+s.v[k+1])*seg; }
        k++;
      }
      v[i] = sum/Δ;
    }
  }
  constexpr coefficientλ operator*(coefficientλ const &s2) const
    { coefficientλ s=*this; s.v*=s2.v; return s; }
  constexpr coefficientλ operator*(float x) const
    { coefficientλ s=*this; s.v=s.v*x; return s; }
  constexpr coefficientλ operator+(coefficientλ const &s2) const
    { coefficientλ s=*this; s.v=s.v+s2.v; return s; }
  constexpr float operator|(coefficientλ const &s) const
  {
    float sum=0.f;
    for (uint32_t i=0;i<n;i++) { sum+=v[i]*s.v[i]; }
    return sum*Δ;
  }
  constexpr float operator()(float l) const
  { // interpolates spectrum coefficients for given value
    if (l<=inf) return v[0];
    if (l>=sup) return v[n-1];
    float x = (l-inf)/Δ;
    uint32_t i = static_cast<uint32_t>(x);
    float t=x-i;
    return (1.f-t)*v[i] + t*v[i+1];
  }
  coefficientλ& operator=(coefficientλ&& o) noexcept 
    { for (uint32_t i=0;i<n;i++) {v[i]=std::move(o.v[i]);} return *this; }
  std::string toString() const 
  {
    std::string s=""; 
    for (uint32_t i=0;i<n;i++) {s+=std::format("{:.2f}",v[i]);} 
    return s;
  }
};

inline coefficientλ<Nλ> CMF_X(sampledλ("../lib/numli/data/xyz_x.csv"));
inline coefficientλ<Nλ> CMF_Y(sampledλ("../lib/numli/data/xyz_y.csv"));
inline coefficientλ<Nλ> CMF_Z(sampledλ("../lib/numli/data/xyz_z.csv"));
inline coefficientλ<Nλ> bt_709_r(
  sampledλ("../lib/numli/data/cie1931-basis-bt709-380+5+780.csv",380,5,0));
inline coefficientλ<Nλ> bt_709_g(
  sampledλ("../lib/numli/data/cie1931-basis-bt709-380+5+780.csv",380,5,1));
inline coefficientλ<Nλ> bt_709_b(
  sampledλ("../lib/numli/data/cie1931-basis-bt709-380+5+780.csv",380,5,2));

constexpr XYZ λ2XYZ(float l) { return {CMF_X(l), CMF_Y(l), CMF_Z(l)};}
constexpr linRGB coefλ2linRGB(coefficientλ<Nλ> const &spec)
  { return XYZ2linRGB({spec|CMF_X,spec|CMF_Y,spec|CMF_Z}); }
constexpr coefficientλ<Nλ> linRGB2coefλ(linRGB c)
  { return bt_709_r*c.r()+bt_709_g*c.g()+bt_709_b*c.b(); }
// ** end of spectrums ****************
// ** end of light ************************************************************


// ****************************************************************************
/// @name spatial

struct vec 
{
  ℝ4 dir;
  constexpr vec() {dir[3]=0.f;}
  constexpr vec(float const (&x)[3]) 
    { for (size_t i=0;i<3;i++) dir[i]=x[i]; dir[3]=0.f; }
  constexpr vec(vec const &v) {for(size_t i=0;i<4;i++)dir[i]=v.dir[i];}
  constexpr vec(ℝ3 const &x) {for(size_t i=0;i<3;i++)dir[i]=x[i];dir[3]=0.f;}
  constexpr vec(ℝ4 const &x) {for(size_t i=0;i<3;i++)dir[i]=x[i];dir[3]=0.f;}
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
  ℝ3 inv_u;
  ray() {}
  ray(ℝ3 const &p, ℝ3 const &u) : p(p), u(u) 
    { for (int i=0;i<3;i++) inv_u[i]=1.f/u[i]; }
  ray(ℝ4 const &p, ℝ4 const &u) : 
    p({p.elem[0],p.elem[1],p.elem[2]}), u({u.elem[0],u.elem[1],u.elem[2]}) 
    { for (int i=0;i<3;i++) inv_u[i]=1.f/u[i]; }
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
  static constexpr transform rotate(ℝ3 const &axis, float degrees)
  { // using Rodrigues' formula R(u,θ) = cosθI+(1-cosθ)*outer(u)-sinθ*skew(u)
    ℝ3 const u = axis.normalized();
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
  void init() { inf=1e30f; sup=-1e30f; }
  void operator+=(aabb const &b) 
  {
    for (int i=0; i<3; i++) 
      { inf[i]=min(inf[i],b.inf[i]); sup[i]=max(sup[i],b.sup[i]); }
  }
};
struct aabb_avx2
{
  alignas(32) float inf0[8], sup0[8];
  alignas(32) float inf1[8], sup1[8];
  alignas(32) float inf2[8], sup2[8];
  aabb_avx2() = default;
  aabb_avx2(aabb (&bb)[8])
  {
    for (int i=0; i<8; i++) 
    { // copy data from bb into member variables
      inf0[i]=bb[i].inf[0];
      inf1[i]=bb[i].inf[1];
      inf2[i]=bb[i].inf[2];
      sup0[i]=bb[i].sup[0];
      sup1[i]=bb[i].sup[1];
      sup2[i]=bb[i].sup[2];
    }
  }
};

struct triangle
{
  uint32_t V[3], N[3], u[3];
};

/// @name BVH constants
constexpr uint8_t PRIM_PER_LEAF = 4;
constexpr uint32_t IDX_MASK   = 0xfffffff0;
constexpr uint32_t COUNT_MASK = 0x0000000e;
constexpr uint32_t LEAF_MASK  = 0x00000001;

/// @brief generic BVH over primatives T with mapping from T to aabb bbFunc
/// @tparam T type of the primatives to index over
/// @tparam bbFunc callable mapping from T to aabb for each T
/// @tparam centerFunc callable mapping from T to centroid for each T
/// @warning Leaf nodes can have up to 8 primatives in them
template <typename T, typename bbFunc, typename centerFunc> struct bvh 
{
  /// @name mappings from primative T to aabb or centroid
  bbFunc aabbof;
  centerFunc centroidof;

  /////////////////////////////////////
  /// @name internal data structures

  struct bvhbuildnode
  {
    aabb bb;
    std::array<bvhbuildnode*,8> children{};
    std::vector<uint32_t> prim_idxs;
    bool leaf = false;

    bvhbuildnode(aabb const &box, std::vector<uint32_t> const &idxs) 
      { bb=box; prim_idxs=idxs; }
    ~bvhbuildnode() { for (auto child : children) delete child; }
  };

  struct bvhnode 
  {
    uint32_t childrenbb_idx;
    uint32_t data;

    // idx and count: 
    // index and number of children (internal node) or prims (leaf node)
    inline uint32_t idx() const { return (IDX_MASK&data)>>4; }
    inline uint8_t count() const { return ((COUNT_MASK&data)>>1)+1; }
    inline bool isLeaf() const { return (LEAF_MASK&data) == 1; }

    /// @warning count must be between 1 and 8
    bvhnode() : childrenbb_idx(UB<uint32_t>), data(0) {}
    bvhnode(uint32_t bb_idx, uint32_t idx, uint8_t count, bool leaf) 
      : childrenbb_idx(bb_idx)
      { data = (idx<<4) | ((count-1)<<1) | (leaf ? 1 : 0); }
  };

  /////////////////////////////////////
  /// @name member variables

  std::vector<T> const &prims;     ///< ref to vector of primitives
  std::vector<uint32_t> prim_idxs; ///< indices into prims array
  std::vector<aabb_avx2> bbs;      ///< bounding box storage
  std::vector<bvhnode> nodes;      ///< tree structure stored as array
  uint32_t n_nodes = 0;
  uint32_t n_interior_nodes = 0;

  /////////////////////////////////////
  /// @name member functions

  /// @brief splits this buildnode into 8 child nodes
  /// @details splits the bb into 8 boxes along the longest axis
  void splitbuildnode(bvhbuildnode *node)
  {
    if (node==nullptr) return;
    n_nodes++;

    // check if this node should be a leaf node
    if (node->prim_idxs.size() <= PRIM_PER_LEAF) { node->leaf=true; return; }

    // must be an interior node
    n_interior_nodes++;

    // find splitting axis
    ℝ3 const l = node->bb.sup-node->bb.inf;
    uint8_t i=0;
    if (l[1]>l[0]) i=1;
    if (l[2]>l[1] && l[2]>l[0]) i=2;

    std::vector<uint32_t> child_prims[8];
    for (uint32_t e : node->prim_idxs)
    { // iterate through elements
      float c = centroidof(prims[e],i);
      int bin = min(7,int(8*(c-node->bb.inf[i])/l[i]));
      child_prims[bin].push_back(e);
    }

    size_t n = node->prim_idxs.size();
    for (int bin = 0; bin < 8; bin++)
    { // check for bin with all triangles
      if (child_prims[bin].size() != n) continue; // good bin
      
      std::vector<uint32_t> temp = std::move(child_prims[bin]);
      size_t chunk_size = (n+7)/8;
      for (int b=0; b<8; b++)
      { // distribute across children
        size_t start = b*chunk_size;
        size_t end = min(start+chunk_size, n);
        if (start < end) child_prims[b] = 
          std::vector<uint32_t>(temp.begin()+start, temp.begin()+end);
      }
      break; // done redistributing
    }

    for (int bin=0; bin<8; bin++)
    { // check for empty, compute aabb, recurse
      if (child_prims[bin].empty()) continue;
      aabb child_bb;
      child_bb.init();
      for (uint32_t idx : child_prims[bin]) { child_bb += aabbof(prims[idx]); }
      node->children[bin] = new bvhbuildnode(child_bb, child_prims[bin]);
      splitbuildnode(node->children[bin]);
    }
  }

  /// @brief converts the bvh from buildnodes to nodes
  /// @param root pointer to root buildnode
  void convertbuildnodes(bvhbuildnode *root)
  {
    nodes.resize(n_nodes);
    bbs.resize(n_interior_nodes);

    std::queue<std::pair<bvhbuildnode*,uint32_t>> queue; // node, index
    queue.push({root, 0});
    uint32_t next_free = 1; // 0 reserved for root

    while (!queue.empty())
    { // breadth-first traveral of bvhbuildnode tree
      auto [node, idx] = queue.front();
      queue.pop();

      if (node->leaf)
      { // leaf node, store primative indices
        uint32_t prim_idx = prim_idxs.size();
        uint8_t count = node->prim_idxs.size();
        for (uint32_t idx : node->prim_idxs) { prim_idxs.push_back(idx); }
        nodes[idx] = {UB<uint32_t>, prim_idx, count, true};
      } else 
      { // interior node, pack child_bb, data, push children to stack
        uint32_t first_child_idx = next_free;
        uint8_t n_child = 0;
        aabb child_bbs[8];
        for (int i=0; i<8; i++)
        { // check each potential child
          child_bbs[i].init();
          if (node->children[i])
          { // if there is a child there
            child_bbs[n_child] = node->children[i]->bb;
            queue.push({node->children[i], next_free+n_child});
            n_child++;
          }
        }
        next_free += n_child;
        uint32_t bbs_idx = bbs.size();
        bbs.emplace_back(child_bbs);
        nodes[idx] = {bbs_idx, first_child_idx, n_child, false};
      }
    }
  }

  /// @brief builds the bvh
  /// @param N number of prims the current node needs to split
  void build(uint32_t N)
  {
    n_nodes = 0;
    n_interior_nodes = 0;
    if (N==0) return;
    aabb bb;
    bb.init();
    std::vector<uint32_t> temp_idxs;
    for (uint32_t i=0;i<N;i++)
    {
      bb += aabbof(prims[i]);
      temp_idxs.push_back(i);
    }
    bvhbuildnode *temp_root = new bvhbuildnode(bb, temp_idxs);
    splitbuildnode(temp_root);
    convertbuildnodes(temp_root);
    delete temp_root;
  }

  uint32_t root() const { return 0; }
  bool leafNode(uint32_t node) const { return nodes[node].isLeaf(); }
  uint8_t count(uint32_t node) const { return nodes[node].count(); }
  const aabb_avx2& childAABB(uint32_t node) const 
    { return bbs[nodes[node].childrenbb_idx]; }
  uint32_t childOf(uint32_t node, uint8_t child) const 
    { return nodes[node].idx()+child; }
  uint32_t primOf(uint32_t node, uint8_t i) const 
    { return prim_idxs[nodes[node].idx()+i]; }


  bvh(std::vector<T> const &primatives, bbFunc f, centerFunc g) 
    : aabbof(f), centroidof(g), prims(primatives) {}
};

struct trimeshdata : item
{
  aabb bounds;
  std::vector<ℝ3>       V;     ///< vertices
  std::vector<ℝ3>       N;     ///< vertex normals
  std::vector<ℝ3>       T;     ///< vertex tangent vectors
  std::vector<ℝ2>       u;     ///< vertex texture coords
  std::vector<ℝ3>       GN;    ///< face normals
  std::vector<ℝ3>       GT;    ///< face tangents
  std::vector<triangle> F;     ///< faces (triangles)
  std::function<aabb(triangle const&)> aabboftri=[&](triangle const &tri)
  {
    aabb bb; 
    bb.init(); 
    for(int k=0;k<3;k++)
    {
      ℝ3 const &v = V[tri.V[k]];
      for (int i=0;i<3;i++) 
        { bb.inf[i]=min(bb.inf[i],v[i]); bb.sup[i]=max(bb.sup[i],v[i]); }
    }
    return bb;
  };
  std::function<float(triangle const&, int i)> centroidoftri =
    [&](triangle const &tri,int i)
    { return (V[tri.V[0]][i]+V[tri.V[1]][i]+V[tri.V[2]][i])/3.f; };
  bvh<triangle, decltype(aabboftri), decltype(centroidoftri)> _bvh;
  void buildBVH() { _bvh.build(F.size()); }
  constexpr ℝ3 const& v(uint32_t face, int i) const {return V[F[face].V[i]];}
  constexpr ℝ3 const& t(uint32_t face, int i) const {return T[F[face].N[i]];}
  constexpr ℝ2 const& uv(uint32_t face, int i) const {return u[F[face].u[i]];}
  constexpr ℝ3 const& gn(uint32_t face) const {return GN[face];}
  constexpr ℝ3 const& gt(uint32_t face) const {return GT[face];}
  constexpr ℝ3 n(uint32_t face, ℝ3 const &b) const
  { 
    triangle const &tri = F[face]; 
    return b[0]*N[tri.N[0]]+b[1]*N[tri.N[1]]+b[2]*N[tri.N[2]]; 
  }
  constexpr ℝ3 t(uint32_t face, ℝ3 const &b) const
  {
    triangle const &tri = F[face];
    return b[0]*T[tri.N[0]]+b[1]*T[tri.N[1]]+b[2]*T[tri.N[2]];
  }
  constexpr ℝ2 uv(uint32_t face, ℝ3 const &b) const
  {
    triangle const &tri = F[face];
    return b[0]*u[tri.u[0]]+b[1]*u[tri.u[1]]+b[2]*u[tri.u[2]];
  }

  trimeshdata() : _bvh(F, aabboftri, centroidoftri) {}
  trimeshdata(trimeshdata&& o) noexcept
    : item(std::move(o)), bounds(o.bounds), 
    V(std::move(o.V)), N(std::move(o.N)), T(std::move(o.T)), u(std::move(o.u)),
    GN(std::move(o.GN)), GT(std::move(o.GT)), F(std::move(o.F)), 
    _bvh(std::move(o._bvh)) {}
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
enum class LightType {AMBIENT, POINT, DIR, SPHERE, QUADDIR};
constexpr LightType str2light(std::string s)
{
  if (s=="ambient")   return LightType::AMBIENT;
  if (s=="point")     return LightType::POINT;
  if (s=="direction") return LightType::DIR;
  if (s=="sphere")    return LightType::SPHERE;
  if (s=="quaddir")   return LightType::QUADDIR;
  throw std::domain_error("No corresponding LightType.");
}

struct ambientlight : item
{
  coefficientλ<Nλ> radiance;
};
struct pointlight : item
{
  coefficientλ<Nλ> radiant_intensity;
  ℝ3 pos;
};
struct dirlight : item
{
  coefficientλ<Nλ> radiant_intensity;
  ℝ3 dir;
};
struct spherelight : item
{
  coefficientλ<Nλ> radiance;
  float size;
  sphere _sphere;
};
struct quaddirlight : item
{
  coefficientλ<Nλ> radiance;
  float inv_area;
  float spec;
  plane _plane;
};

/// @brief Light interface
using Light = 
  std::variant<ambientlight, pointlight, dirlight, spherelight, quaddirlight>;
// ** end of lights ***********************************************************

struct hitinfo;
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
  coefficientλ<Nλ> albedo;
  constexpr float BRDFcosθ(float l, ℝ3 const &i, ℝ3 const &n) const 
    { return albedo(l)*inv_π<float>*nl::max(0.f,i|n); }
};
struct blinn : item
{
  coefficientλ<Nλ> Kd, Ks, Kt, Le, ior;
  float α, p_d, p_spec, w_r, w_t;
  constexpr void init() 
  { 
    p_d    = Kd|Kd;
    w_r    = Ks|Ks;
    w_t    = Kt|Kt;
    p_spec = 1.f-p_d;
  }
  constexpr float F0(float n) const
    { return (n*n-2.f*n+1.f)/(n*n+2.f*n+1.f); }
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
  constexpr float fdcosθ(float l, float cos_i) const 
    { return Kd(l)*inv_π<float>*nl::max(0.f,cos_i); }
  constexpr float frcosθ(float l, float cos_h, float F) const
    { return (Ks(l)+F*Kt(l))*(α+2)*.125f*inv_π<float>*std::pow(cos_h,α); }
  constexpr float ftcosθ(float l, float cos_h_t, float F) const
    { return Kt(l)*(1-F)*(α+2)*.125f*inv_π<float>*std::pow(cos_h_t,α); }
  constexpr float BSDFcosθ(
    float l, ℝ3 const &i, ℝ3 const &o, ℝ3 const &n, bool front) const 
  { 
    float const η = front ? 1.f/ior(l) : ior(l);
    float const cos_i = i|n;
    float const cos_o = o|n;
    float const F = fresnel(F0(η), cos_o);
    if (cos_i>0.f)
    { 
      ℝ3 const h = (i+o).normalized();
      return fdcosθ(l, cos_i) + frcosθ(l, h|n,F);
    } else 
    { 
      ℝ3 const ht = -(i+η*o).normalized();
      return ftcosθ(l, std::abs(ht|n),F);
    }
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
  coefficientλ<Nλ> albedo, ior;
  float αu, αv;
  float fresnel_g(float n, float c) const // n is n_i/n_o
    { float q=n*n-1+c*c; return q>0.f ? std::sqrtf(q) : -1.f; }
  float F(float n, float c) const // c is abs(i|m)
  { 
    float const g=fresnel_g(n,c);
    if (g==-1.f) return 1.f;
    float const gmc = g-c;
    float const gpc = g+c;
    return .5f*gmc*gmc*(1+(c*gpc-1)*(c*gpc-1)/((c*gmc+1)*(c*gmc+1)))/(gpc*gpc);
  }
};
struct emitter : item
{
  coefficientλ<Nλ> radiance;
  constexpr float Radiance(float l) const
    { return radiance(l); }
};
struct diremitter : item
{
  coefficientλ<Nλ> radiance;
  float specular;
  constexpr float Radiance(float l, ℝ3 const &o, ℝ3 const &n) const
    { return radiance(l)*std::pow(max(o|n,0.f),specular); }
};

/// @brief Material interface
using Material = 
  std::variant<lambertian, blinn, microfacet, emitter, diremitter>;

/// @brief Evaluation of BSDF times geometry term
constexpr float BxDFcosθ(
  float l,
  Material const &mat, 
  ℝ3 const &i, 
  ℝ3 const &o, 
  ℝ3 const &n,
  bool front)
{
  return std::visit(Overload{
    [&](lambertian const &mat){return mat.BRDFcosθ(l,i,n);},
    [&](blinn const &mat)     {return mat.BSDFcosθ(l,i,o,n,front);},
    [](auto const &){return 0.f;}},
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
  ℝ3 gn;
  bool front;
  materialidx mat;
  objectidx obj;
  constexpr ℝ3 n() const { return front ? F.z : -F.z; }
  constexpr ℝ3 geoN() const { return front ? gn : -gn; }
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
  /// @todo provide Object aabb function for obvh
  // bvh<Object, > obvh;

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
  float const b = 2*(l_ray.p|l_ray.u);
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
  ℝ3 const n = s.T.toWorld(normal(p)).normalized();
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
  hinfo.gn = n;
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
  hinfo.F.x = p.T.toWorld(normal({0.f,1.f,0.f})).normalized();
  hinfo.F.y = hinfo.F.z^hinfo.F.x;
  hinfo.gn  = hinfo.F.z;
  hinfo.front = (w_ray.u|hinfo.F.z) < 0.f;
  hinfo.mat = p.mat;
  hinfo.obj = p.obj;
  return true;
}

/// @brief aabb intersection
constexpr float aabb(cg::aabb const &bb, ray const &l_ray, float t_max)
{
  float const inv_x = l_ray.inv_u[0];
  float const inv_y = l_ray.inv_u[1];
  float const inv_z = l_ray.inv_u[2];
  float t0x = (bb.inf[0]-l_ray.p[0])*inv_x;
  float t1x = (bb.sup[0]-l_ray.p[0])*inv_x;
  float t0y = (bb.inf[1]-l_ray.p[1])*inv_y;
  float t1y = (bb.sup[1]-l_ray.p[1])*inv_y;
  float t0z = (bb.inf[2]-l_ray.p[2])*inv_z;
  float t1z = (bb.sup[2]-l_ray.p[2])*inv_z;
  if (t0x>t1x) std::swap(t0x,t1x);
  if (t0y>t1y) std::swap(t0y,t1y);
  if (t0z>t1z) std::swap(t0z,t1z);
  float const tmin = max(t0x,max(t0y,t0z));
  float const tmax = min(t1x,min(t1y,t1z));
  if (tmax<BIAS || tmax<=tmin || tmin>t_max) return UB<float>;
  return tmin<BIAS ? tmax : tmin;
}

/// @brief 8 aabb intersection using avx2 registers
/// @param bbs 32 byte aligned 8-way aabb
/// @param o broadcasted ℝ3 l_ray.o
/// @param inv_u broadcasted ℝ3 l_ray.inv_u
/// @param t_max maximum distance along ray to consider for intersections
/// @param t return value array
/// @warning aabb with no intersection are given t=0 NOT t=UB<float>
TARGET_AVX2 inline void aabb_avx2(
  const cg::aabb_avx2 &bbs, 
  const __m256 (&p)[3],
  const __m256 (&inv_u)[3], 
  float t_max, 
  float (&t)[8])
{
  // broadcast min and max t
  __m256 tmin = _mm256_set1_ps(BIAS);
  __m256 tmax = _mm256_set1_ps(t_max);

  // load boxes
  __m256 inf0 = _mm256_load_ps(bbs.inf0);
  __m256 inf1 = _mm256_load_ps(bbs.inf1);
  __m256 inf2 = _mm256_load_ps(bbs.inf2);
  __m256 sup0 = _mm256_load_ps(bbs.sup0);
  __m256 sup1 = _mm256_load_ps(bbs.sup1);
  __m256 sup2 = _mm256_load_ps(bbs.sup2);

  // compute distances to planes
  __m256 t0x = _mm256_mul_ps(_mm256_sub_ps(inf0, p[0]), inv_u[0]);
  __m256 t1x = _mm256_mul_ps(_mm256_sub_ps(sup0, p[0]), inv_u[0]);
  __m256 t0y = _mm256_mul_ps(_mm256_sub_ps(inf1, p[1]), inv_u[1]);
  __m256 t1y = _mm256_mul_ps(_mm256_sub_ps(sup1, p[1]), inv_u[1]);
  __m256 t0z = _mm256_mul_ps(_mm256_sub_ps(inf2, p[2]), inv_u[2]);
  __m256 t1z = _mm256_mul_ps(_mm256_sub_ps(sup2, p[2]), inv_u[2]);

  // find near and far intersection for each axis
  __m256 tnx = _mm256_min_ps(t0x, t1x);
  __m256 tfx = _mm256_max_ps(t0x, t1x);
  __m256 tny = _mm256_min_ps(t0y, t1y);
  __m256 tfy = _mm256_max_ps(t0y, t1y);
  __m256 tnz = _mm256_min_ps(t0z, t1z);
  __m256 tfz = _mm256_max_ps(t0z, t1z);

  // find in and out t vals
  __m256 tin  = _mm256_max_ps(tnz, _mm256_max_ps(tnx, tny));
  __m256 tout = _mm256_min_ps(tfz, _mm256_min_ps(tfx, tfy));
  tin  = _mm256_max_ps(tin, tmin); 
  tout = _mm256_min_ps(tout, tmax);

  // zero out non-intersections, and store result
  _mm256_store_ps(t, _mm256_and_ps(_mm256_cmp_ps(tin, tout, _CMP_LE_OQ), tin));
}

/// @brief Triangle-Ray intersection
/// @warning hinfo.mat and hinfo.obj are NOT set
/// @todo fix plucker coordinate intersection
constexpr bool trimeshdata(
  uint64_t face, cg::trimeshdata const &mesh, ray const &l_ray, hitinfo &hinfo)
{
  ℝ3 const &v0 = mesh.v(face,0);
  ℝ3 const &v1 = mesh.v(face,1);
  ℝ3 const &v2 = mesh.v(face,2);
  ℝ3 const &e0 = v1-v0;
  ℝ3 const &e1 = v2-v1;
  ℝ3 const &e2 = v0-v2;
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
  if (t<0.f || hinfo.z<t) { return false; } // behind ray or not closest hit

  // barycentric coordinates
  float const tot = 1.f/D;
  ℝ3 const b(s0*tot, s1*tot, s2*tot);

  // populate hinfo
  hinfo.z = t;
  hinfo.p = l_ray(t);
  hinfo.gn = gn;
  hinfo.F.z = mesh.n(face,b);
  hinfo.F.x = mesh.t(face,b);
  hinfo.front = (gn|l_ray.u)<0.f;
  return true;
}

// Woop et al. "Watertight Ray/Triangle Intersection" 
// Journal of Computer Graphics Techniques. Vol. 2, No. 1, 2013
inline bool trimeshdata_koi( 
  unsigned int faceID,
  cg::trimeshdata const &mesh,
  ray const &l_ray, 
  hitinfo &hInfo)
{
  const auto face = mesh.F[faceID];
  ℝ3 const &v0  = mesh.V[face.V[0]];
  ℝ3 const &v1  = mesh.V[face.V[1]];
  ℝ3 const &v2  = mesh.V[face.V[2]];
  ℝ3 const v10 = v1-v0;
  ℝ3 const v20 = v2-v0;
  ℝ3 const pv0 = l_ray.p-v0;
  ℝ3 const   n = v10 ^ v20;
  float const udotn = l_ray.u|n;
  float const det = -1.f/udotn;

  // compute t, u, v, check for early exits
  float const t = det*pv0|n;
  if (t<BIAS || t>hInfo.z) [[likely]] { return false; }
  float const b1 = det * l_ray.u | (v20 ^ pv0);
  if (b1<0) [[unlikely]] { return false; }
  float const b2 = det * l_ray.u | (pv0 ^ v10);
  if (b2<0 || b1+b2>1) [[unlikely]] { return false; }
  
  // triangle is hit
  float const b0 = 1-b1-b2;
  ℝ3 const b = {b0, b1, b2}; 
  hInfo.gn = mesh.gn(faceID);
  hInfo.F.z = mesh.n(faceID,b);
  hInfo.F.x = mesh.t(faceID,b);
  hInfo.front = (mesh.GN[faceID] | l_ray.u) < 0;
  hInfo.p = l_ray.p + t*l_ray.u;
  hInfo.z = t;
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
  ray const l_ray = tmesh.T.toLocal(w_ray);

  /// bounding box check
  float t = aabb(mesh.bounds, l_ray, hinfo.z);
  if (t == UB<float>) return false;

  // prep local ray for bb intersections
  __m256 p[3];
  __m256 inv_u[3];
  for (int i=0; i<3; i++)
  {
    p[i] = _mm256_set1_ps(l_ray.p[i]);
    inv_u[i] = _mm256_set1_ps(l_ray.inv_u[i]);
  }

  // bvh traversal
  const auto &BVH = mesh._bvh;
  static constexpr uint8_t STACK_SIZE = 64;
  uint8_t sp = 0;
  std::pair<uint32_t, float> stack[STACK_SIZE]; ///< stack of pairs (node,t)
  stack[sp++] = {BVH.root(), t};

  // lambda for aabb sorting
  static constexpr auto further = [](auto const& a, auto const& b)
    { return a.second > b.second; };

  while (sp > 0)
  { // traverse bvh depth first
    const auto& [node, t] = stack[--sp];
    if (t > hinfo.z) continue;  // hit behind closest hit so far, skip

    if (!BVH.leafNode(node))
    { // if node is internal, push intersected children onto stack
      // simd 8-wide box intersection
      alignas(32) float t_children[8];
      aabb_avx2(BVH.childAABB(node), p, inv_u, hinfo.z, t_children);
      
      // find and sort intersections, push onto stack so closest is on top
      const uint8_t n = BVH.count(node);
      std::pair<uint32_t,float> vals[n];
      uint8_t hits = 0;
      for (int i=0; i<n; i++) 
      { 
        if (t_children[i]!=0.f)
          { vals[hits++] = {BVH.childOf(node,i), t_children[i]}; }
      }
      std::sort(vals, vals+hits, further);
      for (int i=0; i<hits; i++) { stack[sp++] = vals[i]; }
      continue;
    }

    const uint8_t count = BVH.count(node);
    for (uint8_t i=0; i<count; i++)
    { // leaf node, test for triangle intersection
      hit_any |= trimeshdata_koi(BVH.primOf(node,i), mesh, l_ray, hinfo);
    }
  }
  if (!hit_any) return false;

  // convert hinfo to world-space
  hinfo.p   = tmesh.T.toWorld(pnt(hinfo.p));
  hinfo.gn  = tmesh.T.toWorld(normal(hinfo.gn)).normalized();
  hinfo.F.z = tmesh.T.toWorld(normal(hinfo.F.z)).normalized();
  hinfo.F.x = tmesh.T.toWorld(normal(hinfo.F.x)).normalized();
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

/// @brief samples a wavelength λ from the spectrum s
template <uint32_t n>
inline void spectrum(cg::coefficientλ<n> const &s, info<float> &info, RNG &rng)
{
  float const Ω = s.sup-s.inf;
  info.val = s.inf+Ω*rng.flt();
  info.prob = 1./Ω;
}

/// @brief ray from camera c through SS (s[0],s[1]) from disk (s[2],s[3])
inline void camera(cg::camera const &c, ℝ2 const &uv, info<ray> &info,RNG &rng)
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

/// @brief samples an ambient light at wavelength l
inline void ambientlight(
  float l,
  cg::ambientlight const &al,
  hitinfo const &hinfo,
  info<ℝ3,float> &info,
  RNG &rng)
{
  // get direction in hemisphere
  ℝ3 dir = stoch::UnifHemi(rng.flt(),rng.flt());
  if (!hinfo.front) dir[2]*=-1.f; // flip to correct hemisphere
  ℝ3 const i = hinfo.F.toBasis(dir);
  info.val = i;
  info.prob = .5f*inv_π<float>;
  float const radiance = al.radiance(l);
  info.mult = radiance;
  info.weight = 2.f*π<float>*radiance;
}
inline float probForAmbientLight() { return .5f*inv_π<float>; }

/// @brief samples a point light at wavelength l
inline void pointlight(
  float l, 
  cg::pointlight const &pl, 
  hitinfo const &hinfo, 
  scene const &sc, 
  info<ℝ3,float> &info)
{
  ℝ3 const L = pl.pos-hinfo.p;
  ℝ3 const i = L.normalized();
  info.val = i;
  info.prob = 1.f;

  // check shadowing
  float const dist = L.l2();
  hitinfo temp;
  temp.z = dist;
  float const shadowing = 
    intersect::scene(sc,{hinfo.p+.05f*hinfo.n(),i},temp) ? 0.f : 1.f;
  info.mult = pl.radiant_intensity(l)*shadowing/(dist*dist);
  info.weight = info.mult;
}
constexpr float probForPointLight() {return 0.f;}

/// @brief samples the radiant intensity coming from a direction light
inline void dirlight(
  float l, 
  cg::dirlight const &dl,
  hitinfo const &hinfo,
  scene const &sc,
  info<ℝ3,float> &info)
{
  ℝ3 const i = -dl.dir;
  info.val = i;
  info.prob = 1.f;

  // check shadowing
  hitinfo temp;
  float const shadowing = 
    intersect::scene(sc,{hinfo.p+.05f*hinfo.n(),i},temp) ? 0.f : 1.f;
  info.mult = dl.radiant_intensity(l)*shadowing;
  info.weight = info.mult;
}
constexpr float probForDirLight() {return 0.f;}

/// @brief uniformly samples a point of a quad dir light from a point
inline void quaddirlight(
  float l,
  cg::quaddirlight const &qdl,
  hitinfo const &hinfo,
  scene const &sc,
  info<ℝ3,float> &info,
  RNG &rng)
{
  // sample point on plane, then get ωi
  float sx = 2.f*rng.flt()-1.f;
  float sy = 2.f*rng.flt()-1.f;
  const ℝ3 x = qdl._plane.T.pos() + 
    sx*bra::column<3>(qdl._plane.T.M, 0)+sy*bra::column<3>(qdl._plane.T.M,1);
  ℝ3 const r = x-hinfo.p;
  ℝ3 const ωi = r.normalized();
  info.val = ωi;

  // compute prob by converting from sr at hit to area at light
  float const dist2 = r|r;
  float const cos_spec = 
    std::pow(-ωi|(bra::column<3>(qdl._plane.T.M,2).normalized()), qdl.spec);
  info.prob = 
    max(1e-6f,(qdl.spec+1)*.5f*inv_π<float>*cos_spec*qdl.inv_area);

  // get L(ωi)
  hitinfo temp;
  temp.z = std::sqrtf(dist2);
  float const shadowing = intersect::scene(
    sc, {hinfo.p+.05f*hinfo.n(),ωi}, temp, qdl._plane.obj) ? 0.f : 1.f;
  info.mult = qdl.radiance(l)*cos_spec*shadowing;
}
/// @brief probability that a ray could be generated by sampling a qdl
inline float probForQuadDirLight(cg::quaddirlight const &qdl, ray const &r)
{
  // check if ray hits plane
  hitinfo hinfo;
  bool hit = intersect::plane(qdl._plane, r, hinfo);
  if (!hit || (hit && !hinfo.front)) return 0.f;

  // direction could be generated, compute prob
  float const cosθ = -r.u.normalized()|hinfo.F.z;
  return max(1e-6f, (qdl.spec+1)*.5f*inv_π<float>*std::pow(cosθ,qdl.spec)
    * qdl.inv_area);
}

/// @brief uniformly samples the solid angle of a sphere light from a point
inline void spherelight(
  float l,
  cg::spherelight const &sl, 
  hitinfo const &hinfo, 
  scene const &sc,
  info<ℝ3,float> &info, 
  RNG &rng) 
{
  // compute probability for ωi
  ℝ3 const L = sl._sphere.T.pos()-hinfo.p;
  float const dist2 = L|L;
  float const r2 = sl.size*sl.size;
  float const sr = 1.f-std::sqrtf(1.f-r2/dist2);
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
  float const shadowing = intersect::scene(
    sc, {hinfo.p+.05f*hinfo.n(),ωi}, temp, sl._sphere.obj) ? 0.f : 1.f;
  info.mult = sl.radiance(l)*shadowing;
}
/// @brief probability that a ray (direction from a hit point) could be sampled
inline float probForSphereLight(cg::spherelight const &sl, ray const &r)
{
  // check for intersection
  hitinfo hinfo;
  bool hit = intersect::sphere(sl._sphere, r, hinfo);
  if (!hit) return 0.f;

  // compute probability
  ℝ3 const L = sl._sphere.T.pos()-r.p;
  float const dist2 = L|L;
  float const sr = (1.f-std::sqrtf(1.f-sl.size*sl.size/dist2));
  return .5f/(π<float>*sr);
}

/// @brief light sampling dispatch
constexpr void light(
  float l,
  Light const *light, 
  hitinfo const &hinfo, 
  scene const &sc,
  info<ℝ3,float> &info,
  RNG &rng)
{
  std::visit(Overload{
    [&](cg::spherelight const &sl){spherelight(l,sl,hinfo,sc,info,rng);},
    [&](cg::quaddirlight const &qdl){quaddirlight(l,qdl,hinfo,sc,info,rng);},
    [&](cg::pointlight const &pl){pointlight(l,pl,hinfo,sc,info);},
    [&](cg::dirlight const &dl){dirlight(l,dl,hinfo,sc,info);},
    [&](cg::ambientlight const &al){ambientlight(l,al,hinfo,info,rng);},
    [](auto const &){}
  }, *light);
}

/// @brief probability of a light generating a direction as a sample
constexpr float probForLight(
  Light const *l,
  ray const &r)
{
  return std::visit(Overload{
      [&](cg::spherelight const &sl){return probForSphereLight(sl,r);},
      [&](cg::quaddirlight const &qdl){return probForQuadDirLight(qdl,r);},
      [&](cg::pointlight const &){return probForPointLight();},
      [&](cg::dirlight const &){return probForDirLight();},
      [&](cg::ambientlight const &){return probForAmbientLight();},
      [](auto const &){return 0.f;}
    }, *l);
}
// ** end of light sampling ***********


// ************************************
/// @name material sampling

/// @brief cosine weighted sample of lambertian brdf
inline bool lambertiani(
  float l,
  cg::lambertian const &mat,
  hitinfo const &hinfo,
  info<ℝ3,float> &info,
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
  info.mult = mat.BRDFcosθ(l,i,hinfo.n());
  info.weight = mat.albedo(l)/p;
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
  float l,
  cg::blinn const &b, 
  hitinfo const &hinfo,
  ℝ3 const &o,
  info<ℝ3,float> &info, 
  RNG &rng, 
  float p)
{
  float const lobe = rng.flt();
  if (lobe>=p) { return false; }
  
  float const pp_d = p*b.p_d; float const pp_spec = p*b.p_spec;

  if (lobe<pp_spec)
  { // specular sample
    // sample half vector, split on Fresnel
    float const η = hinfo.front ? 1.f/b.ior(l) : b.ior(l);
    ℝ3 ω, h, i_R, i_T;
    ω = blinnh(b.α,rng.flt(),rng.flt());
    if (!hinfo.front) { ω[2]*=-1; } // flip half vector to outgoing hemisphere
    h = hinfo.F.toBasis(ω);
    i_R = bra::reflect(o,h);
    i_T = transmit(o,h,η); // returns {0,0,0} if TIR
    
    /// @todo, compare with transmitted Fresnel split
    float const cos_o = o|h;
    float F = b.fresnel(b.F0(η),cos_o);
    if (i_T[0]==0.f) { F=1.f; } // TIR, all reflection
    auto [p_r, p_t] = b.fresnelSplit(F);

    if (lobe<pp_spec*p_r)
    { // specular sample
      // p(sample)p(spec|sample)p(r|spec)p(i|r)
      info.prob = pp_spec*p_r*b.blinnhProb(ω[2])*.25f;
      info.val  = i_R;
      info.mult = b.frcosθ(l,ω[2],F);
      // frcosθ/p(i)
      info.weight = 
        (b.Ks(l)+b.Kt(l)*F)*(b.α+2)/(pp_spec*p_r*std::abs(ω[2])*(b.α+1));
      return true;
    } else
    { // transmissive sample
      info.prob = pp_spec*p_t*b.blinnhProb(ω[2])*.25f;
      info.val  = i_T;
      info.mult = b.ftcosθ(l,ω[2],F);
      info.weight = b.Kt(l)*(1-F)*(b.α+2)/(pp_spec*p_t*std::abs(ω[2]));
      return true;
    }
  } else if (lobe<pp_spec+pp_d)
  { // diffuse lobe sample
    ℝ3 dir = stoch::CosHemi(rng.flt(), rng.flt());
    if (!hinfo.front) { dir[2]*=-1; }
    ℝ3 const i = hinfo.F.toBasis(dir);
    info.prob = pp_d*b.fdiProb(dir[2]);
    info.val = i;
    info.mult = b.fdcosθ(l,dir[2]);
    info.weight = b.Kd(l)/pp_d;
    return true;
  }
  return false;
}
/// @brief probability of blinn generating the sample direction
inline float probForBlinn(
  float l,
  blinn const &b, 
  hitinfo const &hinfo, 
  ℝ3 const &o,
  ℝ3 const &i, 
  float p)
{
  ℝ3 const n = hinfo.n();
  float const ior = hinfo.front ? 1.f/b.ior(l) : b.ior(l);
  float const cos_in = i|n;
  ℝ3 h;
  if (cos_in>0.f) { h = (i+o).normalized(); }
  else { h = -(i+ior*o).normalized(); } // always points to lower ior mat
  float const cos_o = std::abs(o|h);
  float const F = b.fresnel(b.F0(ior), cos_o);
  auto [p_r, p_t] = b.fresnelSplit(F);
  if (cos_in>0.f)
  { // reflection/diffuse
    // p(i) = p*(p(d)p(i|d)+p(spec)*p(r|spec)*p(i|r))
    return p*(b.p_d*b.fdiProb(cos_in) + b.p_spec*p_r*b.blinnhProb(h|n)*.25f);
  } else
  { // transmition
    float const cos2_it = 1.f-ior*ior*(1.f-cos_o*cos_o);
    if (cos2_it < 0.f) { return 0.f; } // should be tir
    // p(i) = pp(spec)p(t|spec)p(i|t)
    return p*b.p_spec*p_t*b.blinnhProb(h|n)*.25f;
  }
}


/// @brief material incoming light direction sampling dispatch
constexpr bool materiali(
  float l,
  Material const *mat,
  hitinfo const &hinfo,
  ℝ3 const &o,
  info<ℝ3,float> &info,
  RNG &rng,
  float p)
{
  return std::visit(Overload{
      [&](cg::lambertian const &lm){return lambertiani(l,lm,hinfo,info,rng,p);},
      [&](cg::blinn const &b){return blinni(l,b,hinfo,o,info,rng,p);},
      [](auto const &){return false;}
    }, *mat);
}

/// @brief material sample evaluation dispatch 
constexpr float probForMateriali(
  float l,
  Material const *mat, 
  hitinfo const &hinfo, 
  ℝ3 const &i, 
  ℝ3 const &o, 
  float p)
{
  return std::visit(Overload{
      [&](cg::lambertian const &){return probForLambertian(hinfo,i,o,p);},
      [&](cg::blinn const &m){return probForBlinn(l,m,hinfo,i,o,p);},
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
inline void loadSpectrum(coefficientλ<Nλ> &s, json const &j)
{
  if (j.is_string()) {s=sampledλ(j.get<std::string>());}
  else if (j.is_number()) {s=j.get<float>();}
  else 
  {
    ℝ3 rgb_color;
    loadℝ3(rgb_color, j); 
    s=linRGB2coefλ(rgb_color);
  }
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
  try { load(dof, j.at("dof")); } catch(...) { dof=0.f; }
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
  {loadSpectrum(light.radiance, j.at("radiance"));}
inline void loadPointLight(pointlight &light, json const &j)
{
  loadSpectrum(light.radiant_intensity, j.at("radiant_intensity"));
  loadℝ3(light.pos, j.at("pos"));
}
inline void loadDirLight(dirlight &light, json const &j)
{
  loadSpectrum(light.radiant_intensity, j.at("radiant_intensity"));
  loadℝ3(light.dir, j.at("dir"));
}
inline void loadQuadDirLight(
  quaddirlight &light, json const &j, list<Material> &mats)
{
  plane p;
  loadSpectrum(light.radiance, j.at("radiance"));
  loadTransform(p.T, j.at("transform"));
  float spec;
  try { load(spec, j.at("specular")); } catch(...) { spec=1024.f; }
  std::string name = "emitter_"+light.radiance.toString();
  materialidx mat = mats.idxOf(name);
  if (mat==mats.size())
  {
    diremitter m = {name, light.radiance, spec};
    mats.emplace_back(std::in_place_type<diremitter>, std::move(m));
  }
  p.mat = mat;
  light.spec = spec;
  light.inv_area = 
    1.f/(bra::column<3>(p.T.M,0).l2()*bra::column<3>(p.T.M,1).l2());
  light._plane = p;
}
inline void loadSphereLight(
  spherelight &light, json const &j, list<Material> &mats)
{
  sphere s;
  loadSpectrum(light.radiance, j.at("radiance"));
  loadTransform(s.T, j.at("transform"));
  float const sx = bra::column<3>(s.T.M,0).l2();
  float const sy = bra::column<3>(s.T.M,1).l2();
  float const sz = bra::column<3>(s.T.M,2).l2();
  assert(std::abs(sx-sy)<0.01);
  assert(std::abs(sx-sz)<0.01); // check for uniform scaling
  std::string name = "emitter_"+light.radiance.toString();
  materialidx mat = mats.idxOf(name);
  if (mat==mats.size()) 
  {
    emitter m = {name, light.radiance}; 
    mats.emplace_back(std::in_place_type<emitter>, std::move(m));
  }
  s.mat = mat;
  light.size = sx;
  light._sphere = s;
}
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
      scene.lights.emplace_back(std::in_place_type<ambientlight>,std::move(l));
      break;
    }
    case LightType::POINT: 
    {
      pointlight l;
      l.name=name;
      loadPointLight(l, j_light);
      scene.lights.emplace_back(std::in_place_type<pointlight>,std::move(l));
      break;
    }
    case LightType::DIR: 
    {
      dirlight l;
      l.name=name;
      loadDirLight(l, j_light);
      scene.lights.emplace_back(std::in_place_type<dirlight>,std::move(l));
      break;
    }
    case LightType::SPHERE:
    {
      spherelight light;
      light.name=name;
      loadSphereLight(light, j_light, scene.materials);
      light._sphere.obj = scene.objects.size();
      scene.objects.emplace_back(
        std::in_place_type<sphere>,std::move(light._sphere));
      scene.lights.emplace_back(
        std::in_place_type<spherelight>,std::move(light));
      break;
    }
    case LightType::QUADDIR:
    {
      quaddirlight light;
      light.name=name;
      loadQuadDirLight(light, j_light, scene.materials);
      light._plane.obj = scene.objects.size();
      scene.objects.emplace_back(
        std::in_place_type<plane>,std::move(light._plane));
      scene.lights.emplace_back(
        std::in_place_type<quaddirlight>,std::move(light));
      break;
    }
    } // end switch
  }
}
// ************************************

// ************************************
/// @name object loading

/// @todo group loading
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

inline void loadTriMeshFromFile(trimeshdata &m, std::string const& fpath)
{
  fastObjMesh* obj = fast_obj_read(fpath.c_str());

  // prep bounds for setting
  m.bounds.inf = UB<float>;
  m.bounds.sup = -UB<float>;

  // vertices
  m.V.reserve(obj->position_count);
  for (unsigned i=0; i<obj->position_count; i++) 
  {
    m.V.emplace_back(
      obj->positions[3*i],obj->positions[3*i+1],obj->positions[3*i+2]);
    m.bounds.inf[0] = min(m.bounds.inf[0],obj->positions[3*i]);
    m.bounds.inf[1] = min(m.bounds.inf[1],obj->positions[3*i+1]);
    m.bounds.inf[2] = min(m.bounds.inf[2],obj->positions[3*i+2]);
    m.bounds.sup[0] = max(m.bounds.sup[0],obj->positions[3*i]);
    m.bounds.sup[1] = max(m.bounds.sup[1],obj->positions[3*i+1]);
    m.bounds.sup[2] = max(m.bounds.sup[2],obj->positions[3*i+2]);
  }

  // tex coords
  m.u.reserve(obj->texcoord_count);
  for (unsigned i=0; i<obj->texcoord_count; i++)
    { m.u.emplace_back(obj->texcoords[2*i],obj->texcoords[2*i+1]); }

  // normals and tangents
  m.N.reserve(obj->normal_count);
  for (unsigned i=0; i<obj->normal_count; i++) 
  {
    m.N.emplace_back(
      obj->normals[3*i],obj->normals[3*i+1],obj->normals[3*i+2]);
  }
  
  // triangles, edges, tangents, and geometric normal
  m.T.resize(obj->normal_count);
  m.F.reserve(obj->face_count);
  m.GN.reserve(obj->face_count);
  m.GT.reserve(obj->face_count);
  size_t k = 0;
  for (unsigned f=0; f<obj->face_count; f++) 
  { // iterate through faces
    triangle tri;
    for (unsigned i=0; i<3; i++) 
    { // vertices
      fastObjIndex idx = obj->indices[k+i];
      tri.V[i] = idx.p;
      tri.N[i] = idx.n;
      tri.u[i] = idx.t;
    }

    // compute geometric normal
    ℝ3 const e0 = m.V[tri.V[1]]-m.V[tri.V[0]];
    ℝ3 const e2 = m.V[tri.V[2]]-m.V[tri.V[0]];
    m.GN.emplace_back((e0^e2).normalized());
    m.F.emplace_back(tri);

    // tangents
    float const du1 = m.u[tri.u[1]][0]-m.u[tri.u[0]][0];
    float const du2 = m.u[tri.u[2]][0]-m.u[tri.u[0]][0];
    float const dv1 = m.u[tri.u[1]][1]-m.u[tri.u[0]][1];
    float const dv2 = m.u[tri.u[2]][1]-m.u[tri.u[0]][1];
    float const r = 1.f/(du1*dv2-du2*dv1);
    ℝ3 const B = r*(e2*du1-e0*du2);
    m.GT.emplace_back((B^m.GN[f]).normalized());
    for (unsigned i=0; i<3; i++)
    { // compute tangent per vertex
      m.T[tri.N[i]] = B^m.N[tri.N[i]].normalized();
    }
    k += 3;
  }
  fast_obj_destroy(obj);
}

inline void loadTriMesh(
  trimesh &m, list<Mesh> &meshes, json const &j, list<Material> const &mats)
{
  transform T;
  loadTransform(T, j.at("transform"));
  m.T = T;
  m.mat = mats.idxOf(j.at("material").get<std::string>());
  auto fpath = j.at("source").get<std::string>();
  m.mesh = meshes.idxOf(fpath);
  if (m.mesh==meshes.size())
  { // source is not loaded yet
    trimeshdata mesh_data;
    mesh_data.name = fpath;
    loadTriMeshFromFile(mesh_data,fpath);
    mesh_data.buildBVH();
    meshes.emplace_back(std::in_place_type<trimeshdata>, std::move(mesh_data));
  }
}

/// @brief loads all scene objects
inline void loadObjects(
  list<Object> &objs,
  list<Mesh> &meshes,
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
      {loadObjects(objs, meshes, j_obj.at("children"), mats); break;}
    case ObjectType::SPHERE: 
    {
      sphere s; 
      s.name=name; 
      loadSphere(s, j_obj, mats);
      s.obj = objs.size();
      objs.emplace_back(std::in_place_type<sphere>, std::move(s)); 
      break;
    }
    case ObjectType::PLANE:
    {
      plane p; 
      p.name=name; 
      loadPlane(p, j_obj, mats);
      p.obj = objs.size();
      objs.emplace_back(std::in_place_type<plane>, std::move(p)); 
      break;
    }
    case ObjectType::TRIMESH:
    {
      trimesh m;
      m.name=name;
      loadTriMesh(m, meshes, j_obj, mats);
      m.obj = objs.size();
      objs.emplace_back(std::in_place_type<trimesh>, std::move(m));
      break;
    }
    }
  }
}
// ************************************

// ************************************
/// @name material loading

inline void loadLambertian(lambertian &m, json const &j)
{
  loadSpectrum(m.albedo, j.at("albedo"));
}
inline void loadBlinn(blinn &m, json const &j)
{
  float alpha;
  try {loadSpectrum(m.Kd, j.at("Kd"));} catch(...) {m.Kd = 0.f;}
  try {loadSpectrum(m.Ks, j.at("Ks"));} catch(...) {m.Ks = 0.f;}
  try {loadSpectrum(m.Kt, j.at("Kt"));} catch(...) {m.Kt = 0.f;}
  try {loadSpectrum(m.Le, j.at("Le"));} catch(...) {m.Le = 0.f;}
  try {loadSpectrum(m.ior, j.at("ior"));} catch(...) {m.ior = 1.54f;}
  try {load(alpha, j.at("glossiness"));} catch(...) {alpha = 1024;}
  m.α = alpha;
  m.init();
}
/// @todo microfacet material loading
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
  loadObjects(scene.objects, scene.meshes, j.at("objects"), scene.materials);
  loadLights(scene, j.at("lights"));
  return true;
}



} // ** end of namespace load *************************************************

} // ** end of namespace cg ***********
} // ** end of namespace nl ***********

// ****************************************************************************
#endif // #ifndef CG_HPP