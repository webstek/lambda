# λ (Lambda)
A C++ spectral renderer using my library numli

![](./bin/render/coverpage-32768spp-64b-0.90p.png)

Sample rendering of two teapots on a glossy floor at 32768 samples per pixel illuminated by a D65 illuminant. Left: 550nm thick Acetate thin-film coating. Right: Solid teapot with Indium-Tin Oxide index of refraction.

## Highlight Features
* Hero-wavelength Spectral MIS path tracing
* Wide BVH with 8-way AVX2 box intersection
* Instanced triangle meshes and materials
* Blinn, Lambertian, and thin-film materials
* Scene definition in `.nls` files

## Building
From the project root invoke `make [debug, release]` for debug or release 

## Usage
Navigate to `bin/`, call `./lambda <fpath> -s <SPP> -b <MAX_BOUNCES> -p <BOUNCE_PROB>` for rendering.


## Future Work
* Tone mapping
* Estimated render time output
* Distributed rendering
* Bidirectional path tracing
* Anisotropic Microfacet material
* Absorption
* Volumetric scattering
* Lens-based Camera
* Textures
* loading `gltf` scenes linked in `.nls` files

## Dependencies
This project uses `C++23`, `OpenMP`, and `AVX2` instructions, and has been tested to compile with 
`g++15`.