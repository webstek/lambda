# λ (Lambda)
A C++ spectral renderer using my library numli

![](coverpage-32768spp-32b-0.80p.png)

## Highlight Features
* Hero-wavelength Spectral MIS path tracing
* Wide BVH with 8-way AVX2 box intersection
* Instanced triangle meshes and materials
* Blinn, Lambertian, and thin-film materials with measured spectra
* Filmic tone-mapping
* Scene definition in `.nls` files

## Building
From the project root invoke `make [debug, release]` for debug or release 

## Usage
Navigate to `bin/`, call `./lambda <fpath> [-s <SPP>, -b <MAX_BOUNCES>, -p <BOUNCE_PROB>, -Y <MID_GREY>]` for rendering.


## Future Work
* Estimated render time output
* Distributed rendering
* Bidirectional path tracing
* Scene level BVH
* Anisotropic Microfacet material
* Absorption
* Volumetric scattering
* Lens-based Camera
* Textures
* loading `gltf` scenes linked in `.nls` files

## Dependencies
This project uses `C++23`, `OpenMP`, and `AVX2` instructions, and has been tested to compile with 
`g++15`.