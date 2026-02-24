# λ (Lambda)
A C++ spectral renderer using my library numli

![](coverpage-128000spp-32b-0.80p.png)

128000 sample per pixel rendering highlighting some of the renderer's capabilities. It took 4hr 15min 17.4s on an AMD EPYC 9654P 96-Core Processor with 192 threads.

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