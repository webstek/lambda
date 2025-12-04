# λ (Lambda)
A C++ Spectral Path Tracer (hopefully) using my library numli

## Features
* Path tracing supporting Blinn and Lambertian material models. Renders spheres
and planes.
* Scene definition in `.nls` files

## Building
From the project root invoke `make [debug, release]` for debug or release 

## Usage
Navigate to `bin/`, call `./lambda` to perform a rendering. To change the SPP,
maximum scatterings per path, and scatter sampling chance, edit `renderer.cpp`
directly and rebuild. To change the scene, edit `main.cpp` directly and 
rebuild.

## Future Work
* HPC script
* Anisotropic material
* Spectral rendering
* Triangle meshes + BVH
* Textures
* loading `gltf` scenes linked in `.nls` files
* Verifying Depth of Field

## Dependencies
This project uses `C++23` and `OpenMP` and has been tested to compile with 
`g++15`.