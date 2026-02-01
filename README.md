# λ (Lambda)
A C++ Spectral Path Tracer using my library numli

## Features
* Spectral path tracing supporting Blinn and Lambertian material models.
* Scene definition in `.nls` files

## Building
From the project root invoke `make [debug, release]` for debug or release 

## Usage
Navigate to `bin/`, call `./lambda <fpath> -s <SPP> -b <MAX_BOUNCES> -p <BOUNCE_PROB>` for rendering.


## Future Work
* Anisotropic material
* Absorption
* Textures
* loading `gltf` scenes linked in `.nls` files
* Verifying Depth of Field

## Dependencies
This project uses `C++23` and `OpenMP` and has been tested to compile with 
`g++15`.