// ****************************************************************************
/// @file main.cpp
/// @author Kyle Webster
/// @version 0.3
/// @date Feb 22 2026
/// @brief program entry point
// ****************************************************************************
#include <print>
#include <fenv.h>
#include "lambda.hpp"
// ************************************

int main(int argc, char **argv)
{
  feenableexcept(FE_INVALID | FE_DIVBYZERO );
  
  Lambda λ;
  // check command line arguments
  float Y=0.12f;
  int argn;
  while ((argn = getopt(argc, argv, "b:s:p:Y:")) != -1)
  {
    switch (argn)
    {
    case 'b': λ.renderer.MAX_SCATTERINGS = std::stoi(optarg); break;
    case 's': λ.renderer.SPP = std::stoi(optarg); break;
    case 'p': λ.renderer.SAMPLE_P = std::stof(optarg); break;
    case 'Y': Y = std::stof(optarg); break;
    default: std::println("Usage: ./lambda <fileName> [args]"); return 1;
    }
  }
  
  // load scene from file
  const char* fileName = argv[optind];
  if (!λ.loadScene(fileName)) 
    { std::println("Failed to load scene."); return 1; }

  // allocate image buffers
  rendering raw_buffer;
  rendering img_buffer;
  λ.renderer.render(λ.scene, raw_buffer);
  λ.renderer.toneMap(raw_buffer, img_buffer, Y);
  λ.renderer.saveImage(img_buffer,fileName);
  
  return 0;
}

// ****************************************************************************