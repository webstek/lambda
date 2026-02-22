// ****************************************************************************
/// @file main.cpp
/// @author Kyle Webster
/// @version 0.2
/// @date Dec 12 2025
/// @brief program entry point
// ****************************************************************************
#include <print>
#ifdef DEBUG
#include <fenv.h>
#endif
#include "lambda.hpp"
// ************************************

int main(int argc, char **argv)
{
  // enable FP traps if in debug
  #ifdef DEBUG
  feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
  #endif
  
  Lambda λ;
  // check command line arguments
  int argn;
  while ((argn = getopt(argc, argv, "b:s:p:")) != -1)
  {
    switch (argn)
    {
    case 'b': λ.renderer.MAX_SCATTERINGS = std::stoi(optarg); break;
    case 's': λ.renderer.SPP = std::stoi(optarg); break;
    case 'p': λ.renderer.SAMPLE_P = std::stof(optarg); break;
    default: std::println("Usage: ./lambda <fileName> [args]"); return 1;
    }
  }
  const char* fileName = argv[optind];
  λ.renderer.loadScene(fileName);
  λ.renderer.render();
  λ.renderer.saveImage();
  
  return 0;
}

// ****************************************************************************