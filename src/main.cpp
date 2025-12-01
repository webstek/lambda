// ****************************************************************************
/// @file main.cpp
/// @author Kyle Webster
/// @version 0.1
/// @date 30 Nov 2025
/// @brief program entry point
// ****************************************************************************
#include "lambda.hpp"
// ************************************

int main(int argc, const char *argv)
{
  Lambda λ;
  
  λ.renderer.loadScene("scenes/sphere-spherelight.nls");
  λ.renderer.render();
  λ.renderer.saveImage("render/NEE.png");
  
  return 0;
}

// ****************************************************************************