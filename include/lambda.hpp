// ****************************************************************************
/// @file lambda.hpp
/// @author Kyle Webster
/// @version 0.2
/// @date 22 Feb 2026
/// @brief λ (Lambda) program definition
// ****************************************************************************
#pragma once
// ** Includes ************************
#include "cg.hpp"
#include "renderer.hpp"
// ************************************

class Lambda 
{
public:
  nl::cg::scene scene;
  Renderer renderer;

  bool loadScene(const char *f_name);
};

// ****************************************************************************