// ****************************************************************************
/// @file lambda.cpp
/// @author Kyle Webster
/// @version 1.0
/// @date 11 Apr 2026
/// @brief λ (Lambda) program implementation
// ****************************************************************************
#include "lambda.hpp"
// ********************************************************

bool Lambda::loadScene(const char *f_path)
  { return nl::cg::load::loadNLS(scene, f_path); }

// ****************************************************************************