//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_DEFINITIONS_HPP
#define SPRMULTITARGETTRACKING_DEFINITIONS_HPP

#define _USE_MATH_DEFINES

#include <cstdlib>  // size_t
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

typedef float Real;
typedef long CostInt;

const int kNumOfStateVars = 4;          // x,y,v_x,v_y
const int kNumOfDetectionVars = 2;      // x,y
const int kNumOfExtractedFeatures = 8;  // x,y,v_x,v_y,area,slope,width,height

inline Real WrappingModulo(Real numerator, Real denominator)
{
  return numerator - denominator * std::floorf(numerator / denominator);
}

#endif //SPRMULTITARGETTRACKING_DEFINITIONS_HPP
