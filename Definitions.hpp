//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_DEFINITIONS_HPP
#define SPRMULTITARGETTRACKING_DEFINITIONS_HPP

#include <cstdlib>  // size_t
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#define PARTIAL_IMAGE_OUTPUT

typedef double Real;
typedef long CostInt;

const int kNumOfStateVars = 4;          // x,y,v_x,v_y
const int kNumOfDetectionVars = 2;      // x,y
const int kNumOfExtractedFeatures = 8;  // x,y,v_x,v_y,area,slope,width,height

/*
 * Transform numerator into [0, denominator)
 */
inline Real WrappingModulo(Real numerator, Real denominator)
{
  return numerator - denominator * std::floor(numerator / denominator);
}

/*
 * Transform x into [0, 2\pi)
 */
inline Real ConstrainAnglePositive(Real x)
{
  x = std::fmod(x, 2.0 * M_PI);
  if (x < 0.0)
  {
    x += 2.0 * M_PI;
  }
  return x;
}

/*
 * Transform x into [-\pi, \pi)
 */
inline Real ConstrainAngleCentered(Real x)
{
  x = std::fmod(x + M_PI, 2.0 * M_PI);
  if (x < 0.0)
  {
    x += 2.0 * M_PI;
  }
  return x - M_PI;
}

#endif //SPRMULTITARGETTRACKING_DEFINITIONS_HPP
