//
// Created by Nikita Kruk on 03.07.18.
//

#include "PeriodicBoundaryConditionsConfiguration.hpp"

#include <cmath>

PeriodicBoundaryConditionsConfiguration::PeriodicBoundaryConditionsConfiguration(Real x_size, Real y_size) :
    x_size_(x_size),
    y_size_(y_size),
    x_rsize_(1.0 / x_size),
    y_rsize_(1.0 / y_size)
{

}

PeriodicBoundaryConditionsConfiguration::~PeriodicBoundaryConditionsConfiguration()
{

}

void PeriodicBoundaryConditionsConfiguration::ClassAEffectiveParticleDistance(Real x_i,
                                                                              Real y_i,
                                                                              Real x_j,
                                                                              Real y_j,
                                                                              Real &dx,
                                                                              Real &dy)
{
  dx = x_j - x_i;
  dx -= static_cast<int>(dx * 2.0 * x_rsize_) * x_size_;

  dy = y_j - y_i;
  dy -= static_cast<int>(dy * 2.0 * y_rsize_) * y_size_;
}

//The minimum image convention for the calculation of effective particle distances
//if the sign of the distance is relevant
void PeriodicBoundaryConditionsConfiguration::ClassCEffectiveParticleDistance_signed(Real x_i,
                                                                                     Real y_i,
                                                                                     Real x_j,
                                                                                     Real y_j,
                                                                                     Real &dx,
                                                                                     Real &dy)
{
  dx = x_j - x_i;
  dx -= x_size_ * std::nearbyint(dx * x_rsize_);

  dy = y_j - y_i;
  dy -= y_size_ * std::nearbyint(dy * y_rsize_);
}

//The minimum image convention for the calculation of effective particle distances
//if the sign of the distances is not relevant
void PeriodicBoundaryConditionsConfiguration::ClassCEffectiveParticleDistance_unsigned(Real x_i,
                                                                                       Real y_i,
                                                                                       Real x_j,
                                                                                       Real y_j,
                                                                                       Real &dx,
                                                                                       Real &dy)
{
  dx = std::fabs(x_j - x_i);
  dx -= static_cast<int>(dx * x_rsize_ + 0.5) * x_size_;

  dy = std::fabs(y_j - y_i);
  dy -= static_cast<int>(dy * y_rsize_ + 0.5) * y_size_;
}

void PeriodicBoundaryConditionsConfiguration::ApplyPeriodicBoundaryConditions(Real x, Real y, Real &x_pbc, Real &y_pbc)
{
  x_pbc = x - std::floor(x * x_rsize_) * x_size_;
  y_pbc = y - std::floor(y * y_rsize_) * y_size_;
}

//Restrict particle coordinates to the simulation box
void PeriodicBoundaryConditionsConfiguration::ApplyPeriodicBoundaryConditions(std::map<int, Eigen::VectorXf> &targets)
{
#pragma unroll
  for (std::map<int, Eigen::VectorXf>::iterator it = targets.begin(); it != targets.end(); ++it)
  {
    (it->second)[0] -= std::floor((it->second[0]) * x_rsize_) * x_size_;
    (it->second)[1] -= std::floor((it->second[1]) * y_rsize_) * y_size_;
  }
}