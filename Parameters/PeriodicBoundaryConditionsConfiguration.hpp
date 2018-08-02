//
// Created by Nikita Kruk on 03.07.18.
//

#ifndef SPRMULTITARGETTRACKING_PERIODICBOUNDARYCONDITIONSCONFIGURATION_HPP
#define SPRMULTITARGETTRACKING_PERIODICBOUNDARYCONDITIONSCONFIGURATION_HPP

#include "../Definitions.hpp"

#include <vector>
#include <map>

#include <eigen3/Eigen/Dense>

/**
 * Based on:
 * https://www.degruyter.com/downloadpdf/j/zpch.2013.227.issue-2-3/zpch.2013.0311/zpch.2013.0311.pdf
 */
class PeriodicBoundaryConditionsConfiguration
{
 public:

  PeriodicBoundaryConditionsConfiguration(Real x_size, Real y_size);
  ~PeriodicBoundaryConditionsConfiguration();

//  void SetNewBoundaries(Real x_size, Real y_size);

  void ClassAEffectiveParticleDistance(Real x_i, Real y_i, Real x_j, Real y_j, Real &dx, Real &dy);

  void ClassBEffectiveParticleDistance_signed(Real x_i, Real y_i, Real x_j, Real y_j, Real &dx, Real &dy);
  void ClassBEffectiveParticleDistance_unsigned(Real x_i, Real y_i, Real x_j, Real y_j, Real &dx, Real &dy);

  void ClassCEffectiveParticleDistance_signed(Real x_i, Real y_i, Real x_j, Real y_j, Real &dx, Real &dy);
  void ClassCEffectiveParticleDistance_unsigned(Real x_i, Real y_i, Real x_j, Real y_j, Real &dx, Real &dy);

  void ApplyPeriodicBoundaryConditions(Real x, Real y, Real &x_pbc, Real &y_pbc);
  void ApplyPeriodicBoundaryConditions(std::map<int, Eigen::VectorXf> &targets);

 private:

  Real x_size_;
  Real y_size_;
  Real x_rsize_;
  Real y_rsize_;

};

#endif //SPRMULTITARGETTRACKING_PERIODICBOUNDARYCONDITIONSCONFIGURATION_HPP
