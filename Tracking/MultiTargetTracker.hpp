//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_MULTITARGETTRACKER_HPP
#define SPRMULTITARGETTRACKING_MULTITARGETTRACKER_HPP

#include "../Definitions.hpp"

#include <vector>
#include <map>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <eigen3/Eigen/Dense>

class MultitargetTracker
{
 public:

  MultitargetTracker();
  ~MultitargetTracker();

  void PerformTrackingForOneExperiment(const std::string &configuration_file_name);
  void PerformOnlyImageProcessingForOneExperiment(const std::string &file_name);
  void PerformOnlyFilteringForOneExperiment(const std::string &file_name);
  void PerformOnlyTrackLinkingForOneExperiment(const std::string &configuration_file_name);
  void PerformActionForMultipleExperiments(int action, const std::string &experiments_directory);

  void StartOnSyntheticData(Real phi, Real a, Real U0, Real kappa, Real percentage_of_misdetections);
  void StartOnSyntheticDataForDifferentParameters();

 private:

  std::map<int, Eigen::VectorXd> targets_;    // i -> x_i y_i v_x_i v_y_i area_i slope_i width_i height_i
  std::vector<Eigen::VectorXd> detections_;   // observations
  std::map<int, std::vector<Eigen::VectorXd>> trajectories_;   // vector of vectors of i -> x_i y_i v_x_i v_y_i area_i slope_i width_i height_i
  std::map<int, std::vector<int>> timestamps_;   // vector of timestamps for bacteria

};

#endif //SPRMULTITARGETTRACKING_MULTITARGETTRACKER_HPP
