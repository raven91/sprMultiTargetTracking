//
// Created by Nikita Kruk on 03.07.18.
//

#ifndef SPRMULTITARGETTRACKING_PARAMETERHANDLERSYNTHETIC_HPP
#define SPRMULTITARGETTRACKING_PARAMETERHANDLERSYNTHETIC_HPP

#include "../Definitions.hpp"

#include <unordered_map>

class ParameterHandlerSynthetic
{
 public:

  ParameterHandlerSynthetic();
  ~ParameterHandlerSynthetic();

  const std::string &GetTrackingFolder();
  const std::string &GetSimulationFolder();
  const std::string &GetFilteringStatisticsFolder();
//  const std::string &GetInputFile0();
//  const std::string &GetInputFile1();
//  const std::string &GetInputFile2();
//  const std::string &GetOutputFolder();
  const std::string &GetKalmanFilterOutputFileName();
  const std::string &GetKalmanFilterMatlabOutputFileName();
  int GetFirstImage();
  int GetLastImage();
  int GetSubimageXPos();
  int GetSubimageYPos();
  int GetSubimageXSize();
  int GetSubimageYSize();
  Real GetDataAssociationCost();
  int GetNumberOfOriginalTargets();
  Real GetSimulationParameter(const std::string &name);
  Real GetPercentageOfMisdetections();

  void SetSubimageSize(int subimage_x_size, int subimage_y_size);
  void SetNewParametersFromSimulation(Real phi, Real a, Real U0, Real kappa);
  void SetPercentageOfMisdetections(Real percentage_of_misdetections);

 private:

  std::string tracking_folder_;
  std::string simulation_folder_;
  std::string filtering_statistics_folder_;
//  std::string input_file_0_;
//  std::string input_file_1_;
//  std::string input_file_2_;
//  std::string output_folder_;
  std::string kalman_filter_output_file_name_;
  std::string kalman_filter_matlab_output_file_name_;

  std::unordered_map<std::string, Real> parameters_dictionary_;
  int first_image_;
  int last_image_;
  int subimage_x_pos_;
  int subimage_y_pos_;
  int subimage_x_size_;
  int subimage_y_size_;
  Real data_association_cost_;
  int number_of_original_targets_;
  Real percentage_of_misdetections_;
};

#endif //SPRMULTITARGETTRACKING_PARAMETERHANDLERSYNTHETIC_HPP
