//
// Created by Nikita Kruk on 03.07.18.
//

#include "ParameterHandlerSynthetic.hpp"

#include <fstream>
#include <sstream>

ParameterHandlerSynthetic::ParameterHandlerSynthetic()
{
  std::ifstream parameters_file
      ("/Users/nikita/CLionProjects/sprMultiTargetTracking/Parameters/ConfigSynthetic.cfg", std::ios::in);
  assert(parameters_file.is_open());

  // read string values
  parameters_file >> tracking_folder_ >> tracking_folder_;
  parameters_file >> simulation_folder_ >> simulation_folder_;
  parameters_file >> filtering_statistics_folder_ >> filtering_statistics_folder_;
//  parameters_file >> input_file_0_ >> input_file_0_;
//  parameters_file >> input_file_1_ >> input_file_1_;
//  parameters_file >> input_file_2_ >> input_file_2_;
//  parameters_file >> output_folder_ >> output_folder_;
  parameters_file >> kalman_filter_output_file_name_ >> kalman_filter_output_file_name_;
  parameters_file >> kalman_filter_matlab_output_file_name_ >> kalman_filter_matlab_output_file_name_;

  // read parameter values
  std::string key;
  Real value;
  while (parameters_file >> key >> value)
  {
    parameters_dictionary_[key] = value;
  }
  parameters_file.close();
  // initialize real-valued variables
  first_image_ = (int) parameters_dictionary_["first_image"];
  last_image_ = (int) parameters_dictionary_["last_image"];
  subimage_x_pos_ = (int) parameters_dictionary_["subimage_x_pos"];
  subimage_y_pos_ = (int) parameters_dictionary_["subimage_y_pos"];
  subimage_x_size_ = (int) parameters_dictionary_["subimage_x_size"];
  subimage_y_size_ = (int) parameters_dictionary_["subimage_y_size"];
  data_association_cost_ = parameters_dictionary_["data_association_cost"];
  number_of_original_targets_ = (int) parameters_dictionary_["number_of_original_targets"];
  percentage_of_misdetections_ = parameters_dictionary_["percentage_of_misdetections"];
}

ParameterHandlerSynthetic::~ParameterHandlerSynthetic() = default;

const std::string &ParameterHandlerSynthetic::GetTrackingFolder()
{
  return tracking_folder_;
}

const std::string &ParameterHandlerSynthetic::GetSimulationFolder()
{
  return simulation_folder_;
}

const std::string &ParameterHandlerSynthetic::GetFilteringStatisticsFolder()
{
  return filtering_statistics_folder_;
}

//const std::string& ParameterHandlerSynthetic::GetInputFile0()
//{
//  return input_file_0_;
//}
//
//const std::string& ParameterHandlerSynthetic::GetInputFile1()
//{
//  return input_file_1_;
//}
//
//const std::string& ParameterHandlerSynthetic::GetInputFile2()
//{
//  return input_file_2_;
//}

//const std::string& ParameterHandlerSynthetic::GetOutputFolder()
//{
//  return output_folder_;
//}

const std::string &ParameterHandlerSynthetic::GetKalmanFilterOutputFileName()
{
  return kalman_filter_output_file_name_;
}

const std::string &ParameterHandlerSynthetic::GetKalmanFilterMatlabOutputFileName()
{
  return kalman_filter_matlab_output_file_name_;
}

int ParameterHandlerSynthetic::GetFirstImage()
{
  return first_image_;
}

int ParameterHandlerSynthetic::GetLastImage()
{
  return last_image_;
}

int ParameterHandlerSynthetic::GetSubimageXPos()
{
  return subimage_x_pos_;
}

int ParameterHandlerSynthetic::GetSubimageYPos()
{
  return subimage_y_pos_;
}

int ParameterHandlerSynthetic::GetSubimageXSize()
{
  return subimage_x_size_;
}

int ParameterHandlerSynthetic::GetSubimageYSize()
{
  return subimage_y_size_;
}

Real ParameterHandlerSynthetic::GetDataAssociationCost()
{
  return data_association_cost_;
}

int ParameterHandlerSynthetic::GetNumberOfOriginalTargets()
{
  return number_of_original_targets_;
}

Real ParameterHandlerSynthetic::GetPercentageOfMisdetections()
{
  return percentage_of_misdetections_;
}

void ParameterHandlerSynthetic::SetSubimageSize(int subimage_x_size, int subimage_y_size)
{
  subimage_x_size_ = subimage_x_size;
  subimage_y_size_ = subimage_y_size;
}

void ParameterHandlerSynthetic::SetNewParametersFromSimulation(Real phi, Real a, Real U0, Real kappa)
{
  std::ostringstream parameters_file_name_buffer;
  parameters_file_name_buffer << simulation_folder_ << number_of_original_targets_ << "/parameters_phi_" << phi << "_a_"
                              << a << "_U0_" << U0 << "_k_" << kappa << ".txt";
  std::ifstream parameters_file(parameters_file_name_buffer.str(), std::ios::in);
  assert(parameters_file.is_open());
  std::string key;
  Real value;
  while (parameters_file >> key >> value)
  {
    parameters_dictionary_[key] = value;
  }
  parameters_file.close();
  SetSubimageSize((int) parameters_dictionary_["L"], (int) parameters_dictionary_["L"]);
}

Real ParameterHandlerSynthetic::GetSimulationParameter(const std::string &name)
{
  return parameters_dictionary_[name];
}

void ParameterHandlerSynthetic::SetPercentageOfMisdetections(Real percentage_of_misdetections)
{
  percentage_of_misdetections_ = percentage_of_misdetections;
}