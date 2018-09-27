//
// Created by Nikita Kruk on 27.11.17.
//

#include "ParameterHandlerExperimental.hpp"

#include <fstream>
#include <map>

ParameterHandlerExperimental::ParameterHandlerExperimental(const std::string &file_name)
{
//  std::ifstream parameters_file("/Users/nikita/CLionProjects/sprMultiTargetTracking/Parameters/ConfigExperimental.cfg", std::ios::in);
  std::ifstream parameters_file(file_name, std::ios::in);
  assert(parameters_file.is_open());

  // read string values
  parameters_file >> input_folder_ >> input_folder_;
  parameters_file >> file_name_0_ >> file_name_0_;
  parameters_file >> file_name_1_ >> file_name_1_;
  parameters_file >> original_images_subfolder_ >> original_images_subfolder_;
  parameters_file >> image_processing_subfolder_ >> image_processing_subfolder_;
  parameters_file >> image_processing_output_file_name_ >> image_processing_output_file_name_;
  parameters_file >> kalman_filter_subfolder_ >> kalman_filter_subfolder_;
  parameters_file >> kalman_filter_output_file_name_ >> kalman_filter_output_file_name_;
  parameters_file >> kalman_filter_matlab_output_file_name_ >> kalman_filter_matlab_output_file_name_;
  parameters_file >> track_linking_output_file_name_ >> track_linking_output_file_name_;
  parameters_file >> track_linking_matlab_output_file_name_ >> track_linking_matlab_output_file_name_;
  parameters_file >> data_analysis_subfolder_ >> data_analysis_subfolder_;

  // read real values
  std::map<std::string, Real> parameters_dictionary;
  std::string key;
  Real value;
  while (parameters_file >> key >> value)
  {
    parameters_dictionary[key] = value;
  }
  parameters_file.close();

  // initialize real-valued variables
  first_image_ = (int) parameters_dictionary["first_image"];
  last_image_ = (int) parameters_dictionary["last_image"];
  subimage_x_pos_ = (int) parameters_dictionary["subimage_x_pos"];
  subimage_y_pos_ = (int) parameters_dictionary["subimage_y_pos"];
  subimage_x_size_ = (int) parameters_dictionary["subimage_x_size"];
  subimage_y_size_ = (int) parameters_dictionary["subimage_y_size"];
  image_subdivisions_ = (int) parameters_dictionary["image_subdivisions"];
  blur_type_ = (int) parameters_dictionary["blur_type"];
  blur_radius_ = (int) parameters_dictionary["blur_radius"];
  blur_sigma_ = (int) parameters_dictionary["blur_sigma"];
  threshold_type_ = (int) parameters_dictionary["threshold_type"];
  threshold_value_ = (int) parameters_dictionary["threshold_value"];
  morphological_operator_ = (int) parameters_dictionary["morphological_operator"];
  morphological_element_ = (int) parameters_dictionary["morphological_element"];
  morphological_size_ = (int) parameters_dictionary["morphological_size"];
  min_contour_area_ = (int) parameters_dictionary["min_contour_area"];
  area_increase_ = parameters_dictionary["area_increase"];
  convexity_defect_magnitude_ = parameters_dictionary["convexity_defect_magnitude"];
  data_association_cost_ = parameters_dictionary["data_association_cost"];
  secondary_data_association_cost_ = parameters_dictionary["secondary_data_association_cost"];
  height_to_width_ratio_ = parameters_dictionary["height_to_width_ratio"];
  center_of_mass_distance_ = parameters_dictionary["center_of_mass_distance"];
  roi_margin_ = (int) parameters_dictionary["roi_margin"];
  nl_means_denoising_h_ = parameters_dictionary["nl_means_denoising_h"];
  nl_means_denoising_template_window_size_ = (int) parameters_dictionary["nl_means_denoising_template_window_size"];
  nl_means_denoising_search_window_size_ = (int) parameters_dictionary["nl_means_denoising_search_window_size"];
  contrast_ = parameters_dictionary["contrast"];
  brightness_ = (int) parameters_dictionary["brightness"];
}

ParameterHandlerExperimental::~ParameterHandlerExperimental()
{

}

const std::string &ParameterHandlerExperimental::GetInputFolder()
{
  return input_folder_;
}

const std::string &ParameterHandlerExperimental::GetFileName0()
{
  return file_name_0_;
}

const std::string &ParameterHandlerExperimental::GetFileName1()
{
  return file_name_1_;
}

const std::string &ParameterHandlerExperimental::GetOriginalImagesSubfolder()
{
  return original_images_subfolder_;
}

const std::string &ParameterHandlerExperimental::GetImageProcessingSubfolder()
{
  return image_processing_subfolder_;
}

const std::string &ParameterHandlerExperimental::GetImageProcessingOutputFileName()
{
  return image_processing_output_file_name_;
}

const std::string &ParameterHandlerExperimental::GetKalmanFilterSubfolder()
{
  return kalman_filter_subfolder_;
}

const std::string &ParameterHandlerExperimental::GetKalmanFilterOutputFileName()
{
  return kalman_filter_output_file_name_;
}

const std::string &ParameterHandlerExperimental::GetKalmanFilterMatlabOutputFileName()
{
  return kalman_filter_matlab_output_file_name_;
}

const std::string &ParameterHandlerExperimental::GetTrackLinkingOutputFileName()
{
  return track_linking_output_file_name_;
}

const std::string &ParameterHandlerExperimental::GetTrackLinkingMatlabOutputFileName()
{
  return track_linking_matlab_output_file_name_;
}

const std::string &ParameterHandlerExperimental::GetDataAnalysisSubfolder()
{
  return data_analysis_subfolder_;
}

int ParameterHandlerExperimental::GetFirstImage()
{
  return first_image_;
}

int ParameterHandlerExperimental::GetLastImage()
{
  return last_image_;
}

int ParameterHandlerExperimental::GetSubimageXPos()
{
  return subimage_x_pos_;
}

int ParameterHandlerExperimental::GetSubimageYPos()
{
  return subimage_y_pos_;
}

int ParameterHandlerExperimental::GetSubimageXSize()
{
  return subimage_x_size_;
}

int ParameterHandlerExperimental::GetSubimageYSize()
{
  return subimage_y_size_;
}

int ParameterHandlerExperimental::GetImageSubdivisions()
{
  return image_subdivisions_;
}

int ParameterHandlerExperimental::GetBlurType()
{
  return blur_type_;
}

int ParameterHandlerExperimental::GetBlurRadius()
{
  return blur_radius_;
}

int ParameterHandlerExperimental::GetBlurSigma()
{
  return blur_sigma_;
}

int ParameterHandlerExperimental::GetThresholdType()
{
  return threshold_type_;
}

int ParameterHandlerExperimental::GetThresholdValue()
{
  return threshold_value_;
}

int ParameterHandlerExperimental::GetMorphologicalOperator()
{
  return morphological_operator_;
}

int ParameterHandlerExperimental::GetMorphologicalElement()
{
  return morphological_element_;
}

int ParameterHandlerExperimental::GetMorphologicalSize()
{
  return morphological_size_;
}

int ParameterHandlerExperimental::GetMinContourArea()
{
  return min_contour_area_;
}

Real ParameterHandlerExperimental::GetAreaIncrease()
{
  return area_increase_;
}

Real ParameterHandlerExperimental::GetConvexityDefectMagnitude()
{
  return convexity_defect_magnitude_;
}

Real ParameterHandlerExperimental::GetDataAssociationCost()
{
  return data_association_cost_;
}

Real ParameterHandlerExperimental::GetSecondaryDataAssociationCost()
{
  return secondary_data_association_cost_;
}

Real ParameterHandlerExperimental::GetHeightToWidthRatio()
{
  return height_to_width_ratio_;
}

Real ParameterHandlerExperimental::GetCenterOfMassDistance()
{
  return center_of_mass_distance_;
}

int ParameterHandlerExperimental::GetRoiMargin()
{
  return roi_margin_;
}

Real ParameterHandlerExperimental::GetNlMeansDenoisingH()
{
  return nl_means_denoising_h_;
}

int ParameterHandlerExperimental::GetNlMeansDenoisingTemplateWindowSize()
{
  return nl_means_denoising_template_window_size_;
}

int ParameterHandlerExperimental::GetNlMeansDenoisingSearchWindowSize()
{
  return nl_means_denoising_search_window_size_;
}

Real ParameterHandlerExperimental::GetContrast()
{
  return contrast_;
}

int ParameterHandlerExperimental::GetBrightness()
{
  return brightness_;
}