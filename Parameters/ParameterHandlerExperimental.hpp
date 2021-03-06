//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_PARAMETERHANDLER_HPP
#define SPRMULTITARGETTRACKING_PARAMETERHANDLER_HPP

#include "../Definitions.hpp"

class ParameterHandlerExperimental
{
 public:

  explicit ParameterHandlerExperimental(const std::string &file_name);
  ~ParameterHandlerExperimental();

  const std::string &GetInputFolder();
  const std::string &GetFileName0();
  const std::string &GetFileName1();
  const std::string &GetOriginalImagesSubfolder();
  const std::string &GetImageProcessingSubfolder();
  const std::string &GetImageProcessingOutputFileName();
  const std::string &GetKalmanFilterSubfolder();
  const std::string &GetKalmanFilterOutputFileName();
  const std::string &GetKalmanFilterMatlabOutputFileName();
  const std::string &GetTrackLinkingSubfolder();
  const std::string &GetTrackLinkingOutputFileName();
  const std::string &GetTrackLinkingMatlabOutputFileName();
  const std::string &GetDataAnalysisSubfolder();

  int GetFirstImage();
  int GetLastImage();
  int GetSubimageXPos();
  int GetSubimageYPos();
  int GetSubimageXSize();
  int GetSubimageYSize();
  int GetImageSubdivisions();
  int GetBlurType();
  int GetBlurRadius();
  int GetBlurSigma();
  int GetThresholdType();
  int GetThresholdValue();
  int GetMorphologicalOperator();
  int GetMorphologicalElement();
  int GetMorphologicalSize();
  int GetMinContourArea();
  Real GetAreaIncrease();
  Real GetConvexityDefectMagnitude();
  Real GetLowerDataAssociationCost();
  Real GetUpperDataAssociationCost();
  Real GetSecondaryDataAssociationCost();
  Real GetHeightToWidthRatio();
  Real GetCenterOfMassDistance();
  int GetRoiMargin();
  Real GetNlMeansDenoisingH();
  int GetNlMeansDenoisingTemplateWindowSize();
  int GetNlMeansDenoisingSearchWindowSize();
  Real GetContrast();
  int GetBrightness();
  int GetBackgroundSubtractionCoefficient();
  int GetTrackLinkingRoiMargin();
  int GetTrackLinkingIntersectionTime();
  int GetTrackLinkingLagTime();
  Real GetTrackLinkingDataAssociationCost();
  int GetMinTrajectoryLength();

 private:

  std::string input_folder_;
  std::string file_name_0_;
  std::string file_name_1_;
  std::string original_images_subfolder_;
  std::string image_processing_subfolder_;
  std::string image_processing_output_file_name_;
  std::string kalman_filter_subfolder_;
  std::string kalman_filter_output_file_name_;
  std::string kalman_filter_matlab_output_file_name_;
  std::string track_linking_subfolder_;
  std::string track_linking_output_file_name_;
  std::string track_linking_matlab_output_file_name_;
  std::string data_analysis_subfolder_;

  int first_image_;
  int last_image_;
  int subimage_x_pos_;
  int subimage_y_pos_;
  int subimage_x_size_;
  int subimage_y_size_;
  int image_subdivisions_;
  int blur_type_;
  int blur_radius_;
  int blur_sigma_;
  int threshold_type_;
  int threshold_value_;
  int morphological_operator_;
  int morphological_element_;
  int morphological_size_;
  int min_contour_area_;
  Real area_increase_;
  Real convexity_defect_magnitude_;
  Real lower_data_association_cost_;
  Real upper_data_association_cost_;
  Real secondary_data_association_cost_;
  Real height_to_width_ratio_;
  Real center_of_mass_distance_;
  int roi_margin_;
  Real nl_means_denoising_h_;
  int nl_means_denoising_template_window_size_;
  int nl_means_denoising_search_window_size_;
  Real contrast_;
  int brightness_;
  int background_subtraction_coefficient_;
  int track_linking_roi_margin_;
  int track_linking_intersection_time_;
  int track_linking_lag_time_;
  Real track_linking_data_association_cost_;
  int min_trajectory_length_;

};

#endif //SPRMULTITARGETTRACKING_PARAMETERHANDLER_HPP
