//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_IMAGEPROCESSINGENGINE_HPP
#define SPRMULTITARGETTRACKING_IMAGEPROCESSINGENGINE_HPP

#include "../Definitions.hpp"
#include "../Parameters/ParameterHandlerExperimental.hpp"

#include <fstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <eigen3/Eigen/Dense>

class ImageProcessingEngine
{
 public:

  explicit ImageProcessingEngine(ParameterHandlerExperimental &parameter_handler);
  ~ImageProcessingEngine();

  void CreateNewImageProcessingOutputFile(ParameterHandlerExperimental & parameter_handler);
  void RetrieveBacterialData(int image, std::vector<Eigen::VectorXf> &detections);
  void ProcessAdditionalDetections(const std::vector<int> &indexes_to_unassigned_detections,
                                   std::vector<Eigen::VectorXf> &additional_detections,
                                   const std::vector<Eigen::VectorXf> &detections);

  const cv::Mat &GetSourceImage();
  void RetrieveSourceImage(int image);
  const cv::Mat & GetSourceImage(int image);
  const std::vector<cv::Point>& GetContour(int idx);

 private:

  static int test_image_counter_;

  ParameterHandlerExperimental &parameter_handler_;
  std::ofstream image_processing_output_file_;

  cv::Mat source_image_;
  cv::Mat gray_image_;
  cv::Mat blurred_background_image_;
  cv::Mat illumination_corrected_image_;
  cv::Mat blur_image_;
  cv::Mat closing_image_;
  cv::Mat subtracted_image_;
  cv::Mat threshold_image_;
  cv::Mat contour_image_;
  cv::Mat morphology_image_;
  cv::Mat edge_image_;
  cv::Mat convexity_defects_image_;
  cv::Mat disconnected_image_;

  std::vector<std::vector<cv::Point>> contours_;
  std::vector<cv::Moments> mu_;

  
  void IncreaseContrast(const cv::Mat &I, cv::Mat &O);
  void CorrectForIllumination(const cv::Mat &I, cv::Mat &O);
  void ApplyBlurFilter(const cv::Mat &I, cv::Mat &O);
  void SubtractClosingImage(const cv::Mat &I, cv::Mat &O);
  void ApplyThreshold(const cv::Mat &I, cv::Mat &O);
  void FindContours(const cv::Mat &I);
  void DrawContours();
  void SaveImage(const cv::Mat &I, int image);
  void SaveDetectedObjects(int image, std::vector<Eigen::VectorXf> &detections);

  void ApplyMorphologicalTransform(int morph_operator, int morph_element, int morph_size);
  void DetectEdges();
  void AnalyzeConvexityDefects();
  void AnalyzeConvexityDefectsBasedOnCrossSection();
  void FindImprovedContourCenters();
  void DisconnectBacteria();
  void StandardizeImage(cv::Mat &image);

};
#endif //SPRMULTITARGETTRACKING_IMAGEPROCESSINGENGINE_HPP
