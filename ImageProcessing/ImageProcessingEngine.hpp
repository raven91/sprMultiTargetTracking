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
#include <opencv2/photo/photo.hpp>

#include <eigen3/Eigen/Dense>

class ImageProcessingEngine
{
 public:

  explicit ImageProcessingEngine(ParameterHandlerExperimental &parameter_handler);
  ~ImageProcessingEngine();

  void CreateNewImageProcessingOutputFile(ParameterHandlerExperimental &parameter_handler);
  void RetrieveBacterialData(int image, std::vector<Eigen::VectorXd> &detections);
  void ComposeImageForFilterOutput(int image_idx, cv::Mat &image);

  const cv::Mat &GetSourceImage();
  const cv::Mat &GetSourceImage(int image);
  const std::vector<cv::Point> &GetContour(int idx);

 private:

  static int test_image_counter_;

  ParameterHandlerExperimental &parameter_handler_;
  std::ofstream image_processing_output_file_;

  cv::Mat source_image_;
  cv::Mat gray_image_;
  cv::Mat blurred_background_image_;
  cv::Mat illumination_corrected_image_;
  cv::Mat denoised_image_;
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

  void RetrieveSourceImage(int image);
  void IncreaseContrast(const cv::Mat &I, cv::Mat &O);
  void SubtractBackgroundNoise(const cv::Mat &I, cv::Mat &O);
  void CorrectForIllumination(const cv::Mat &I, cv::Mat &O);
  void ApplyBlurFilter(const cv::Mat &I, cv::Mat &O);
  void SubtractClosingImage(const cv::Mat &I, cv::Mat &O);
  void ApplyThreshold(const cv::Mat &I, cv::Mat &O);
  void FindContours(const cv::Mat &I);
  void FindImprovedContours(const cv::Mat &I);
  void DrawContours();
  void SaveImage(const cv::Mat &I, int image);
  void SaveDetectedObjects(int image, std::vector<Eigen::VectorXd> &detections);
  bool IsContourInRoi(const std::vector<cv::Point> &contour);
  void AnalyzeConvexityDefectsOnePass(const cv::Mat &I, cv::Mat &O);
  void AnalyzeConvexityDefectsRecursively();
  std::vector<std::vector<cv::Point>> AnalyzeConvexityDefectsRecursively(const std::vector<cv::Point> &contour,
                                                                         const cv::Rect &bounding_rect,
                                                                         int recursion_counter);
  bool CorrectConvexityDefect(const std::vector<cv::Vec4i>::iterator &cd_it,
                              const std::vector<cv::Point> &contour,
                              cv::Mat &image);
  void FindSubcontours(const cv::Mat &subcontour_image, std::vector<std::vector<cv::Point>> &new_subcontours);
  void StandardizeImage(cv::Mat &image);

};
#endif //SPRMULTITARGETTRACKING_IMAGEPROCESSINGENGINE_HPP
