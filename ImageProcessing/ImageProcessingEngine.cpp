//
// Created by Nikita Kruk on 27.11.17.
//

#include "ImageProcessingEngine.hpp"

#include <sstream>
#include <iomanip> // std::setfill, std::setw
#include <vector>
#include <iterator>
#include <algorithm> // std::max_element, std::remove_if, std::copy

int ImageProcessingEngine::test_image_counter_ = 0;

ImageProcessingEngine::ImageProcessingEngine(ParameterHandlerExperimental &parameter_handler) :
    parameter_handler_(parameter_handler)
{
  source_image_ = cv::Mat::zeros(0, 0, CV_8UC3);
  gray_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  blurred_background_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  illumination_corrected_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  denoised_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  blur_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  closing_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  subtracted_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  threshold_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  contour_image_ = cv::Mat::zeros(0, 0, CV_8UC3);
  morphology_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  edge_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  convexity_defects_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
  disconnected_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
}

ImageProcessingEngine::~ImageProcessingEngine()
{
  image_processing_output_file_.close();
}

void ImageProcessingEngine::CreateNewImageProcessingOutputFile(ParameterHandlerExperimental &parameter_handler)
{
  std::string image_processing_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler.GetDataAnalysisSubfolder()
          + parameter_handler.GetImageProcessingOutputFileName();
  image_processing_output_file_.open(image_processing_output_file_name, std::ios::out | std::ios::trunc);
  assert(image_processing_output_file_.is_open());
}

void ImageProcessingEngine::RetrieveBacterialData(int image, std::vector<Eigen::VectorXf> &detections)
{
  std::cout << "image processing: image#" << image << std::endl;

  RetrieveSourceImage(image);
  IncreaseContrast(source_image_, gray_image_);
  SubtractBackgroundNoise(gray_image_, denoised_image_);
//  CorrectForIllumination(gray_image_, illumination_corrected_image_);
//  SubtractClosingImage(illumination_corrected_image_, subtracted_image_);
//  ApplyBlurFilter(subtracted_image_,
//                  blur_image_);
  ApplyThreshold(denoised_image_, threshold_image_);
  FindContours(threshold_image_);
  AnalyzeConvexityDefects(threshold_image_, convexity_defects_image_);
  FindImprovedContours(convexity_defects_image_);

  DrawContours();
  SaveImage(contour_image_, image);
  SaveDetectedObjects(image, detections);
}

/*void ImageProcessingEngine::RetrieveBacterialPositions()
{
    std::string input_folder("/Users/nikita/Documents/spr/20161026/100x_01-BF0_1ms_1to200_300um_yepd_v5/");
    std::string file_name_0("img_");//("img_channel000_position000_time");//("img_")
    std::string file_name_1("_01-BF0_000.tif");//("_z000.tif");//("_01-BF0_000.tif")
    std::string output_folder = input_folder + std::string("ImageProcessing/");
    std::string image_processing_output_file_name = output_folder + std::string("image_processing_output.txt");
    std::ofstream image_processing_output_file(image_processing_output_file_name, std::ios::out | std::ios::trunc);

    for (int i = parameter_handler_.GetFirstImage(); i <= parameter_handler_.GetLastImage(); ++i)
    {
        std::ostringstream image_name_buf;
        image_name_buf << input_folder << file_name_0 << std::setfill('0') << std::setw(9) << i << file_name_1;
        std::string image_name = image_name_buf.str();

        std::cout << image_name << std::endl;

        source_image_ = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
        if (!source_image_.data)
        {
            std::cerr << "image not found" << std::endl;
        }
        source_image_ = cv::Mat(source_image_, cv::Rect(1024, 1024, 1024, 1024));

        cv::resize(source_image_, source_image_, cv::Size(), 1.0, 1.0, cv::INTER_LINEAR);
        cv::cvtColor(source_image_, gray_image_, cv::COLOR_BGR2GRAY);
        int subdivision = 0;
        for (int xs = 0; xs < subdivision; ++xs)
        {
            for (int ys = 0; ys < subdivision; ++ys)
            {
                cv::Mat sub_img = cv::Mat(gray_image_, cv::Rect(gray_image_.cols / subdivision * xs, gray_image_.rows / subdivision * ys, gray_image_.cols / subdivision, gray_image_.rows / subdivision));
                cv::equalizeHist(sub_img, sub_img);
            }
        }

        ApplyBlurFilter(1, 2);
        ApplyThreshold(0, 130);
//		ApplyMorphologicalTransform(2, 1, 3);
//		DetectEdges();
        DisconnectBacteria();
        FindContours();
        AnalyzeConvexityDefects();
        FindImprovedContours();

        std::ostringstream output_image_name_buf;
        output_image_name_buf << output_folder << file_name_0 << std::setfill('0') << std::setw(9) << i << file_name_1;
        std::string output_image_name = output_image_name_buf.str();
        cv::imwrite(output_image_name, contour_image_);

        image_processing_output_file << i << " " << contours_.size() << " ";
        cv::Vec4f fitted_line;
        for (int b = 0; b < (int)contours_.size(); ++b)
        {
            cv::fitLine(contours_[b], fitted_line, cv::DIST_L2, 0, 0.01, 0.01);
            image_processing_output_file << b << " " << mu_[b].m10 / mu_[b].m00 << " " << mu_[b].m01 / mu_[b].m00 << " " << cv::contourArea(contours_[b]) << " " << std::atan(fitted_line(1) / fitted_line(0)) << " ";
        }
        image_processing_output_file << std::endl;
    }
}*/

void ImageProcessingEngine::RetrieveSourceImage(int image)
{
  std::ostringstream image_name_buf;
  image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetOriginalImagesSubfolder()
                 << parameter_handler_.GetFileName0() << std::setfill('0')
                 << std::setw(9) << image << parameter_handler_.GetFileName1();
  std::string image_name = image_name_buf.str();

  source_image_ = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
  assert(source_image_.data != NULL);
  source_image_ = cv::Mat(source_image_,
                          cv::Rect(parameter_handler_.GetSubimageXPos(),
                                   parameter_handler_.GetSubimageYPos(),
                                   parameter_handler_.GetSubimageXSize(),
                                   parameter_handler_.GetSubimageYSize()));
  cv::resize(source_image_, source_image_, cv::Size(), 1.0, 1.0, cv::INTER_LINEAR);
}

void ImageProcessingEngine::IncreaseContrast(const cv::Mat &I, cv::Mat &O)
{
  cv::cvtColor(I, O, cv::COLOR_BGR2GRAY);
  for (int xs = 0; xs < parameter_handler_.GetImageSubdivisions(); ++xs)
  {
    for (int ys = 0; ys < parameter_handler_.GetImageSubdivisions(); ++ys)
    {
      cv::Mat sub_img = cv::Mat(O,
                                cv::Rect(O.cols / parameter_handler_.GetImageSubdivisions() * xs,
                                         O.rows / parameter_handler_.GetImageSubdivisions() * ys,
                                         O.cols / parameter_handler_.GetImageSubdivisions(),
                                         O.rows / parameter_handler_.GetImageSubdivisions()));
      cv::equalizeHist(sub_img, sub_img);
    }
  }

  for (int y = 0; y < O.rows; y++)
  {
    for (int x = 0; x < O.cols; x++)
    {
      O.at<uchar>(y, x) = cv::saturate_cast<uchar>(
          parameter_handler_.GetContrast() * (O.at<uchar>(y, x)) + parameter_handler_.GetBrightness());
    }
  }
}

void ImageProcessingEngine::SubtractBackgroundNoise(const cv::Mat &I, cv::Mat &O)
{
  blurred_background_image_ = I.clone();
//  cv::fastNlMeansDenoising(blurred_background_image_.clone(),
//                           blurred_background_image_,
//                           parameter_handler_.GetNlMeansDenoisingH(),
//                           parameter_handler_.GetNlMeansDenoisingTemplateWindowSize(),
//                           parameter_handler_.GetNlMeansDenoisingSearchWindowSize());
  cv::GaussianBlur(blurred_background_image_.clone(),
                   blurred_background_image_,
                   cv::Size(2 * parameter_handler_.GetBlurRadius() + 1, 2 * parameter_handler_.GetBlurRadius() + 1),
                   parameter_handler_.GetBlurSigma(),
                   parameter_handler_.GetBlurSigma());
  O = I - blurred_background_image_;
}

void ImageProcessingEngine::CorrectForIllumination(const cv::Mat &I, cv::Mat &O)
{
  blurred_background_image_ = I.clone();
  cv::blur(blurred_background_image_.clone(),
           blurred_background_image_,
           cv::Size(2 * 101 + 1, 2 * 101 + 1),
           cv::Point(-1, -1));
  O = (gray_image_ - blurred_background_image_) + cv::mean(blurred_background_image_)[0];
}

void ImageProcessingEngine::ApplyBlurFilter(const cv::Mat &I, cv::Mat &O)
{
  /*
   0: Normalized Block Filter
   1: Gaussian Filter
   2: Median Filter
   3: Bilateral Filter
   */
  int blur_type = parameter_handler_.GetBlurType();
  int blur_radius = parameter_handler_.GetBlurRadius();

  switch (blur_type)
  {
    case 0: // Normalized Block Filter
      cv::blur(I, O, cv::Size(2 * blur_radius + 1, 2 * blur_radius + 1), cv::Point(-1, -1));
      break;

    case 1: // Gaussian Filter
      cv::GaussianBlur(I, O, cv::Size(2 * blur_radius + 1, 2 * blur_radius + 1), 3, 3);
      break;

    case 2: // Median Filter
      cv::medianBlur(I, O, 2 * blur_radius + 1);
      break;

    case 3: // Bilateral Filter
      cv::bilateralFilter(I, O, 2 * blur_radius + 1, blur_radius * 2, blur_radius / 2);
      break;

    default:std::cerr << "wrong blur index" << std::endl;
      break;
  }
}

/**
 *  https://clouard.users.greyc.fr/Pantheon/experiments/illumination-correction/index-en.html#retrospective
 */
void ImageProcessingEngine::SubtractClosingImage(const cv::Mat &I, cv::Mat &O)
{
  /*
   0: Erosion
   1: Dilation
   2: Opening
   3: Closing
   4: Gradient
   5: Top Hat
   6: Black Hat
   */
  int morphological_element = parameter_handler_.GetMorphologicalElement();
  int morphological_size = parameter_handler_.GetMorphologicalSize();
  int morphological_operator = parameter_handler_.GetMorphologicalOperator();
  cv::Mat element = cv::getStructuringElement(morphological_element,
                                              cv::Size(2 * morphological_size + 1, 2 * morphological_size + 1),
                                              cv::Point(morphological_size, morphological_size));
  cv::morphologyEx(I, closing_image_, morphological_operator, element);

  O = (I - closing_image_) + cv::mean(closing_image_)[0];
  cv::equalizeHist(O, O);
}

void ImageProcessingEngine::ApplyThreshold(const cv::Mat &I, cv::Mat &O)
{
  /*
   0: Binary
   1: Binary Inverted
   2: Threshold Truncated
   3: Threshold to Zero
   4: Threshold to Zero Inverted
   */
  const int max_binary_threshold_value = 255;
  cv::threshold(I,
                O,
                parameter_handler_.GetThresholdValue(),
                max_binary_threshold_value,
                parameter_handler_.GetThresholdType());
}

// search for white objects on a black background
void ImageProcessingEngine::FindContours(const cv::Mat &I)
{
  int offset = 2;
  cv::Mat extended_contour_image;
  contour_image_ = I.clone();
  cv::copyMakeBorder(contour_image_,
                     extended_contour_image,
                     offset,
                     offset,
                     offset,
                     offset,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(0, 0, 0));

  contours_.clear();
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(extended_contour_image,
                   contours_,
                   hierarchy,
                   cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE,
                   cv::Point(-offset, -offset));
  contours_.erase(std::remove_if(contours_.begin(),
                                 contours_.end(),
                                 [&](const std::vector<cv::Point> &contour)
                                 {
                                   double contour_area = cv::contourArea(contour);
//                                   cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
//                                   std::vector<cv::Point> convex_hull;
//                                   cv::convexHull(cv::Mat(contour), convex_hull, false);
//                                   double convex_hull_area = cv::contourArea(convex_hull);
//                                   cv::Moments mu = cv::moments(contour, true);
//                                   cv::Point center = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
                                   return (contour_area < parameter_handler_.GetMinContourArea());
//                                   return (contour_area < parameter_handler_.GetMinContourArea())
//                                       || (std::max(rotated_rect.size.width, rotated_rect.size.height)
//                                           / std::min(rotated_rect.size.width, rotated_rect.size.height)
//                                           < parameter_handler_.GetHeightToWidthRatio())
//                                       || (convex_hull_area / contour_area > parameter_handler_.GetAreaIncrease())
//                                       || (cv::pointPolygonTest(contour, center, true)
//                                           < -parameter_handler_.GetCenterOfMassDistance());
                                 }),
                  contours_.end());
}

void ImageProcessingEngine::FindImprovedContours(const cv::Mat &I)
{
  int offset = 2;
  cv::Mat extended_contour_image;
  contour_image_ = I.clone();
  cv::copyMakeBorder(contour_image_,
                     extended_contour_image,
                     offset,
                     offset,
                     offset,
                     offset,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(0, 0, 0));

  contours_.clear();
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(extended_contour_image,
                   contours_,
                   hierarchy,
                   cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE,
                   cv::Point(-offset, -offset));
  contours_.erase(std::remove_if(contours_.begin(),
                                 contours_.end(),
                                 [&](const std::vector<cv::Point> &contour)
                                 {
                                   double contour_area = cv::contourArea(contour);
//                                   cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
//                                   std::vector<cv::Point> convex_hull;
//                                   cv::convexHull(cv::Mat(contour), convex_hull, false);
//                                   double convex_hull_area = cv::contourArea(convex_hull);
//                                   cv::Moments mu = cv::moments(contour, true);
//                                   cv::Point center = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
                                   return (contour_area < parameter_handler_.GetMinContourArea());
//                                   return (contour_area < parameter_handler_.GetMinContourArea())
//                                       || (std::max(rotated_rect.size.width, rotated_rect.size.height)
//                                           / std::min(rotated_rect.size.width, rotated_rect.size.height)
//                                           < parameter_handler_.GetHeightToWidthRatio())
//                                       || (convex_hull_area / contour_area > parameter_handler_.GetAreaIncrease())
//                                       || (cv::pointPolygonTest(contour, center, true)
//                                           < -parameter_handler_.GetCenterOfMassDistance());
                                 }),
                  contours_.end());
}

void ImageProcessingEngine::DrawContours()
{
  // draw colored contours
/*  cv::RNG rng(12345); // random color generator
  cv::Mat fitted_contours_image = cv::Mat::zeros(contour_image.size(), CV_8UC3);
  for (int i = 0; i < contours_.size(); ++i)
  {
    cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    cv::drawContours(fitted_contours_image, contours_, i, color, -1, cv::LINE_AA, hierarchy, 0, cv::Point(0, 0));
//			cv::ellipse(fitted_contours_image, min_ellipse[i], color, 2, 8);
//			cv::Point2f rect_points[4];
//			min_rect[i].points(rect_points);
//			for (int j = 0; j < 4; ++j)
//			{
//				cv::line(fitted_contours_image, rect_points[j], rect_points[(j + 1) % 4], color, 2, 8);
//			}
//		}
  }
  contour_image_ = fitted_contours_image;
*/

  // draw contour centers
  cv::Mat indexed_contours_image;
//  gray_image_.convertTo(indexed_contours_image, CV_8UC3);
  cv::cvtColor(gray_image_, indexed_contours_image, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < (int) contours_.size(); ++i)
  {
    std::string index = std::to_string(i);
    cv::Moments mu = cv::moments(contours_[i], true);
    cv::Point center = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);

    // define color based on the distance from the boundary
    cv::Scalar color;
    if (IsContourInRoi(contours_[i]))
    {
      color = cv::Scalar(0, 255, 0);
    } else
    {
      color = cv::Scalar(0, 0, 255);
    }
    cv::circle(indexed_contours_image, center, 4, color, -1, 8);
    cv::putText(indexed_contours_image, index, center, cv::FONT_HERSHEY_DUPLEX, 0.4, color);
  }
  contour_image_ = indexed_contours_image;
}

void ImageProcessingEngine::SaveImage(const cv::Mat &I, int image)
{
  std::ostringstream output_image_name_buf;
  output_image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetImageProcessingSubfolder()
                        << parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << image
                        << parameter_handler_.GetFileName1();
  std::string output_image_name = output_image_name_buf.str();
  cv::imwrite(output_image_name, I);
}

// according to the format
// i -> x_i y_i v_x_i v_y_i area_i slope_i width_i height_i
void ImageProcessingEngine::SaveDetectedObjects(int image, std::vector<Eigen::VectorXf> &detections)
{
  detections.clear();

  image_processing_output_file_ << image << " " << contours_.size() << " ";

  cv::Vec4f fitted_line;
  cv::RotatedRect min_rect;
  Eigen::VectorXf new_detection(kNumOfExtractedFeatures);
  cv::Moments mu;

  for (int b = 0; b < (int) contours_.size(); ++b)
  {
    mu = cv::moments(contours_[b], true);
    new_detection(0) = Real(mu.m10 / mu.m00);
    new_detection(1) = Real(mu.m01 / mu.m00);

    new_detection(2) = 0.0;
    new_detection(3) = 0.0;

    new_detection(4) = Real(cv::contourArea(contours_[b]));

    // fitted_line = (vx, vy, x0, y0)
    cv::fitLine(contours_[b], fitted_line, cv::DIST_L2, 0, 0.01, 0.01);
    // TODO: verify slope range
    // [-pi,+pi]
    new_detection(5) = std::atan2(fitted_line[1], fitted_line[0]);

    min_rect = cv::minAreaRect(cv::Mat(contours_[b]));
    new_detection(6) = std::min(min_rect.size.width, min_rect.size.height);
    new_detection(7) = std::max(min_rect.size.width, min_rect.size.height);

    // consider the objects that are not at the boundary
    if (IsContourInRoi(contours_[b]))
    {
      detections.push_back(new_detection);
      image_processing_output_file_ << b << " " << new_detection(0) << " " << new_detection(1) << " "
                                    << new_detection(2)
                                    << " " << new_detection(3) << " " << new_detection(4) << " " << new_detection(5)
                                    << " " << new_detection(6) << " " << new_detection(7) << " ";
    }
  }
  image_processing_output_file_ << std::endl;
}

bool ImageProcessingEngine::IsContourInRoi(const std::vector<cv::Point> &contour)
{
  int margin = parameter_handler_.GetRoiMargin();
  std::vector<cv::Point> roi =
      {
          cv::Point(margin, margin),
          cv::Point(margin, parameter_handler_.GetSubimageYSize() - margin),
          cv::Point(parameter_handler_.GetSubimageXSize() - margin, parameter_handler_.GetSubimageYSize() - margin),
          cv::Point(parameter_handler_.GetSubimageXSize() - margin, margin)
      };
/*  cv::Mat mask_with_roi =
      cv::Mat::zeros(parameter_handler_.GetSubimageXSize(), parameter_handler_.GetSubimageYSize(), CV_8UC1);
  cv::rectangle(mask_with_roi, roi[0], roi[2], 255, CV_FILLED, 8, 0);
//  cv::namedWindow("roi", cv::WINDOW_AUTOSIZE);
//  cv::imshow("roi", mask_with_roi);
//  cv::waitKey(1);

  cv::RotatedRect min_rect = cv::minAreaRect(cv::Mat(contour));
  cv::Point2f min_rect_points_f[4];
  min_rect.points(min_rect_points_f);
  cv::Mat mask_with_min_rect =
      cv::Mat::zeros(parameter_handler_.GetSubimageXSize(), parameter_handler_.GetSubimageYSize(), CV_8UC1);
  cv::Point min_rect_points_i[4];
  for (int i = 0; i < 4; ++i)
  {
    min_rect_points_i[i] = min_rect_points_f[i];
  }
  cv::fillConvexPoly(mask_with_min_rect, min_rect_points_i, 4, 255, 8, 0);
//  cv::namedWindow("min_rect", cv::WINDOW_AUTOSIZE);
//  cv::imshow("min_rect", mask_with_min_rect);
//  cv::waitKey(1);

  cv::Mat
      mask = cv::Mat::zeros(parameter_handler_.GetSubimageXSize(), parameter_handler_.GetSubimageYSize(), CV_8UC1);
  cv::bitwise_and(mask_with_roi, mask_with_min_rect, mask);
//  cv::namedWindow("abc", cv::WINDOW_AUTOSIZE);
//  cv::imshow("abc", mask);
//  cv::waitKey(1);

  return (cv::sum(mask)[0] > 0.0);
  */
  cv::Moments mu = cv::moments(contour, true);
  cv::Point2f center = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);

  return cv::pointPolygonTest(roi, center, false) >= 0.0;
}

void ImageProcessingEngine::AnalyzeConvexityDefects(const cv::Mat &I, cv::Mat &O)
{
  O = I.clone();

  for (int i = 0; i < contours_.size(); ++i)
  {
    const std::vector<cv::Point> &contour = contours_[i];
    if (contour.size() > 2) // convex hull may be computed for contours with number of points > 2
    {
      std::vector<int> convex_hull_indices;
//      std::vector<cv::Point> convex_hull;
      cv::convexHull(contour, convex_hull_indices, false, false);
//      cv::convexHull(contour, convex_hull, false, true);
//      if (cv::contourArea(convex_hull) / cv::contourArea(contour) > parameter_handler_.GetAreaIncrease())
      if (!cv::isContourConvex(contour))
      {
        std::vector<cv::Vec4i> convexity_defects;
        cv::convexityDefects(contour, convex_hull_indices, convexity_defects);
        for (std::vector<cv::Vec4i>::iterator cd_it = convexity_defects.begin(); cd_it != convexity_defects.end();
             ++cd_it)
        {
          double max_fixpt_depth = (*cd_it)[3] / 256.0;
          if (max_fixpt_depth > parameter_handler_.GetConvexityDefectMagnitude())
          {
            const cv::Point2f &start_point = contour[(*cd_it)[0]];
            const cv::Point2f &end_point = contour[(*cd_it)[1]];
            const cv::Point2f &farthest_point = contour[(*cd_it)[2]];

            cv::Point2f middle_point(0.5f * (start_point.x + end_point.x), 0.5f * (start_point.y + end_point.y));
            cv::Point2f inward_cutting_vec = farthest_point - middle_point;
            double inward_cutting_vec_norm =
                std::sqrt(inward_cutting_vec.x * inward_cutting_vec.x + inward_cutting_vec.y * inward_cutting_vec.y);
            inward_cutting_vec /= inward_cutting_vec_norm;
            double intersection_length = 0.0;
            do
            {
              intersection_length += 1.0;
            } while (cv::pointPolygonTest(contour,
                                          middle_point
                                              + inward_cutting_vec * (inward_cutting_vec_norm + intersection_length),
                                          false) >= 0.0);
            if (intersection_length <= 15.0) // if the cut is transverse to bacteria
            {
              cv::line(O,
                       farthest_point,
                       middle_point + inward_cutting_vec * (inward_cutting_vec_norm + intersection_length),
                       cv::Scalar(0, 0, 0),
                       2,
                       cv::LINE_AA);
            }
          }
        } // cd_it
      }
    }
  } // i
}

const cv::Mat &ImageProcessingEngine::GetSourceImage()
{
  return source_image_;
}

const cv::Mat &ImageProcessingEngine::GetSourceImage(int image)
{
  RetrieveSourceImage(image);
  return source_image_;
}

void ImageProcessingEngine::ComposeImageForFilterOutput(int image_idx, cv::Mat &image)
{
  RetrieveSourceImage(image_idx);
  IncreaseContrast(source_image_, gray_image_);
  cv::bitwise_not(gray_image_, gray_image_);
  cv::equalizeHist(gray_image_, gray_image_);
  cv::cvtColor(gray_image_, image, CV_GRAY2BGR);
}

const std::vector<cv::Point> &ImageProcessingEngine::GetContour(int idx)
{
  return contours_[idx];
}

void ImageProcessingEngine::StandardizeImage(cv::Mat &image)
{
  double min = 0.0, max = 0.0;
  cv::minMaxLoc(image, &min, &max);
  image -= min;
  image.clone().convertTo(image, CV_8UC1, 255.0 / (max - min));
}