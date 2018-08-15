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


//
// MODIFIED by Stanislav Stepaniuk on  14.08.18
//
ImageProcessingEngine::ImageProcessingEngine(ParameterHandlerExperimental &parameter_handler) :
	parameter_handler_(parameter_handler)
{
	source_image_ = cv::Mat::zeros(0, 0, CV_8UC3);
	gray_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	blurred_background_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	illumination_corrected_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	blur_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	closing_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	subtracted_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	threshold_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	contour_image_ = cv::Mat::zeros(0, 0, CV_8UC3);
	morphology_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	edge_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	convexity_defects_image_ = cv::Mat::zeros(0, 0, CV_8UC1);
	disconnected_image_ = cv::Mat::zeros(0, 0, CV_8UC1);


	// ISOLATED INTO A SEPARATE METHOD ==> CreateNewImageProcessingOutputFile

	//std::string image_processing_output_file_name =
	//    parameter_handler_.GetInputFolder() + parameter_handler.GetDataAnalysisSubfolder()
	//        + parameter_handler.GetImageProcessingOutputFileName();
	//image_processing_output_file_.open(image_processing_output_file_name, std::ios::out | std::ios::trunc);
	//assert(image_processing_output_file_.is_open());
}

ImageProcessingEngine::~ImageProcessingEngine()
{
	image_processing_output_file_.close();
}



//
// Created by Stanislav Stepaniuk on 14.08.18
//
// Separate Method for creating image processing output file
void ImageProcessingEngine::CreateNewImageProcessingOutputFile(ParameterHandlerExperimental &parameter_handler)
{
	std::string image_processing_output_file_name =
		parameter_handler_.GetInputFolder() + parameter_handler.GetDataAnalysisSubfolder()
		+ parameter_handler.GetImageProcessingOutputFileName();
	image_processing_output_file_.open(image_processing_output_file_name, std::ios::out | std::ios::trunc);
	assert(image_processing_output_file_.is_open());
}//END of Separate Method for creating image processing output file



void ImageProcessingEngine::RetrieveBacterialData(int image, std::vector<Eigen::VectorXf> &detections)
{
	RetrieveSourceImage(image);
	IncreaseContrast(source_image_, gray_image_);
	CorrectForIllumination(gray_image_, illumination_corrected_image_);
	SubtractClosingImage(illumination_corrected_image_, subtracted_image_);
	ApplyBlurFilter(subtracted_image_,
		blur_image_);
	ApplyThreshold(blur_image_, threshold_image_);
	FindContours(threshold_image_);

	//  ApplyBlurFilter(parameter_handler_.GetBlurType(), parameter_handler_.GetBlurRadius());
	//  ApplyThreshold(parameter_handler_.GetThresholdType(), parameter_handler_.GetThresholdValue());
	//	ApplyMorphologicalTransform(parameter_handler_.GetMorphologicalOperator(), parameter_handler_.GetMorphologicalElement(), parameter_handler_.GetMorphologicalSize());
	//	DetectEdges();
	//  DisconnectBacteria();
	//	FindContours();
	//	AnalyzeConvexityDefects();
	//	AnalyzeConvexityDefectsBasedOnCrossSection();
	//  FindImprovedContourCenters();

	DrawContours();
	//  SaveImage(contour_image_, image);
	SaveDetectedObjects(image, detections);
}

void ImageProcessingEngine::ProcessAdditionalDetections(const std::vector<int> &indexes_to_unassigned_detections,
	std::vector<Eigen::VectorXf> &additional_detections,
	const std::vector<Eigen::VectorXf> &detections)
{
	for (int i = 0; i < indexes_to_unassigned_detections.size(); ++i)
	{
		std::vector<cv::Point> &contour = contours_[indexes_to_unassigned_detections[i]];
		if (contour.size() > 2) // convex hull may be computed for contours with number of points > 2
		{
			std::vector<int> convex_hull_indices;
			//			std::vector<cv::Point> convex_hull;
			cv::convexHull(contour, convex_hull_indices, false, false);
			//			cv::convexHull(contour, convex_hull, false, true);
			std::vector<cv::Vec4i> convexity_defects;
			cv::convexityDefects(contour, convex_hull_indices, convexity_defects);
			if (convexity_defects.size() > 0)
			{
				std::vector<cv::Vec4i>::iterator max_iter = std::max_element(convexity_defects.begin(),
					convexity_defects.end(),
					[](const cv::Vec4i &a, const cv::Vec4i &b)
				{
					return a[3] < b[3];
				});
				if ((*max_iter)[3] / 256.0 > parameter_handler_.GetConvexityDefectMagnitude())
				{
					cv::Rect bounding_rect = cv::boundingRect(contour);
					cv::Mat bounding_image = disconnected_image_(bounding_rect).clone();
					cv::Mat mask = cv::Mat::zeros(bounding_image.size(), bounding_image.type());
					cv::drawContours(mask,
						contours_,
						indexes_to_unassigned_detections[i],
						cv::Scalar(255, 255, 255),
						-1,
						cv::LINE_AA,
						NULL,
						0,
						-bounding_rect.tl());
					bounding_image = mask.clone();

					const cv::Point2f &A = contour[(*max_iter)[0]];
					const cv::Point2f &B = contour[(*max_iter)[1]];
					const cv::Point2f &C = contour[(*max_iter)[2]];
					cv::Point2f D;
					if (A.x == B.x)
					{
						D.x = A.x;
						D.y = C.y;
					}
					else if (A.y == B.y)
					{
						D.x = C.x;
						D.y = A.y;
					}
					else
					{
						float m_AB = (A.y - B.y) / (A.x - B.x);
						float b_AB = -((A.y - B.y) * B.x - B.y * (A.x - B.x)) / (A.x - B.x);
						float m_CD = -1.0 / m_AB;
						float b_CD = C.y - m_CD * C.x;
						D.x = (b_CD - b_AB) / (m_AB - m_CD);
						D.y = m_CD * D.x + b_CD;
					}
					float increment = 1.0 / cv::norm(C - D);
					cv::Point2f point_along_defect(C + (C - D) * increment);
					while (cv::pointPolygonTest(contour, point_along_defect, false) >= 0.0)
					{
						point_along_defect += (C - D) * increment;
					}
					point_along_defect += (C - D) * increment;
					cv::Scalar color = cv::Scalar(0, 0, 0);
					cv::Point2f shifted_point_0(C.x - (C.x - D.x) * 2.0 * increment - bounding_rect.x,
						C.y - (C.y - D.y) * 2.0 * increment - bounding_rect.y);
					cv::Point2f shifted_point_1(point_along_defect.x - bounding_rect.x, point_along_defect.y - bounding_rect.y);
					cv::line(bounding_image, shifted_point_0, shifted_point_1, color, 4, cv::LINE_AA);

					int offset = 2;
					cv::Mat extended_bounding_image(bounding_image.size(), bounding_image.type());
					cv::copyMakeBorder(bounding_image,
						extended_bounding_image,
						offset,
						offset,
						offset,
						offset,
						cv::BORDER_CONSTANT,
						cv::Scalar(0, 0, 0));
					std::vector<std::vector<cv::Point>> bounding_contours;
					std::vector<cv::Vec4i> bounding_hierarchy;
					cv::findContours(extended_bounding_image,
						bounding_contours,
						bounding_hierarchy,
						cv::RETR_EXTERNAL,
						cv::CHAIN_APPROX_SIMPLE,
						cv::Point(-offset, -offset));
					// TODO: remove small contours
					std::vector<cv::Moments> bounding_mu = std::vector<cv::Moments>(bounding_contours.size());
					for (int m = 0; m < bounding_contours.size(); ++m)
					{
						bounding_mu[m] = cv::moments(bounding_contours[m], true);
					}

					//					cv::Mat color_image;
					//					cv::cvtColor(bounding_image, color_image, cv::COLOR_GRAY2BGR);
					//					cv::RNG rng(12345);
					//					for (int m = 0; m < bounding_contours.size(); ++m)
					//					{
					//						cv::drawContours(color_image, bounding_contours, m, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, cv::LINE_AA, NULL, 0);
					//					}
					//					cv::circle(bounding_image, shifted_point_0, 1, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
					//					cv::circle(bounding_image, shifted_point_1, 1, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));

					cv::Vec4f fitted_line;
					cv::RotatedRect min_rect;
					Eigen::VectorXf new_detection(kNumOfExtractedFeatures);
					if (bounding_contours.size() == 2)
					{
						new_detection(0) = bounding_rect.x + bounding_mu[0].m10 / bounding_mu[0].m00;
						new_detection(1) = bounding_rect.y + bounding_mu[0].m01 / bounding_mu[0].m00;
						new_detection(2) = 0.0;
						new_detection(3) = 0.0;
						new_detection(4) = cv::contourArea(bounding_contours[0]);
						cv::fitLine(bounding_contours[0], fitted_line, cv::DIST_L2, 0, 0.01, 0.01);
						new_detection(5) = std::atan(fitted_line[1] / fitted_line[0]);
						min_rect = cv::minAreaRect(cv::Mat(bounding_contours[0]));
						new_detection(6) = min_rect.size.width;
						new_detection(7) = min_rect.size.height;
						additional_detections.push_back(new_detection);

						new_detection(0) = bounding_rect.x + bounding_mu[1].m10 / bounding_mu[1].m00;
						new_detection(1) = bounding_rect.y + bounding_mu[1].m01 / bounding_mu[1].m00;
						new_detection(2) = 0.0;
						new_detection(3) = 0.0;
						new_detection(4) = cv::contourArea(bounding_contours[1]);
						cv::fitLine(bounding_contours[1], fitted_line, cv::DIST_L2, 0, 0.01, 0.01);
						new_detection(5) = std::atan(fitted_line[1] / fitted_line[0]);
						min_rect = cv::minAreaRect(cv::Mat(bounding_contours[1]));
						new_detection(6) = min_rect.size.width;
						new_detection(7) = min_rect.size.height;
						additional_detections.push_back(new_detection);
					}
					else
					{
						cv::imwrite(
							parameter_handler_.GetInputFolder() + std::string("debug/") + std::to_string(++test_image_counter_)
							+ std::string(".tif"), mask);
						std::cerr << "wrong number of new contours after kalman estimation" << std::endl;
					}
				}
				else // convexity defect is small
				{
					additional_detections.push_back(detections[indexes_to_unassigned_detections[i]]);
				}
			} // if (convexity_defects.size() > 0)
		} // if (contour.size() > 2)
	} // for i
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
FindImprovedContourCenters();

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

	std::cout << image_name << std::endl;

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

	//	std::vector<std::vector<cv::Point>> contours;
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
		cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
		std::vector<cv::Point> convex_hull;
		cv::convexHull(cv::Mat(contour), convex_hull, false);
		double convex_hull_area = cv::contourArea(convex_hull);
		cv::Moments mu = cv::moments(contour, true);
		cv::Point center = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
		return (contour_area < parameter_handler_.GetMinContourArea())
			|| (std::max(rotated_rect.size.width, rotated_rect.size.height)
				/ std::min(rotated_rect.size.width, rotated_rect.size.height)
				< parameter_handler_.GetHeightToWidthRatio())
			|| (convex_hull_area / contour_area > parameter_handler_.GetAreaIncrease())
			|| (cv::pointPolygonTest(contour, center, true)
				< -parameter_handler_.GetCenterOfMassDistance());
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
	for (int i = 0; i < contours_.size(); ++i)
	{
		std::string index = std::to_string(i);
		cv::Moments mu = cv::moments(contours_[i], true);
		cv::Point center = cv::Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
		cv::Scalar color = cv::Scalar(0, 255, 0);
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

	assert(image_processing_output_file_.is_open());
	image_processing_output_file_ << image << " " << contours_.size() << " ";

	cv::Vec4f fitted_line;
	cv::RotatedRect min_rect;
	Eigen::VectorXf new_detection(kNumOfExtractedFeatures);
	cv::Moments mu;

	for (int b = 0; b < (int)contours_.size(); ++b)
	{
		mu = cv::moments(contours_[b], true);
		new_detection(0) = Real(mu.m10 / mu.m00);
		new_detection(1) = Real(mu.m01 / mu.m00);

		new_detection(2) = 0.0;
		new_detection(3) = 0.0;

		new_detection(4) = Real(cv::contourArea(contours_[b]));

		cv::fitLine(contours_[b], fitted_line, cv::DIST_L2, 0, 0.01, 0.01);
		new_detection(5) = std::atan(fitted_line[1] / fitted_line[0]); // TODO: verify slope range

		min_rect = cv::minAreaRect(cv::Mat(contours_[b]));
		new_detection(6) = std::min(min_rect.size.width, min_rect.size.height);
		new_detection(7) = std::max(min_rect.size.width, min_rect.size.height);

		detections.push_back(new_detection);
		image_processing_output_file_ << b << " " << new_detection(0) << " " << new_detection(1) << " " << new_detection(2)
			<< " " << new_detection(3) << " " << new_detection(4) << " " << new_detection(5)
			<< " " << new_detection(6) << " " << new_detection(7) << " ";
	}
	image_processing_output_file_ << std::endl;
}

void ImageProcessingEngine::ApplyMorphologicalTransform(int morph_operator, int morph_element, int morph_size)
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

	cv::Mat element = cv::getStructuringElement(morph_element,
		cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
		cv::Point(morph_size, morph_size));
	cv::morphologyEx(threshold_image_.clone(), morphology_image_, morph_operator, element);
}

void ImageProcessingEngine::DetectEdges()
{
	//	int canny_low_threshold = 0;
	//	int canny_ratio = 2;
	//	int canny_high_threshold = canny_low_threshold * canny_ratio;
	//	int canny_kernel_size = 5; // 3,5,7 ?
	//	cv::Canny(blur_image_.clone(), edge_image_, canny_low_threshold, canny_high_threshold, canny_kernel_size);

	cv::Laplacian(blur_image_.clone(), edge_image_, CV_16S, 1, 1, 0, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(edge_image_, edge_image_);
}

void ImageProcessingEngine::FindImprovedContourCenters()
{
	int offset = 2;
	cv::Mat extended_contour_image;
	cv::copyMakeBorder(disconnected_image_,
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
		cv::RETR_TREE,
		cv::CHAIN_APPROX_SIMPLE,
		cv::Point(-offset, -offset));
	contours_.erase(std::remove_if(contours_.begin(),
		contours_.end(),
		[&](const std::vector<cv::Point> &contour)
	{
		double contour_area = cv::contourArea(contour);
		return contour_area < parameter_handler_.GetMinContourArea();
	}),
		contours_.end());

	mu_ = std::vector<cv::Moments>(contours_.size());
	for (int i = 0; i < contours_.size(); ++i)
	{
		mu_[i] = cv::moments(contours_[i], true);
	}

	cv::Mat contours_centers_image;
	cv::cvtColor(gray_image_, contours_centers_image, cv::COLOR_GRAY2BGR);
	for (int i = 0; i < contours_.size(); ++i)
	{
		cv::Scalar color = cv::Scalar(0, 127, 0);
		cv::Point2f center = cv::Point(mu_[i].m10 / mu_[i].m00, mu_[i].m01 / mu_[i].m00);
		cv::circle(contours_centers_image, center, 4, color, -1, 8);
		cv::putText(contours_centers_image, std::to_string(i), center, cv::FONT_HERSHEY_DUPLEX, 0.4, color);
		//
		//		cv::Vec4f fitted_line;
		//		cv::fitLine(contours_[i], fitted_line, cv::DIST_L2, 0, 0.01, 0.01);
		//		cv::Point2f line_direction(std::cos(std::atan(fitted_line(1) / fitted_line(0))), std::sin(std::atan(fitted_line(1) / fitted_line(0))));
		//		cv::line(contours_centers_image, center - 10.0f * line_direction, center + 10.0f * line_direction, color);
	}
	contour_image_ = contours_centers_image;
}

void ImageProcessingEngine::AnalyzeConvexityDefects()
{
	convexity_defects_image_ = disconnected_image_.clone();

	for (int i = 0; i < contours_.size(); ++i)
	{
		const std::vector<cv::Point> &contour = contours_[i];
		if (contour.size() > 2) // convex hull may be computed for contours with number of points > 2
		{
			std::vector<int> convex_hull_indices;
			std::vector<cv::Point> convex_hull;
			cv::convexHull(contour, convex_hull_indices, false, false);
			cv::convexHull(contour, convex_hull, false, true);
			if (cv::contourArea(convex_hull) / cv::contourArea(contour) > parameter_handler_.GetAreaIncrease())
			{
				std::vector<cv::Vec4i> convexity_defects;
				cv::convexityDefects(contour, convex_hull_indices, convexity_defects);
				std::vector<cv::Vec4i>::iterator max_iter = std::max_element(convexity_defects.begin(),
					convexity_defects.end(),
					[](const cv::Vec4i &a, const cv::Vec4i &b)
				{
					return a[3] < b[3];
				});
				if ((*max_iter)[3] / 256.0 > parameter_handler_.GetConvexityDefectMagnitude())
				{
					//					cv::Rect bounding_rect = cv::boundingRect(contour);
					//					cv::Mat bounding_image = blur_image_(bounding_rect).clone();
					//					cv::Mat mask = cv::Mat::zeros(bounding_image.size(), bounding_image.type());
					//					cv::drawContours(mask, contours_, i, cv::Scalar(255, 255, 255), -1, cv::LINE_AA, NULL, 0, cv::Point(0, 0));
					//					bounding_image.clone().copyTo(bounding_image, mask);
					//					cv::threshold(bounding_image, bounding_image, 127, 255, 0); // 127 -> put actual threshold
					////					cv::Mat element = cv::getStructuringElement(0, cv::Size(2 * 1 + 1, 2 * 1 + 1), cv::Point(1, 1));
					////					cv::morphologyEx(bounding_image, bounding_image, 2, element);
					//
					//					int offset = 2;
					//					cv::Mat extended_bounding_image;
					//					cv::copyMakeBorder(bounding_image, extended_bounding_image, offset, offset, offset, offset, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
					//					std::vector<std::vector<cv::Point>> bounding_contours;
					//					std::vector<cv::Vec4i> bounding_hierarchy;
					//					cv::findContours(extended_bounding_image, bounding_contours, bounding_hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(-offset, -offset));
					//
					//					if (bounding_contours.size() > 1)
					{
						const cv::Point2f &A = contour[(*max_iter)[0]];
						const cv::Point2f &B = contour[(*max_iter)[1]];
						const cv::Point2f &C = contour[(*max_iter)[2]];
						cv::Point2f D;
						if (A.x == B.x)
						{
							D.x = A.x;
							D.y = C.y;
						}
						else if (A.y == B.y)
						{
							D.x = C.x;
							D.y = A.y;
						}
						else
						{
							float m_AB = (A.y - B.y) / (A.x - B.x);
							float b_AB = -((A.y - B.y) * B.x - B.y * (A.x - B.x)) / (A.x - B.x);
							float m_CD = -1.0 / m_AB;
							float b_CD = C.y - m_CD * C.x;
							D.x = (b_CD - b_AB) / (m_AB - m_CD);
							D.y = m_CD * D.x + b_CD;
						}

						float increment = 1.0 / cv::norm(C - D);
						cv::Point2f point_along_defect(C + (C - D) * increment);
						while (cv::pointPolygonTest(contour, point_along_defect, false) >= 0.0)
						{
							point_along_defect.x += (C.x - D.x) * increment;
							point_along_defect.y += (C.y - D.y) * increment;
						}
						//						point_along_defect += 1.0 * (C - D) * increment;
						cv::Scalar color = cv::Scalar(0, 0, 0);
						cv::line(convexity_defects_image_,
							cv::Point2f(C.x - (C.x - D.x) * increment, C.y - (C.y - D.y) * increment),
							point_along_defect,
							color,
							4,
							cv::LINE_8);
					}
				}
			}
		}
	}
}

void ImageProcessingEngine::AnalyzeConvexityDefectsBasedOnCrossSection()
{
	convexity_defects_image_ = disconnected_image_.clone();

	for (int i = 0; i < contours_.size(); ++i)
	{
		const std::vector<cv::Point> &contour = contours_[i];
		if (contour.size() > 2) // convex hull may be computed for contours with number of points > 2
		{
			std::vector<int> convex_hull_indices;
			std::vector<cv::Point> convex_hull;
			cv::convexHull(contour, convex_hull_indices, false, false);
			cv::convexHull(contour, convex_hull, false, true);
			if (cv::contourArea(convex_hull) / cv::contourArea(contour) > parameter_handler_.GetAreaIncrease())
			{
				std::vector<cv::Vec4i> convexity_defects;
				cv::convexityDefects(contour, convex_hull_indices, convexity_defects);
				std::vector<cv::Vec4i>::iterator max_iter = std::max_element(convexity_defects.begin(),
					convexity_defects.end(),
					[](const cv::Vec4i &a, const cv::Vec4i &b)
				{
					return a[3] < b[3];
				});
				if ((*max_iter)[3] / 256.0 > parameter_handler_.GetConvexityDefectMagnitude())
				{
					const cv::Point2f &A = contour[(*max_iter)[0]];
					const cv::Point2f &B = contour[(*max_iter)[1]];
					const cv::Point2f &C = contour[(*max_iter)[2]];
					cv::Point2f D;
					if (A.x == B.x)
					{
						D.x = A.x;
						D.y = C.y;
					}
					else if (A.y == B.y)
					{
						D.x = C.x;
						D.y = A.y;
					}
					else
					{
						float m_AB = (A.y - B.y) / (A.x - B.x);
						float b_AB = -((A.y - B.y) * B.x - B.y * (A.x - B.x)) / (A.x - B.x);
						float m_CD = -1.0 / m_AB;
						float b_CD = C.y - m_CD * C.x;
						D.x = (b_CD - b_AB) / (m_AB - m_CD);
						D.y = m_CD * D.x + b_CD;
					}
					float increment = 1.0 / cv::norm(C - D);
					cv::Point2f E(C + (C - D) * increment);
					while (cv::pointPolygonTest(contour, E, false) >= 0.0)
					{
						E += (C - D) * increment;
					}

					cv::Point2f D1(D + (B - D) / cv::norm(B - D) * 5.0);
					cv::Point2f C1(D1);
					while (cv::pointPolygonTest(contour, C1, false) <= 0.0)
					{
						C1 += (C - D) * increment;
					}
					cv::Point2f E1(C1 + (C - D) * increment);
					while (cv::pointPolygonTest(contour, E1, false) >= 0.0)
					{
						E1 += (C - D) * increment;
					}

					cv::Point2f D2(D - (B - D) / cv::norm(B - D) * 5.0);
					cv::Point2f C2(D2);
					while (cv::pointPolygonTest(contour, C2, false) <= 0.0)
					{
						C2 += (C - D) * increment;
					}
					cv::Point2f E2(C2 + (C - D) * increment);
					while (cv::pointPolygonTest(contour, E2, false) >= 0.0)
					{
						E2 += (C - D) * increment;
					}

					if ((cv::norm(E - C) - cv::norm(E1 - C1) > 3.0) && (cv::norm(E - C) - cv::norm(E2 - C2) > 3.0))
					{
						cv::Scalar color = cv::Scalar(0, 0, 0);
						cv::line(convexity_defects_image_,
							cv::Point2f(C.x - (C.x - D.x) * 2.0 * increment, C.y - (C.y - D.y) * 2.0 * increment),
							cv::Point2f(E.x + (C.x - D.x) * 2.0 * increment, E.y + (C.y - D.y) * 2.0 * increment),
							color,
							2,
							cv::LINE_AA);
					}
				}
			}
		}
	}
}

void ImageProcessingEngine::DisconnectBacteria()
{
	cv::Mat I = threshold_image_;
	cv::Mat J = disconnected_image_ = threshold_image_.clone();
	CV_Assert(I.depth() == CV_8U);

	int channels = I.channels();
	int n_rows = I.rows;
	int n_cols = I.cols * channels;

	if (I.isContinuous())
	{
		const uchar *ptr = I.ptr<uchar>(0);
		uchar *out_ptr = J.ptr<uchar>(0);
		for (int j = 1; j < n_rows - 1; ++j)
		{
			for (int i = 1; i < n_cols - 1; ++i)
			{
				if (ptr[n_cols * j + i] == 255)
				{
					if ((ptr[n_cols * (j - 1) + i] == 0 && ptr[n_cols * (j + 1) + i] == 0) ||
						(ptr[n_cols * j + i - 1] == 0 && ptr[n_cols * j + i + 1] == 0) ||
						(ptr[n_cols * (j - 1) + i - 1] == 0 && ptr[n_cols * (j + 1) + i + 1] == 0) ||
						(ptr[n_cols * (j + 1) + i - 1] == 0 && ptr[n_cols * (j - 1) + i + 1] == 0))
					{
						out_ptr[n_cols * j + i] = 0;
					}
				}
			}
		}
	}
	else
	{
		for (int j = 1; j < n_rows - 1; ++j)
		{
			const uchar *prev_row = I.ptr<uchar>(j - 1);
			const uchar *curr_row = I.ptr<uchar>(j);
			const uchar *next_row = I.ptr<uchar>(j + 1);

			uchar *output_row = J.ptr<uchar>(j);

			for (int i = 1; i < n_cols - 1; ++i)
			{
				if (curr_row[i] == 255)
				{
					if ((prev_row[i] == 0 && next_row[i] == 0) ||
						(curr_row[i - 1] == 0 && curr_row[i + 1] == 0) ||
						(prev_row[i - 1] == 0 && next_row[i + 1] == 0) ||
						(next_row[i - 1] == 0 && prev_row[i + 1] == 0))
					{
						output_row[i] = 0;
					}
				}
			}
		}
	}
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