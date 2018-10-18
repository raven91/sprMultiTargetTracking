//
// Created by Nikita Kruk on 27.11.17.
//

#include "KalmanFilterExperimental.hpp"
#include "HungarianAlgorithm.hpp"

#include <iostream>
#include <sstream>
#include <algorithm> // std::copy, std::max, std::set_difference, std::for_each, std::sort
#include <iterator>  // std::back_inserter, std::prev
#include <cmath>
#include <set>
#include <numeric>   // std::iota
#include <random>
#include <iomanip>   // std::setfill, std::setw
#include <complex>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <eigen3/Eigen/Eigenvalues>

KalmanFilterExperimental::KalmanFilterExperimental(ParameterHandlerExperimental &parameter_handler,
                                                   ImageProcessingEngine &image_processing_engine) :
    parameter_handler_(parameter_handler),
    image_processing_engine_(image_processing_engine),
    costs_order_of_magnitude_(1000.0),
    unmatched_(),
    max_prediction_time_(0),
    max_target_index_(0),
    targets_colors_(),
    rng_(12345)
{
  I_ = Eigen::MatrixXd::Identity(kNumOfStateVars, kNumOfStateVars);
  A_ = Eigen::MatrixXd::Zero(kNumOfStateVars, kNumOfStateVars);
  W_ = Eigen::MatrixXd::Zero(kNumOfStateVars, kNumOfStateVars);
  H_ = Eigen::MatrixXd::Zero(kNumOfDetectionVars, kNumOfStateVars);
  Q_ = Eigen::MatrixXd::Zero(kNumOfDetectionVars, kNumOfDetectionVars);
  P_ = Eigen::MatrixXd::Zero(kNumOfStateVars, kNumOfStateVars);
  K_ = Eigen::MatrixXd::Zero(kNumOfStateVars, kNumOfStateVars);

  Real dt = 1;// in ms==image
  A_(0, 0) = A_(1, 1) = A_(2, 2) = A_(3, 3) = 1.0;
  A_(0, 2) = A_(1, 3) = dt;
  H_(0, 0) = H_(1, 1) = 1.0;

  W_(0, 0) = W_(1, 1) = dt * dt * dt * dt / 4.0;
  W_(2, 2) = W_(3, 3) = dt * dt;
  W_(0, 2) = W_(1, 3) = W_(2, 0) = W_(3, 1) = dt * dt * dt / 2.0;
  W_ *= 2.5 * 2.5; // multiply by variance in acceleration
  Q_(0, 0) = Q_(1, 1) = 2.5 * 2.5;

  P_ = W_;
}

KalmanFilterExperimental::~KalmanFilterExperimental()
{
  CloseKalmanFilterOutputFiles();
}

void KalmanFilterExperimental::CreateKalmanFilterOutputFiles()
{
  std::string kalman_filter_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
          + parameter_handler_.GetKalmanFilterOutputFileName();
  kalman_filter_output_file_.open(kalman_filter_output_file_name, std::ios::out | std::ios::trunc);
  assert(kalman_filter_output_file_.is_open());

  std::string kalman_filter_matlab_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
          + parameter_handler_.GetKalmanFilterMatlabOutputFileName();
  kalman_filter_matlab_output_file_.open(kalman_filter_matlab_output_file_name, std::ios::out | std::ios::trunc);
  assert(kalman_filter_matlab_output_file_.is_open());
}

void KalmanFilterExperimental::CloseKalmanFilterOutputFiles()
{
  if (kalman_filter_output_file_.is_open())
  {
    kalman_filter_output_file_.close();
  }
  if (kalman_filter_matlab_output_file_.is_open())
  {
    kalman_filter_matlab_output_file_.close();
  }
}

void KalmanFilterExperimental::InitializeTargets(std::map<int, Eigen::VectorXd> &targets,
                                                 const std::vector<Eigen::VectorXd> &detections)
{
  targets.clear();

  int last_index = 0;
  Eigen::VectorXd new_target(kNumOfExtractedFeatures);
  for (const Eigen::VectorXd &detection : detections)
  {
    if (targets.empty())
    {
      last_index = -1;
    } else
    {
      last_index = std::prev(targets.end())->first;
//			last_index = targets.rbegin()->first;
    }
    new_target = detection;
    targets[++last_index] = new_target;
  }
  max_target_index_ = last_index;

  CorrectForOrientationUniqueness(targets);

  SaveTargets(kalman_filter_output_file_, parameter_handler_.GetFirstImage(), targets);
  SaveTargetsMatlab(kalman_filter_matlab_output_file_, parameter_handler_.GetFirstImage(), targets);
  SaveImagesWithVectors(parameter_handler_.GetFirstImage(), targets);
}

void KalmanFilterExperimental::InitializeTargets(std::map<int, Eigen::VectorXd> &targets, std::ifstream &file)
{
  int last_index = 0;
  Eigen::VectorXd new_target = Eigen::MatrixXd::Zero(kNumOfExtractedFeatures, 1);
  int time_idx = 0;
  int target_idx = 0;
  int number_of_detections = 0;

  do
  {
    targets.clear();
    file >> time_idx >> number_of_detections;
    for (int b = 0; b < number_of_detections; ++b)
    {
      if (targets.empty())
      {
        last_index = -1;
      } else
      {
        last_index = std::prev(targets.end())->first;
      }
      file >> target_idx
           >> new_target(0) >> new_target(1) >> new_target(2) >> new_target(3)
           >> new_target(4) >> new_target(5) >> new_target(6) >> new_target(7);
      targets[++last_index] = new_target;
    }
    max_target_index_ = last_index;
  } while (time_idx < parameter_handler_.GetFirstImage());

  CorrectForOrientationUniqueness(targets);

  SaveTargets(kalman_filter_output_file_, parameter_handler_.GetFirstImage(), targets);
  SaveTargetsMatlab(kalman_filter_matlab_output_file_, parameter_handler_.GetFirstImage(), targets);
//  SaveImagesWithVectors(parameter_handler_.GetFirstImage(), targets);
}

void KalmanFilterExperimental::ObtainNewDetections(std::vector<Eigen::VectorXd> &detections, std::ifstream &file)
{
  detections.clear();

  Eigen::VectorXd new_detection = Eigen::MatrixXd::Zero(kNumOfExtractedFeatures, 1);
  int time_idx = 0;
  int detection_idx = 0;
  int number_of_detections = 0;

  file >> time_idx >> number_of_detections;
  for (int b = 0; b < number_of_detections; ++b)
  {
    file >> detection_idx
         >> new_detection(0) >> new_detection(1) >> new_detection(2) >> new_detection(3)
         >> new_detection(4) >> new_detection(5) >> new_detection(6) >> new_detection(7);
    detections.push_back(new_detection);
  }
}

void KalmanFilterExperimental::PerformEstimation(int image_idx,
                                                 std::map<int, Eigen::VectorXd> &targets,
                                                 const std::vector<Eigen::VectorXd> &detections)
{
  std::cout << "kalman filter: image#" << image_idx << std::endl;

  int n_max_dim = 0; // max size between targets and detections

  ComputePriorEstimate(targets);
  ComputeKalmanGainMatrix();

  if (!(detections.empty()))
  {
    n_max_dim = (int) std::max(targets.size(), detections.size());
    std::vector<int> target_indexes;
    std::vector<int> assignments((unsigned long) n_max_dim, -1);
    std::vector<CostInt> costs((unsigned long) n_max_dim);

    PerformDataAssociation(targets, detections, n_max_dim, target_indexes, assignments, costs);
    UnassignUnrealisticTargets(targets, detections, n_max_dim, assignments, costs, target_indexes);
    ComputePosteriorEstimate(targets, detections, assignments, target_indexes);
    RemoveRecapturedTargetsFromUnmatched(targets, assignments, target_indexes);
    MarkLostTargetsAsUnmatched(targets, assignments, target_indexes);
    AddNewTargets(targets, detections, assignments);
  } else // detections.size() == 0
  {
    std::cout << "error in detector: no detections found" << std::endl;
    MarkAllTargetsAsUnmatched(targets);
  }
  // if the target has been lost for too long -> remove it
  DeleteLongLostTargets(targets);
  CorrectForOrientationUniqueness(targets);

  SaveTargets(kalman_filter_output_file_, image_idx, targets);
  SaveTargetsMatlab(kalman_filter_matlab_output_file_, image_idx, targets);
#if defined(PARTIAL_IMAGE_OUTPUT)
  if (image_idx < 10)
#endif
  SaveImagesWithVectors(image_idx, targets);

  std::cout << "number of overall targets taken part: " << max_target_index_ + 1 << "; number of current targets: "
            << targets.size() << std::endl;
}

void KalmanFilterExperimental::ComputePriorEstimate(std::map<int, Eigen::VectorXd> &targets)
{
  Eigen::VectorXd x_i_estimate(kNumOfStateVars);
  for (std::pair<const int, Eigen::VectorXd> &target : targets)
  {
    x_i_estimate = (target.second).head(kNumOfStateVars);
    x_i_estimate = A_ * x_i_estimate;
    (target.second).head(kNumOfStateVars) = x_i_estimate;
  }
  P_ = A_ * P_ * A_.transpose() + W_;
}

void KalmanFilterExperimental::ComputeKalmanGainMatrix()
{
  K_ = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + Q_).inverse();
}

void KalmanFilterExperimental::PerformDataAssociation(const std::map<int, Eigen::VectorXd> &targets,
                                                      const std::vector<Eigen::VectorXd> &detections,
                                                      int n_max_dim,
                                                      std::vector<int> &target_indexes,
                                                      std::vector<int> &assignments,
                                                      std::vector<CostInt> &costs)
{
  std::vector<std::vector<CostInt>>
      cost_matrix((unsigned long) n_max_dim, std::vector<CostInt>((unsigned long) n_max_dim, 0));
  CostInt max_cost = InitializeCostMatrix(targets, detections, cost_matrix, target_indexes);
  HungarianAlgorithm hungarian_algorithm(n_max_dim, cost_matrix);
  hungarian_algorithm.Start(assignments, costs);
  std::for_each(costs.begin(),
                costs.end(),
                [&](CostInt &c)
                {
                  c = CostInt((max_cost - c) / costs_order_of_magnitude_);
                });
}

CostInt KalmanFilterExperimental::InitializeCostMatrix(const std::map<int, Eigen::VectorXd> &targets,
                                                       const std::vector<Eigen::VectorXd> &detections,
                                                       std::vector<std::vector<CostInt>> &cost_matrix,
                                                       std::vector<int> &target_indexes)
{
  target_indexes.clear();

  Eigen::VectorXd target(kNumOfExtractedFeatures);
  Eigen::VectorXd detection(kNumOfExtractedFeatures);
  Real cost = 0.0;
  Real max_cost = 0;
  int row = 0;
  Real d_x = 0.0, d_y = 0.0;
  Real dist = 0.0;
  Real max_dist = std::sqrt(parameter_handler_.GetSubimageXSize() * parameter_handler_.GetSubimageXSize()
                                + parameter_handler_.GetSubimageYSize() * parameter_handler_.GetSubimageYSize());
//  Real area_increase = 0.0;
  // the trajectories are not guaranteed to have successive labels
  // row and column indexes are introduced to account for that
  for (std::map<int, Eigen::VectorXd>::const_iterator it = targets.begin(); it != targets.end(); ++it, ++row)
  {
    target_indexes.push_back(it->first);
    target = it->second;

    for (int column = 0; column < detections.size(); ++column)
    {
      detection = detections[column];
      d_x = (target(0) - detection(0));
      d_y = (target(1) - detection(1));
      dist = std::sqrt(d_x * d_x + d_y * d_y);
//      area_increase = std::max(target(4), detection(4)) / std::min(target(4), detection(4));
      // put only close assignment costs in the cost matrix
      if (dist <= parameter_handler_.GetDataAssociationCost())
      {
        cost = dist; // Euclidean norm from a target to a detection
      } else
      {
        cost = max_dist;
      }
//      cost = dist * area_increase;

      cost_matrix[row][column] = CostInt(cost * costs_order_of_magnitude_);
      if (max_cost < cost)
      {
        max_cost = cost;
      }
    }
  }

  // turn min cost problem into max cost problem
  for (int row = 0; row < targets.size(); ++row)
  {
    for (int column = 0; column < detections.size(); ++column)
    {
      cost_matrix[row][column] = CostInt(max_cost * costs_order_of_magnitude_) - cost_matrix[row][column];
    }
  }
  // the complementary values are left zero as needed for the max cost problem

  return CostInt(max_cost * costs_order_of_magnitude_);
}

void KalmanFilterExperimental::UnassignUnrealisticTargets(const std::map<int, Eigen::VectorXd> &targets,
                                                          const std::vector<Eigen::VectorXd> &detections,
                                                          int n_max_dim,
                                                          std::vector<int> &assignments,
                                                          std::vector<CostInt> &costs,
                                                          const std::vector<int> &target_indexes)
{
  for (int i = 0; i < targets.size(); ++i)
  {
    if (assignments[i] >= detections.size()) // if the assignment is into an imaginary detection
    {
      assignments[i] = -1;
    } else // if a cost is too high
    {
      Eigen::VectorXd target = targets.at(target_indexes[i]);
      Eigen::VectorXd detection = detections[assignments[i]];
      Real d_x = (target(0) - detection(0));
      Real d_y = (target(1) - detection(1));
      Real dist = std::sqrt(d_x * d_x + d_y * d_y);
//      Real area_increase = std::max(target(4), detection(4)) / std::min(target(4), detection(4));

      if ((dist > parameter_handler_.GetDataAssociationCost()))
//          || (area_increase > 1.5)) // in pixels
      {
        assignments[i] = -1;
      }
    }
  }
  // if the assignment is from an imaginary target
  for (int i = (int) targets.size(); i < n_max_dim; ++i)
  {
    assignments[i] = -1;
  }
}

void KalmanFilterExperimental::ComputePosteriorEstimate(std::map<int, Eigen::VectorXd> &targets,
                                                        const std::vector<Eigen::VectorXd> &detections,
                                                        const std::vector<int> &assignments,
                                                        const std::vector<int> &target_indexes)
{
  Eigen::VectorXd z_i(kNumOfDetectionVars);
  Eigen::VectorXd x_i_estimate(kNumOfStateVars);
  for (int i = 0; i < targets.size(); ++i)
  {
    if (assignments[i] != -1)
    {
      x_i_estimate = targets[target_indexes[i]].head(kNumOfStateVars);
      z_i = detections[assignments[i]].head(2);
      x_i_estimate = x_i_estimate + K_ * (z_i - H_ * x_i_estimate);
      targets[target_indexes[i]].head(kNumOfStateVars) = x_i_estimate;

      targets[target_indexes[i]][4] = detections[assignments[i]][4];
      targets[target_indexes[i]][5] = detections[assignments[i]][5];
      targets[target_indexes[i]][6] = detections[assignments[i]][6];
      targets[target_indexes[i]][7] = detections[assignments[i]][7];
    }
  }
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(kNumOfStateVars, kNumOfStateVars);
  P_ = (I - K_ * H_) * P_;
}

void KalmanFilterExperimental::MarkLostTargetsAsUnmatched(std::map<int, Eigen::VectorXd> &targets,
                                                          const std::vector<int> &assignments,
                                                          const std::vector<int> &target_indexes)
{
  // consider only the initial targets without appended undetected ones
  // and without appended artificial elements
  for (int i = 0; i < targets.size(); ++i)
  {
    if (assignments[i] == -1)
    {
      if (unmatched_.find(target_indexes[i]) != unmatched_.end())
      {
        ++unmatched_[target_indexes[i]];
      } else
      {
        unmatched_[target_indexes[i]] = 1;
      }
    }
  }
}

void KalmanFilterExperimental::RemoveRecapturedTargetsFromUnmatched(std::map<int, Eigen::VectorXd> &targets,
                                                                    const std::vector<int> &assignments,
                                                                    const std::vector<int> &target_indexes)
{
  for (int i = 0; i < targets.size(); ++i)
  {
    if (assignments[i] != -1)
    {
      if (unmatched_.find(target_indexes[i]) != unmatched_.end())
      {
        unmatched_.erase(target_indexes[i]); // stop suspecting a target if it has been recovered
      }
    }
  }
}

void KalmanFilterExperimental::MarkAllTargetsAsUnmatched(std::map<int, Eigen::VectorXd> &targets)
{
  // all the targets have been lost
  for (std::map<int, Eigen::VectorXd>::const_iterator it = targets.begin(); it != targets.end(); ++it)
  {
    if (unmatched_.find(it->first) != unmatched_.end())
    {
      ++unmatched_[it->first];
    } else
    {
      unmatched_[it->first] = 1;
    }
  }
}

void KalmanFilterExperimental::AddNewTargets(std::map<int, Eigen::VectorXd> &targets,
                                             const std::vector<Eigen::VectorXd> &detections,
                                             const std::vector<int> &assignments)
{
  std::vector<int> all_detection_indexes(detections.size());
  std::iota(all_detection_indexes.begin(),
            all_detection_indexes.end(),
            0); // construct detection indexes from 0 through d.size()-1
  std::vector<int> sorted_assignments(assignments.begin(), assignments.end());
  std::sort(sorted_assignments.begin(), sorted_assignments.end());
  std::vector<int> indexes_to_unassigned_detections;
  std::set_difference(all_detection_indexes.begin(),
                      all_detection_indexes.end(),
                      sorted_assignments.begin(),
                      sorted_assignments.end(),
                      std::back_inserter(indexes_to_unassigned_detections)); // set_difference requires pre-sorted containers
  for (int i = 0; i < indexes_to_unassigned_detections.size(); ++i)
  {
    targets[max_target_index_ + 1] = detections[indexes_to_unassigned_detections[i]];
    ++max_target_index_;
  }
}

void KalmanFilterExperimental::DeleteLongLostTargets(std::map<int, Eigen::VectorXd> &targets)
{
  for (std::map<int, int>::iterator it = unmatched_.begin(); it != unmatched_.end();)
  {
    if (it->second > max_prediction_time_)
    {
      targets.erase(it->first);
      it = unmatched_.erase(it);
    } else
    {
      ++it;
    }
  }
}

void KalmanFilterExperimental::CorrectForOrientationUniqueness(std::map<int, Eigen::VectorXd> &targets)
{
  Eigen::VectorXd x_i(kNumOfExtractedFeatures);
  Eigen::Vector2d velocity_i;
  Eigen::Vector2d orientation_i;
  for (std::pair<const int, Eigen::VectorXd> &target : targets)
  {
    x_i = target.second;
    velocity_i = x_i.segment(2, 2);
    orientation_i << std::cosf(x_i[5]), std::sinf(x_i[5]);

    // in order to determine the orientation vector uniquely,
    // we assume the angle difference between the orientation and velocity is < |\pi/2|
    if (velocity_i.dot(orientation_i) < 0.0)
    {
      (target.second)[5] = WrappingModulo(x_i[5] + M_PI, 2.0 * M_PI);
    } else
    {
      (target.second)[5] = WrappingModulo(x_i[5], 2.0 * M_PI);
    }
    (target.second)[5] = ConstrainAngleCentered((target.second)[5]);
  }
}

void KalmanFilterExperimental::SaveTargets(std::ofstream &file,
                                           int image_idx,
                                           const std::map<int, Eigen::VectorXd> &targets)
{
  Eigen::VectorXd x_i;
  file << image_idx << " " << targets.size() << " ";
  for (const std::pair<const int, Eigen::VectorXd> &target : targets)
  {
    x_i = target.second;
    file << target.first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3) << " " << x_i(4) << " "
         << x_i(5) << " " << x_i(6) << " " << x_i(7) << " ";
  }
  file << std::endl;
}

void KalmanFilterExperimental::SaveTargetsMatlab(std::ofstream &file,
                                                 int image_idx,
                                                 const std::map<int, Eigen::VectorXd> &targets)
{
  Eigen::VectorXd x_i;
  for (const std::pair<const int, Eigen::VectorXd> &target : targets)
  {
    x_i = target.second;
    file << image_idx << " " << target.first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3) << " "
         << x_i(4) << " " << x_i(5) << " " << x_i(6) << " " << x_i(7) << std::endl;
  }
}

void KalmanFilterExperimental::SaveImagesWithVectors(int image_idx, const std::map<int, Eigen::VectorXd> &targets)
{
  cv::Mat image;
  image = image_processing_engine_.GetSourceImage();

  cv::Scalar text_color(255, 127, 0);
  cv::Scalar velocity_color(0, 255, 0);
  cv::Scalar orientation_color(255, 0, 0);

  for (const std::pair<const int, Eigen::VectorXd> &target : targets)
  {
    Eigen::VectorXd x_i = target.second;
    cv::Point2f center = cv::Point2f(x_i(0), x_i(1));
//    cv::circle(image, center, 3, text_color, -1, 8);
    cv::putText(image, std::to_string(target.first), center, cv::FONT_HERSHEY_DUPLEX, 0.4, text_color);
//		cv::Point2f pt = cv::Point2f(std::cosf(x_i(5)), std::sinf(x_i(5)));
    Real length = std::max(x_i(6), x_i(7));
    cv::line(image,
             center,
             center + cv::Point2f(x_i(2), x_i(3)),
             velocity_color);
    cv::line(image,
             center,
             center + cv::Point2f(std::cosf(x_i(5)), std::sinf(x_i(5))) * length / 2.0,
             orientation_color);
//		std::cout << "(" << center.x << "," << center.y << ") -> (" << center.x + std::cosf(x_i(5)) * x_i(4) / 10.0f << "," << center.y + std::sinf(x_i(5)) * x_i(4) / 10.0f << ")" << std::endl;

/*    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, kNumOfStateVars, kNumOfStateVars>> s(P_);
    Eigen::Matrix<std::complex<float>, kNumOfStateVars, 1> eigenvalues = s.eigenvalues();
    Eigen::Matrix<std::complex<float>, kNumOfStateVars, kNumOfStateVars> eigenvectors = s.eigenvectors();
    float angle = std::atan2(std::real(eigenvectors(1, 0)), std::real(eigenvectors(0, 0))) * 180 / M_PI;
    cv::ellipse(image,
                center,
                cv::Size(3 * std::real(eigenvalues(0)), 3 * std::real(eigenvalues(1))),
                angle,
                0,
                360,
                cv::Scalar(0, 0, 255));
*/
  }

  std::ostringstream output_image_name_buf;
  output_image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetKalmanFilterSubfolder()
                        << parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << image_idx
                        << parameter_handler_.GetFileName1();
  std::string output_image_name = output_image_name_buf.str();
  cv::imwrite(output_image_name, image);
}

void KalmanFilterExperimental::SaveImagesWithRectangles(int image_idx, const std::map<int, Eigen::VectorXd> &targets)
{
  cv::Mat image;
  image_processing_engine_.ComposeImageForFilterOutput(image_idx, image);

  for (const std::pair<const int, Eigen::VectorXd> &target : targets)
  {
    cv::Scalar color;
    if (targets_colors_.find(target.first) != targets_colors_.end())
    {
      color = targets_colors_[target.first];
    } else
    {
      color = cv::Scalar(rng_.uniform(0, 255), rng_.uniform(0, 255), rng_.uniform(0, 255));
      targets_colors_[target.first] = color;
    }

    cv::Point2f center(target.second(0), target.second(1));
    Real length = std::max(target.second(6), target.second(7));
    Real width = std::min(target.second(6), target.second(7));
    cv::Point2f lengthwise_vec = cv::Point2f(std::cosf(target.second(5)), std::sinf(target.second(5)));
    cv::Point2f widthwise_vec = RotatePoint(lengthwise_vec, float(M_PI / 2.0));
    lengthwise_vec *= length / 2.0;
    widthwise_vec *= width / 2.0;
    cv::line(image, center + lengthwise_vec + widthwise_vec, center + lengthwise_vec - widthwise_vec, color, 2, 8);
    cv::line(image, center + lengthwise_vec - widthwise_vec, center - lengthwise_vec - widthwise_vec, color, 2, 8);
    cv::line(image, center - lengthwise_vec - widthwise_vec, center - lengthwise_vec + widthwise_vec, color, 2, 8);
    cv::line(image, center - lengthwise_vec + widthwise_vec, center + lengthwise_vec + widthwise_vec, color, 2, 8);
  }

  std::ostringstream output_image_name_buf;
  output_image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetKalmanFilterSubfolder()
                        << parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << image_idx
                        << parameter_handler_.GetFileName1();
  std::string output_image_name = output_image_name_buf.str();
  cv::imwrite(output_image_name, image);
}

cv::Point2f KalmanFilterExperimental::RotatePoint(const cv::Point2f &p, float rad)
{
  const float x = std::cos(rad) * p.x - std::sin(rad) * p.y;
  const float y = std::sin(rad) * p.x + std::cos(rad) * p.y;

  const cv::Point2f rot_p(x, y);
  return rot_p;
}