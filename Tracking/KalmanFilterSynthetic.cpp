//
// Created by Nikita Kruk on 03.07.18.
//

#include "KalmanFilterSynthetic.hpp"
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
#include <fstream>

KalmanFilterSynthetic::KalmanFilterSynthetic(ParameterHandlerSynthetic &parameter_handler,
                                             PeriodicBoundaryConditionsConfiguration &pbc_config) :
    parameter_handler_(parameter_handler),
    pbc_config_(pbc_config),
    tracking_statistics_(),
    costs_order_of_magnitude_(1000.0),
    unmatched_(),
    max_prediction_time_(5),
    max_target_index_(0)
{
  std::ostringstream kalman_filter_output_file_name_buffer;
  kalman_filter_output_file_name_buffer << parameter_handler_.GetTrackingFolder()
                                        << parameter_handler_.GetNumberOfOriginalTargets() << "/"
                                        << parameter_handler_.GetKalmanFilterOutputFileName() << "_phi_"
                                        << parameter_handler_.GetSimulationParameter("phi") << "_a_"
                                        << parameter_handler_.GetSimulationParameter("a") << "_U0_"
                                        << parameter_handler_.GetSimulationParameter("U_0") << "_k_"
                                        << parameter_handler_.GetSimulationParameter("kappa") << "_pom_"
                                        << parameter_handler_.GetPercentageOfMisdetections() << ".txt";
  kalman_filter_output_file_.open(kalman_filter_output_file_name_buffer.str(), std::ios::out | std::ios::trunc);
  assert(kalman_filter_output_file_.is_open());

  std::ostringstream kalman_filter_matlab_output_file_name_buffer;
  kalman_filter_matlab_output_file_name_buffer << parameter_handler_.GetTrackingFolder()
                                               << parameter_handler_.GetNumberOfOriginalTargets() << "/"
                                               << parameter_handler_.GetKalmanFilterMatlabOutputFileName() << "_phi_"
                                               << parameter_handler_.GetSimulationParameter("phi") << "_a_"
                                               << parameter_handler_.GetSimulationParameter("a") << "_U0_"
                                               << parameter_handler_.GetSimulationParameter("U_0") << "_k_"
                                               << parameter_handler_.GetSimulationParameter("kappa") << "_pom_"
                                               << parameter_handler_.GetPercentageOfMisdetections() << ".txt";
  kalman_filter_matlab_output_file_.open(kalman_filter_matlab_output_file_name_buffer.str(),
                                         std::ios::out | std::ios::trunc);
  assert(kalman_filter_matlab_output_file_.is_open());
}

KalmanFilterSynthetic::~KalmanFilterSynthetic()
{
  WriteTrackingStatisticsIntoFile();
  kalman_filter_output_file_.close();
  kalman_filter_matlab_output_file_.close();
}

void KalmanFilterSynthetic::InitializeTargets(std::map<int, Eigen::VectorXf> &targets, std::ifstream &file)
{
  int last_index = 0;
  Eigen::VectorXf new_target = Eigen::MatrixXf::Zero(kNumOfExtractedFeatures, 1);
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
      file >> target_idx >> new_target(0) >> new_target(1) >> new_target(2) >> new_target(3);
      targets[++last_index] = new_target;
    }
    max_target_index_ = last_index;
  } while (time_idx < parameter_handler_.GetFirstImage());

  SaveTargets(kalman_filter_output_file_, parameter_handler_.GetFirstImage(), targets);
  SaveTargetsMatlab(kalman_filter_matlab_output_file_, parameter_handler_.GetFirstImage(), targets);
  tracking_statistics_.IncrementNumberOfTimePoints(1);
}

void KalmanFilterSynthetic::ObtainNewDetections(std::vector<Eigen::VectorXf> &detections, std::ifstream &file)
{
  detections.clear();

  Eigen::VectorXf new_detection = Eigen::MatrixXf::Zero(kNumOfExtractedFeatures, 1);
  int time_idx = 0;
  int detection_idx = 0;
  int number_of_detections = 0;

  file >> time_idx >> number_of_detections;
//  if (time_idx >= 425)
//  {
//    std::cout << "debug" << std::endl;
//  }
  for (int b = 0; b < number_of_detections; ++b)
  {
    file >> detection_idx >> new_detection(0) >> new_detection(1) >> new_detection(2) >> new_detection(3);
    detections.push_back(new_detection);
//    if (detection_idx >= 40)
//    {
//      std::cout << "debug" << std::endl;
//    }
  }
}

void KalmanFilterSynthetic::PerformEstimation(int image_idx,
                                              std::map<int, Eigen::VectorXf> &targets,
                                              const std::vector<Eigen::VectorXf> &detections)
{
  std::cout << "kalman filter: image#" << image_idx << std::endl;

  int n_max_dim = 0; // max size between targets and detections
  Real dt = 1;// in ms

  Eigen::MatrixXf A = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
  Eigen::MatrixXf W = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
  Eigen::MatrixXf H = Eigen::MatrixXf::Zero(kNumOfDetectionVars, kNumOfStateVars);
  Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(kNumOfDetectionVars, kNumOfDetectionVars);
  Eigen::MatrixXf P_estimate = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
  Eigen::MatrixXf K = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);

  A(0, 0) = A(1, 1) = A(2, 2) = A(3, 3) = 1.0;
  A(0, 2) = A(1, 3) = dt;
  H(0, 0) = H(1, 1) = 1.0;

  W(0, 0) = W(1, 1) = dt * dt * dt * dt / 4.0f;
  W(2, 2) = W(3, 3) = dt * dt;
  W(0, 2) = W(1, 3) = W(2, 0) = W(3, 1) = dt * dt * dt / 2.0f;
  Q(0, 0) = Q(1, 1) = dt;

  P_estimate = W;
  ComputePriorEstimate(targets, P_estimate, A, W, H);
  ComputeKalmanGainMatrix(K, P_estimate, H, Q);

  if (detections.size() > 0)
  {
    n_max_dim = (int) std::max(targets.size(), detections.size());
    std::vector<int> target_indexes;
    std::vector<std::vector<CostInt>> cost_matrix(n_max_dim, std::vector<CostInt>(n_max_dim, 0));
    std::vector<int> assignments(n_max_dim, -1);
    std::vector<CostInt> costs(n_max_dim);

    PerformDataAssociation(targets, detections, n_max_dim, target_indexes, cost_matrix, assignments, costs);
    UnassignUnrealisticTargets(targets, detections, n_max_dim, assignments, costs);
    ComputePosteriorEstimate(targets, detections, P_estimate, K, H, assignments, target_indexes);
    RemoveRecapturedTargetsFromStrikes(targets, assignments, target_indexes);
    AddNewTargets(targets, detections, assignments);
    MarkLostTargetsAsUnmatched(targets, assignments, target_indexes);
  } else // detections.size() == 0
  {
    MarkAllTargetsAsUnmatched(targets);
  }
  // if the target has been lost for too long -> remove it
  DeleteLongLostTargets(targets);

  SaveTargets(kalman_filter_output_file_, image_idx, targets);
  SaveTargetsMatlab(kalman_filter_matlab_output_file_, image_idx, targets);

  tracking_statistics_.IncrementNumberOfTimePoints(1);
  std::cout << "number of overall targets taken part: " << max_target_index_ + 1 << "; number of current targets: "
            << targets.size() << std::endl;
}

void KalmanFilterSynthetic::ComputePriorEstimate(std::map<int, Eigen::VectorXf> &targets,
                                                 Eigen::MatrixXf &P_estimate,
                                                 const Eigen::MatrixXf &A,
                                                 const Eigen::MatrixXf &W,
                                                 const Eigen::MatrixXf &H)
{
  Eigen::VectorXf x_i_estimate(kNumOfStateVars);
  for (std::map<int, Eigen::VectorXf>::iterator it = targets.begin(); it != targets.end(); ++it)
  {
    x_i_estimate = (it->second).head(kNumOfStateVars);
    x_i_estimate = A * x_i_estimate;
    pbc_config_.ApplyPeriodicBoundaryConditions(x_i_estimate(0), x_i_estimate(1), x_i_estimate(0), x_i_estimate(1));
    (it->second).head(kNumOfStateVars) = x_i_estimate;
  }
  P_estimate = A * P_estimate * A.transpose() + W;
}

void KalmanFilterSynthetic::ComputeKalmanGainMatrix(Eigen::MatrixXf &K,
                                                    const Eigen::MatrixXf &P_estimate,
                                                    const Eigen::MatrixXf &H,
                                                    const Eigen::MatrixXf &Q)
{
  K = P_estimate * H.transpose() * (H * P_estimate * H.transpose() + Q).inverse();
}

void KalmanFilterSynthetic::PerformDataAssociation(const std::map<int, Eigen::VectorXf> &targets,
                                                   const std::vector<Eigen::VectorXf> &detections,
                                                   int n_max_dim,
                                                   std::vector<int> &target_indexes,
                                                   std::vector<std::vector<CostInt>> &cost_matrix,
                                                   std::vector<int> &assignments,
                                                   std::vector<CostInt> &costs)
{
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

void KalmanFilterSynthetic::UnassignUnrealisticTargets(const std::map<int, Eigen::VectorXf> &targets,
                                                       const std::vector<Eigen::VectorXf> &detections,
                                                       int n_max_dim,
                                                       std::vector<int> &assignments,
                                                       std::vector<CostInt> &costs)
{
  // if a cost is too high or if the assignment is into an imaginary detection
  for (int i = 0; i < targets.size(); ++i)
  {
    if (costs[i] > parameter_handler_.GetDataAssociationCost() || assignments[i] >= detections.size()) // in pixels
    {
      assignments[i] = -1;
    }
  }
  // if the assignment is from an imaginary target
  for (int i = (int) targets.size(); i < n_max_dim; ++i)
  {
    assignments[i] = -1;
  }
}

void KalmanFilterSynthetic::ComputePosteriorEstimate(std::map<int, Eigen::VectorXf> &targets,
                                                     const std::vector<Eigen::VectorXf> &detections,
                                                     Eigen::MatrixXf &P_estimate,
                                                     const Eigen::MatrixXf &K,
                                                     const Eigen::MatrixXf &H,
                                                     const std::vector<int> &assignments,
                                                     const std::vector<int> &target_indexes)
{
  Eigen::VectorXf z_i(kNumOfDetectionVars);
  Eigen::VectorXf x_i_estimate(kNumOfStateVars);
  for (int i = 0; i < targets.size(); ++i)
  {
    if (assignments[i] != -1)
    {
      x_i_estimate = targets[target_indexes[i]].head(kNumOfStateVars);
      z_i = detections[assignments[i]].head(2);
      if (x_i_estimate(0) - z_i(0) > parameter_handler_.GetSubimageXSize() / 2)
      {
        z_i(0) += parameter_handler_.GetSubimageXSize();
      } else if (x_i_estimate(0) - z_i(0) < -parameter_handler_.GetSubimageXSize() / 2)
      {
        z_i(0) -= parameter_handler_.GetSubimageXSize();
      }
      if (x_i_estimate(1) - z_i(1) > parameter_handler_.GetSubimageYSize() / 2)
      {
        z_i(1) += parameter_handler_.GetSubimageYSize();
      } else if (x_i_estimate(1) - z_i(1) < -parameter_handler_.GetSubimageYSize() / 2)
      {
        z_i(1) -= parameter_handler_.GetSubimageYSize();
      }
      x_i_estimate = x_i_estimate + K * (z_i - H * x_i_estimate);
      pbc_config_.ApplyPeriodicBoundaryConditions(x_i_estimate(0), x_i_estimate(1), x_i_estimate(0), x_i_estimate(1));
      targets[target_indexes[i]].head(kNumOfStateVars) = x_i_estimate;

      // zero for SPR system
//        targets[target_indexes[i]][4] = detections[assignments[i]][4];
//        targets[target_indexes[i]][5] = detections[assignments[i]][5];
//        targets[target_indexes[i]][6] = detections[assignments[i]][6];
//        targets[target_indexes[i]][7] = detections[assignments[i]][7];
    }
  }
  Eigen::MatrixXf I = Eigen::MatrixXf::Identity(kNumOfStateVars, kNumOfStateVars);
  P_estimate = (I - K * H) * P_estimate;
}

void KalmanFilterSynthetic::MarkLostTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets,
                                                       const std::vector<int> &assignments,
                                                       const std::vector<int> &target_indexes)
{
  for (int i = 0; i < target_indexes.size(); ++i)
  {
    if (assignments[i] == -1)
    {
      if (unmatched_.find(target_indexes[i]) != unmatched_.end())
      {
        ++unmatched_[target_indexes[i]];
        tracking_statistics_.IncrementNumberOfSuspendedTargets(1);
      } else
      {
        unmatched_[target_indexes[i]] = 1;
        tracking_statistics_.IncrementNumberOfMissedTargets(1);
      }
    }
  }
}

void KalmanFilterSynthetic::MarkAllTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets)
{
  // all the targets have been lost
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
  {
    if (unmatched_.find(it->first) != unmatched_.end())
    {
      ++unmatched_[it->first];
      tracking_statistics_.IncrementNumberOfSuspendedTargets(1);
    } else
    {
      unmatched_[it->first] = 1;
      tracking_statistics_.IncrementNumberOfMissedTargets(1);
    }
  }
}

void KalmanFilterSynthetic::RemoveRecapturedTargetsFromStrikes(std::map<int, Eigen::VectorXf> &targets,
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
        tracking_statistics_.IncrementNumberOfRecapturedTargets(1);
      }
    }
  }
}

void KalmanFilterSynthetic::AddNewTargets(std::map<int, Eigen::VectorXf> &targets,
                                          const std::vector<Eigen::VectorXf> &detections,
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

  // consider unassigned detections as new targets
  for (int i = 0; i < indexes_to_unassigned_detections.size(); ++i)
  {
    targets[max_target_index_ + 1] = detections[indexes_to_unassigned_detections[i]];
    ++max_target_index_;
    tracking_statistics_.IncrementNumberOfNewTargets(1);
  }
}

void KalmanFilterSynthetic::DeleteLongLostTargets(std::map<int, Eigen::VectorXf> &targets)
{
  for (std::map<int, int>::iterator it = unmatched_.begin(); it != unmatched_.end();)
  {
    if (it->second > max_prediction_time_)
    {
//      std::cout << image_idx << " " << it->first << std::endl;
      targets.erase(it->first);
      it = unmatched_.erase(it);
      tracking_statistics_.IncrementNumberOfDeletedTargets(1);
    } else
    {
      ++it;
    }
  }
}

void KalmanFilterSynthetic::SaveTargets(std::ofstream &file,
                                        int time_idx,
                                        const std::map<int, Eigen::VectorXf> &targets)
{
  Eigen::VectorXf x_i;
  file << time_idx << " " << targets.size() << " ";
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
  {
    x_i = it->second;
    file << it->first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3) << " ";
  }
  file << std::endl;
}

void KalmanFilterSynthetic::SaveTargetsMatlab(std::ofstream &file,
                                              int time_idx,
                                              const std::map<int, Eigen::VectorXf> &targets)
{
  Eigen::VectorXf x_i;
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
  {
    x_i = it->second;
    file << time_idx << " " << it->first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3)
         << std::endl;
  }
}

void KalmanFilterSynthetic::SaveTargetsBinary(int time_idx, const std::map<int, Eigen::VectorXf> &targets)
{
  std::ofstream file(
      "/Users/nikita/Documents/spr/sprPerformanceMeasures/spr_simulation_phi_0.063_a_4_U0_250_k_0/kalman_filter_output.bin",
      std::ios::binary | std::ios::out | std::ios::app);
  Real time = time_idx;
  Real x = 0.0;
  file.write((char *) &time, sizeof(Real));
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
  {
    x = (it->second)(0);
    file.write((char *) &x, sizeof(Real));
    x = (it->second)(1);
    file.write((char *) &x, sizeof(Real));
    x = (it->second)(2);
    file.write((char *) &x, sizeof(Real));
    x = (it->second)(3);
    file.write((char *) &x, sizeof(Real));
  }
}

CostInt KalmanFilterSynthetic::InitializeCostMatrix(const std::map<int, Eigen::VectorXf> &targets,
                                                    const std::vector<Eigen::VectorXf> &detections,
                                                    std::vector<std::vector<CostInt>> &cost_matrix,
                                                    std::vector<int> &target_indexes)
{
  target_indexes.clear();

  Eigen::VectorXf target(kNumOfExtractedFeatures);
  Eigen::VectorXf detection(kNumOfExtractedFeatures);
  Real cost = 0.0;
  Real max_cost = 0;
  int i = 0;
  Real d_x = 0.0, d_y = 0.0;// d_v_x = 0.0, d_v_y = 0.0, d_area = 0.0, d_slope = 0.0;
  Real dist = 0.0;
  Real max_dist = Real(std::sqrt(parameter_handler_.GetSubimageXSize() * parameter_handler_.GetSubimageXSize()
                                     + parameter_handler_.GetSubimageYSize() * parameter_handler_.GetSubimageYSize())
                           / 2.0);
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it, ++i)
  {
    target_indexes.push_back(it->first);
    target = it->second;

    for (int j = 0; j < detections.size(); ++j)
    {
      detection = detections[j];
      pbc_config_.ClassAEffectiveParticleDistance(target(0), target(1), detection(0), detection(1), d_x, d_y);

      // put only close assignment costs in the cost matrix
      dist = std::sqrt(d_x * d_x + d_y * d_y);
      if (dist <= parameter_handler_.GetDataAssociationCost())
      {
        cost = dist; // Euclidean norm from a target to a detection
      } else
      {
        cost = max_dist;
      }

      cost_matrix[i][j] = CostInt(cost * costs_order_of_magnitude_);
      if (max_cost < cost)
      {
        max_cost = cost;
      }
    }
  }

  // turn min cost problem into max cost problem
  // rows from n_targets to n_max or columns from n_detections to n_max must remain zero
  for (int i = 0; i < targets.size(); ++i)
  {
    for (int j = 0; j < detections.size(); ++j)
    {
      cost_matrix[i][j] = CostInt(max_cost * costs_order_of_magnitude_) - cost_matrix[i][j];
    }
  }

  return CostInt(max_cost * costs_order_of_magnitude_);
}

void KalmanFilterSynthetic::WriteTrackingStatisticsIntoFile()
{
  std::ostringstream filtering_statistics_file_name_buffer;
  filtering_statistics_file_name_buffer << parameter_handler_.GetFilteringStatisticsFolder() << "statistics.txt";
  std::ofstream filtering_statistics_file(filtering_statistics_file_name_buffer.str(), std::ios::out | std::ios::app);
  filtering_statistics_file << parameter_handler_.GetSimulationParameter("N") << " "
                            << parameter_handler_.GetSimulationParameter("phi") << " "
                            << parameter_handler_.GetSimulationParameter("a") << " "
                            << parameter_handler_.GetSimulationParameter("U_0") << " "
                            << parameter_handler_.GetSimulationParameter("kappa") << " "
                            << parameter_handler_.GetPercentageOfMisdetections() << " "
                            << Real(tracking_statistics_.GetNumberOfSuspendedTargets())
                                / tracking_statistics_.GetNumberOfTimePoints() << " "
                            << Real(tracking_statistics_.GetNumberOfMissedTargets())
                                / tracking_statistics_.GetNumberOfTimePoints() << " "
                            << Real(tracking_statistics_.GetNumberOfDeletedTargets())
                                / tracking_statistics_.GetNumberOfTimePoints() << " "
                            << Real(tracking_statistics_.GetNumberOfNewTargets())
                                / tracking_statistics_.GetNumberOfTimePoints() << " "
                            << Real(tracking_statistics_.GetNumberOfRecapturedTargets())
                                / tracking_statistics_.GetNumberOfTimePoints() << std::endl;
  filtering_statistics_file.clear();
}