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
    max_target_index_(0)
{

}

KalmanFilterExperimental::~KalmanFilterExperimental()
{
  kalman_filter_output_file_.close();
  kalman_filter_matlab_output_file_.close();
  track_linking_output_file_.close();
  track_linking_matlab_output_file_.close();
}

void KalmanFilterExperimental::CreateNewKalmanFilterOutputFiles(ParameterHandlerExperimental &parameter_handler)
{
  std::string kalman_filter_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler.GetDataAnalysisSubfolder()
          + parameter_handler.GetKalmanFilterOutputFileName();
  kalman_filter_output_file_.open(kalman_filter_output_file_name, std::ios::out | std::ios::trunc);
  assert(kalman_filter_output_file_.is_open());

  std::string kalman_filter_matlab_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
          + parameter_handler_.GetKalmanFilterMatlabOutputFileName();
  kalman_filter_matlab_output_file_.open(kalman_filter_matlab_output_file_name, std::ios::out | std::ios::trunc);
  assert(kalman_filter_matlab_output_file_.is_open());
}

void KalmanFilterExperimental::CreateNewTrackLinkingOutputFiles(ParameterHandlerExperimental &parameter_handler)
{
  std::string track_linking_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler.GetDataAnalysisSubfolder()
          + parameter_handler.GetTrackLinkingOutputFileName();
  track_linking_output_file_.open(track_linking_output_file_name, std::ios::out | std::ios::trunc);
  assert(track_linking_output_file_.is_open());

  std::string track_linking_matlab_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
          + parameter_handler_.GetTrackLinkingMatlabOutputFileName();
  track_linking_matlab_output_file_.open(track_linking_matlab_output_file_name, std::ios::out | std::ios::trunc);
  assert(track_linking_matlab_output_file_.is_open());
}

void KalmanFilterExperimental::InitializeTargets(std::map<int, Eigen::VectorXf> &targets,
                                                 const std::vector<Eigen::VectorXf> &detections)
{
  targets.clear();

  int last_index = 0;
  Eigen::VectorXf new_target(kNumOfExtractedFeatures);
  for (int b = 0; b < detections.size(); ++b)
  {
    if (targets.empty())
    {
      last_index = -1;
    } else
    {
      last_index = std::prev(targets.end())->first;
//			last_index = targets.rbegin()->first;
    }
    new_target = detections[b];
    targets[++last_index] = new_target;
  }
  max_target_index_ = last_index;

  CorrectForOrientationUniqueness(targets);

  SaveTargets(kalman_filter_output_file_, parameter_handler_.GetFirstImage(), targets);
  SaveTargetsMatlab(kalman_filter_matlab_output_file_, parameter_handler_.GetFirstImage(), targets);
  SaveImages(parameter_handler_.GetFirstImage(), targets, Eigen::MatrixXf::Identity(kNumOfStateVars, kNumOfStateVars));
}

void KalmanFilterExperimental::InitializeTargets(std::map<int, Eigen::VectorXf> &targets, std::ifstream &file)
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
//  SaveImages(parameter_handler_.GetFirstImage(), targets);
}

void KalmanFilterExperimental::ObtainNewDetections(std::vector<Eigen::VectorXf> &detections, std::ifstream &file)
{
  detections.clear();

  Eigen::VectorXf new_detection = Eigen::MatrixXf::Zero(kNumOfExtractedFeatures, 1);
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

void KalmanFilterExperimental::InitializeTrajectories(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                                      std::map<int, std::vector<int>> &timestamps,
                                                      std::ifstream &file)
{
  std::cout << "trajectory initialization started" << std::endl;
  int last_index = 0;
  Eigen::VectorXf new_trajectory = Eigen::MatrixXf::Zero(kNumOfExtractedFeatures, 1);
  int time_idx = 0;
  int trajectory_idx = 0;
  int number_of_trajectories = 0;

  while (file >> time_idx >> number_of_trajectories)
  {
    for (int b = 0; b < number_of_trajectories; ++b)
    {
      file >> trajectory_idx
           >> new_trajectory(0) >> new_trajectory(1) >> new_trajectory(2) >> new_trajectory(3)
           >> new_trajectory(4) >> new_trajectory(5) >> new_trajectory(6) >> new_trajectory(7);

      if (trajectories.find(trajectory_idx) == trajectories.end())
      {
        trajectories[trajectory_idx] = std::vector<Eigen::VectorXf>();
        timestamps[trajectory_idx] = std::vector<int>();
      }
      trajectories[trajectory_idx].push_back(new_trajectory);
      timestamps[trajectory_idx].push_back(time_idx);
    }
    if (time_idx == 100)
      break;
  }
}

void KalmanFilterExperimental::PerformEstimation(int image_idx,
                                                 std::map<int, Eigen::VectorXf> &targets,
                                                 const std::vector<Eigen::VectorXf> &detections)
{
  std::cout << "kalman filter: image#" << image_idx << std::endl;

  int n_max_dim = 0; // max size between targets and detections
  CostInt max_cost = 0;
  Real dt = 1;// in ms==image

  Eigen::MatrixXf I = Eigen::MatrixXf::Identity(kNumOfStateVars, kNumOfStateVars);
  Eigen::MatrixXf A = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
  Eigen::MatrixXf W = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
  Eigen::MatrixXf H = Eigen::MatrixXf::Zero(kNumOfDetectionVars, kNumOfStateVars);
  Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(kNumOfDetectionVars, kNumOfDetectionVars);
  Eigen::MatrixXf P_estimate = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
  Eigen::MatrixXf K = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);

  A(0, 0) = A(1, 1) = A(2, 2) = A(3, 3) = 1.0;
  A(0, 2) = A(1, 3) = dt;
  H(0, 0) = H(1, 1) = 1.0;

//  W(0, 0) = W(1, 1) = dt * dt * dt * dt / 4.0f;
//  W(2, 2) = W(3, 3) = dt * dt;
//  W(0, 2) = W(1, 3) = W(2, 0) = W(3, 1) = dt * dt * dt / 2.0f;
//  Q(0, 0) = Q(1, 1) = dt;
  W(0, 0) = W(1, 1) = 10.0 / 3.0 * 2.0 * dt;
  W(2, 2) = W(3, 3) = 5.0 / 3.0 * 2.0 * dt;
  W(0, 2) = W(1, 3) = W(2, 0) = W(3, 1) = 7.5 / 3.0 * 2.0 * dt;
  Q(0, 0) = Q(1, 1) = 20.0;

  P_estimate = W;
  ComputePriorEstimate(targets, P_estimate, A, W, H);
  ComputeKalmanGainMatrix(K, P_estimate, H, Q);

  if (detections.size() > 0)
  {
    n_max_dim = (int) std::max(targets.size(), detections.size());
    std::vector<int> target_indexes;
    std::vector<int> assignments(n_max_dim, -1);
    std::vector<CostInt> costs(n_max_dim);

    PerformDataAssociation(targets, detections, n_max_dim, target_indexes, assignments, costs);
    UnassignUnrealisticTargets(targets, detections, n_max_dim, assignments, costs, target_indexes);
    ComputePosteriorEstimate(targets, detections, P_estimate, K, H, assignments, target_indexes);
    RemoveRecapturedTargetsFromUnmatched(targets, assignments, target_indexes);
    AddNewTargets(targets, detections, assignments);
    MarkLostTargetsAsUnmatched(targets, assignments, target_indexes);
  } else // detections.size() == 0
  {
    MarkAllTargetsAsUnmatched(targets);
  }
  // if the target has been lost for too long -> remove it
  DeleteLongLostTargets(targets);
  CorrectForOrientationUniqueness(targets);

  SaveTargets(kalman_filter_output_file_, image_idx, targets);
  SaveTargetsMatlab(kalman_filter_matlab_output_file_, image_idx, targets);
  SaveImages(image_idx, targets, P_estimate);

  std::cout << "number of overall targets taken part: " << max_target_index_ + 1 << "; number of current targets: "
            << targets.size() << std::endl;
}

void KalmanFilterExperimental::PerformTrackLinking(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                                   std::map<int, std::vector<int>> &timestamps)
{
  int counter = 0;
  const int delta = 2;
  const int tau = 2;
  int n_max_dim = 0;
  double max_elem = 0;
  std::cout << "trajectory linking started" << std::endl;

  n_max_dim = (int) trajectories.size();
  std::vector<int> target_indexes;
  std::vector<std::vector<CostInt>> cost_matrix(n_max_dim, std::vector<CostInt>(n_max_dim, 0));

  std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_outer = trajectories.begin();
  for (; iter_trj_outer != trajectories.end(); ++iter_trj_outer)
  {
    std::cout << "outer_trajectory #" << iter_trj_outer->first << std::endl;
    std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_inner = trajectories.begin();
    for (; iter_trj_inner != trajectories.end(); ++iter_trj_inner)
    {
      int first_trj_idx = iter_trj_outer->first;
      int second_trj_idx = iter_trj_inner->first;

      int Ti_e = timestamps[first_trj_idx][timestamps[first_trj_idx].size() - 1];
      int Tj_b = timestamps[second_trj_idx][0];

      // EXCLUDING trajectories with length < tau + 1
      if ((iter_trj_outer->second.size() <= tau) || (iter_trj_inner->second.size() <= tau))
      {
        cost_matrix[iter_trj_outer->first][iter_trj_inner->first] = -1; // TODO: make sure the indexing is correct
        continue;
      }

      if (iter_trj_inner->first == iter_trj_outer->first)
      {
        cost_matrix[iter_trj_outer->first][iter_trj_inner->first] = -1;
        continue; // TODO: is it required?
      }

      // if trajectories do not intersect
      if ((Tj_b - Ti_e >= 1) && (Tj_b - Ti_e <= delta))
      {
        int s = Tj_b - Ti_e;

        Real v_t_x_outer = iter_trj_outer->second[iter_trj_outer->second.size() - 1](0)
            - iter_trj_outer->second[iter_trj_outer->second.size() - 2](0);
        Real v_t_y_outer = iter_trj_outer->second[iter_trj_outer->second.size() - 1](1)
            - iter_trj_outer->second[iter_trj_outer->second.size() - 2](1);
        Real v_t_x_inner = iter_trj_inner->second[1](0) - (iter_trj_inner->second[0](0));
        Real v_t_y_inner = iter_trj_inner->second[1](1) - (iter_trj_inner->second[0](1));

        std::vector<std::vector<Real>> outer_vect;
        std::vector<std::vector<Real>> inner_vect;

        // building continuation of outer trajectory
        Real continued_trj_x = iter_trj_outer->second[iter_trj_outer->second.size() - 1](0);
        Real continued_trj_y = iter_trj_outer->second[iter_trj_outer->second.size() - 1](1);
        for (int continuation_time = 0; continuation_time <= s; ++continuation_time)
        {
          if (continuation_time == 0) // push the last element of outer trajectory
          {
            std::vector<Real> continued_trj{continued_trj_x, continued_trj_y};
            outer_vect.push_back(continued_trj);
          } else // push other elements
          {
            continued_trj_x += v_t_x_outer;
            continued_trj_y += v_t_y_outer;

            std::vector<Real> continued_trj{continued_trj_x, continued_trj_y};
            outer_vect.push_back(continued_trj);
          }
        }

        // building beginning of inner trajectory
        continued_trj_x = iter_trj_inner->second[0](0);
        continued_trj_y = iter_trj_inner->second[0](1);
        for (int continuation_time = 0; continuation_time <= s; ++continuation_time)
        {
          if (continuation_time == 0) // push the first element of inner trajectory
          {
            std::vector<Real> continued_trj{continued_trj_x, continued_trj_y};
            inner_vect.push_back(continued_trj);
          } else // push other elements
          {
            continued_trj_x -= v_t_x_inner;
            continued_trj_y -= v_t_y_inner;

            std::vector<Real> continued_trj{continued_trj_x, continued_trj_y};
            inner_vect.push_back(continued_trj);
          }
        }
        std::reverse(inner_vect.begin(), inner_vect.end());

        double res = 0;
        for (int continuation_time = 0; continuation_time <= s; ++continuation_time)
        {
          res += std::sqrt(std::pow((outer_vect[continuation_time][0] - inner_vect[continuation_time][0]), 2) +
              std::pow((outer_vect[continuation_time][1] - inner_vect[continuation_time][1]), 2));
        }
        cost_matrix[first_trj_idx][second_trj_idx] = CostInt(res / (s + 1) * costs_order_of_magnitude_);

        if (max_elem < cost_matrix[first_trj_idx][second_trj_idx])
        {
          max_elem = cost_matrix[first_trj_idx][second_trj_idx];
        }
      }
      // if trajectories intersect
      if ((Ti_e - Tj_b >= 0) && (Ti_e - Tj_b <= tau))
      {
        double res = 0;
        int s = Ti_e - Tj_b;

        for (int intersection_time = 0; intersection_time <= s; ++intersection_time)
        {
          res +=
              std::sqrt(std::pow((iter_trj_outer->second[iter_trj_outer->second.size() - 1 - s + intersection_time](0)
                  - iter_trj_inner->second[intersection_time](0)), 2) +
                  std::pow((iter_trj_outer->second[iter_trj_outer->second.size() - 1 - s + intersection_time](1)
                      - iter_trj_inner->second[intersection_time](1)), 2));
        }
        cost_matrix[first_trj_idx][second_trj_idx] = CostInt(res / (s + 1) * costs_order_of_magnitude_);

        if (max_elem < cost_matrix[first_trj_idx][second_trj_idx])
        {
          max_elem = cost_matrix[first_trj_idx][second_trj_idx];
        }
      } else
      {
        cost_matrix[iter_trj_outer->first][iter_trj_inner->first] = -1;
      }

    } // iter_trj_inner
  } // iter_trj_outer

  InitializeCostMatrixForTrackLinking(trajectories, timestamps, max_elem, cost_matrix, target_indexes);

  std::vector<int> assignments(n_max_dim, -1);
  std::vector<CostInt> costs(n_max_dim, -1);
  PerformDataAssociationForTrackLinking(trajectories,
                                        timestamps,
                                        max_elem,
                                        target_indexes,
                                        cost_matrix,
                                        assignments,
                                        costs);
  PerformTrackConnecting(trajectories, timestamps, target_indexes, assignments, costs, delta, tau);

  SaveTrajectories(track_linking_output_file_, trajectories);
  SaveTrajectoriesMatlab(track_linking_matlab_output_file_, trajectories);
}

void KalmanFilterExperimental::ComputePriorEstimate(std::map<int, Eigen::VectorXf> &targets,
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
    (it->second).head(kNumOfStateVars) = x_i_estimate;
  }
  P_estimate = A * P_estimate * A.transpose() + W;
}

void KalmanFilterExperimental::ComputeKalmanGainMatrix(Eigen::MatrixXf &K,
                                                       const Eigen::MatrixXf &P_estimate,
                                                       const Eigen::MatrixXf &H,
                                                       const Eigen::MatrixXf &Q)
{
  K = P_estimate * H.transpose() * (H * P_estimate * H.transpose() + Q).inverse();
}

void KalmanFilterExperimental::PerformDataAssociation(const std::map<int, Eigen::VectorXf> &targets,
                                                      const std::vector<Eigen::VectorXf> &detections,
                                                      int n_max_dim,
                                                      std::vector<int> &target_indexes,
                                                      std::vector<int> &assignments,
                                                      std::vector<CostInt> &costs)
{
  std::vector<std::vector<CostInt>> cost_matrix(n_max_dim, std::vector<CostInt>(n_max_dim, 0));
  CostInt max_cost = InitializeCostMatrix(targets, detections, cost_matrix, target_indexes);
//  std::vector<int> assignments_by_distance(assignments[0]);
  HungarianAlgorithm hungarian_algorithm(n_max_dim, cost_matrix);
  hungarian_algorithm.Start(assignments, costs);
  std::for_each(costs.begin(),
                costs.end(),
                [&](CostInt &c)
                {
                  c = CostInt((max_cost - c) / costs_order_of_magnitude_);
                });

//  cost_matrix = std::vector<std::vector<CostInt>>(n_max_dim, std::vector<CostInt>(n_max_dim, 0));
//  CostInt max_cost_by_area = InitializeCostMatrixOnArea(targets, detections, cost_matrix, target_indexes);
////  std::vector<int> assignments_by_area(assignments[1]);
////  std::vector<CostInt> costs_by_area(n_max_dim);
//  HungarianAlgorithm hungarian_algorithm_2(n_max_dim, cost_matrix);
//  hungarian_algorithm_2.Start(assignments[1], costs[1]);
//  std::for_each(costs[1].begin(),
//                costs[1].end(),
//                [&](CostInt &c)
//                {
//                  c = CostInt((max_cost_by_area - c) / costs_order_of_magnitude_);
//                });
}

void KalmanFilterExperimental::UnassignUnrealisticTargets(const std::map<int, Eigen::VectorXf> &targets,
                                                          const std::vector<Eigen::VectorXf> &detections,
                                                          int n_max_dim,
                                                          std::vector<int> &assignments,
                                                          std::vector<CostInt> &costs,
                                                          const std::vector<int> &target_indexes)
{
  // if the assignment is into an imaginary detection
  for (int i = 0; i < targets.size(); ++i)
  {

  }
  // if a cost is too high
  for (int i = 0; i < targets.size(); ++i)
  {
    if (assignments[i] >= detections.size())
    {
      assignments[i] = -1;
    } else
    {
      Eigen::VectorXf target = targets.at(target_indexes[i]);
      Eigen::VectorXf detection = detections[assignments[i]];
      Real d_x = (target(0) - detection(0));
      Real d_y = (target(1) - detection(1));
      Real dist = std::sqrt(d_x * d_x + d_y * d_y);
      Real area_increase = std::max(target(4), detection(4)) / std::min(target(4), detection(4));

      if ((dist > parameter_handler_.GetDataAssociationCost()) || (area_increase > 1.5)) // in pixels
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

void KalmanFilterExperimental::ComputePosteriorEstimate(std::map<int, Eigen::VectorXf> &targets,
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
      x_i_estimate = x_i_estimate + K * (z_i - H * x_i_estimate);
      targets[target_indexes[i]].head(kNumOfStateVars) = x_i_estimate;

      targets[target_indexes[i]][4] = detections[assignments[i]][4];
      targets[target_indexes[i]][5] = detections[assignments[i]][5];
      targets[target_indexes[i]][6] = detections[assignments[i]][6];
      targets[target_indexes[i]][7] = detections[assignments[i]][7];
    }
  }
  Eigen::MatrixXf I = Eigen::MatrixXf::Identity(kNumOfStateVars, kNumOfStateVars);
  P_estimate = (I - K * H) * P_estimate;
}

void KalmanFilterExperimental::MarkLostTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets,
                                                          const std::vector<int> &assignments,
                                                          const std::vector<int> &target_indexes)
{
  // idx.size() == number of targets at the beginning of the iteration
  for (int i = 0; i < target_indexes.size(); ++i)
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

void KalmanFilterExperimental::RemoveRecapturedTargetsFromUnmatched(std::map<int, Eigen::VectorXf> &targets,
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

void KalmanFilterExperimental::MarkAllTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets)
{
  // all the targets have been lost
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
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

void KalmanFilterExperimental::AddNewTargets(std::map<int, Eigen::VectorXf> &targets,
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
  for (int i = 0; i < indexes_to_unassigned_detections.size(); ++i)
  {
    targets[max_target_index_ + 1] = detections[indexes_to_unassigned_detections[i]];
    ++max_target_index_;
  }
}

void KalmanFilterExperimental::DeleteLongLostTargets(std::map<int, Eigen::VectorXf> &targets)
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

void KalmanFilterExperimental::CorrectForOrientationUniqueness(std::map<int, Eigen::VectorXf> &targets)
{
  Eigen::VectorXf x_i(kNumOfExtractedFeatures);
  Eigen::Vector2f velocity_i;
  Eigen::Vector2f orientation_i;
  for (std::map<int, Eigen::VectorXf>::iterator it = targets.begin(); it != targets.end(); ++it)
  {
    x_i = it->second;
    velocity_i = x_i.segment(2, 2);
    orientation_i << std::cosf(x_i[5]), std::sinf(x_i[5]);

    // in order to determine the orientation vector uniquely,
    // we assume the angle difference between the orientation and velocity is < \pi/2
    if (velocity_i.dot(orientation_i) < 0.0f)
    {
      (it->second)[5] = WrappingModulo(x_i[5] + M_PI, 2 * M_PI);
    } else
    {
      (it->second)[5] = WrappingModulo(x_i[5], 2 * M_PI);
    }
  }
}

void KalmanFilterExperimental::PerformDataAssociationForTrackLinking(std::map<int,
                                                                              std::vector<Eigen::VectorXf>> &trajectories,
                                                                     std::map<int, std::vector<int>> &timestamps,
                                                                     double &max_elem,
                                                                     std::vector<int> &target_indexes,
                                                                     std::vector<std::vector<CostInt>> &cost_matrix,
                                                                     std::vector<int> &assignments,
                                                                     std::vector<CostInt> &costs)
{
  std::cout << "data association for track linking" << std::endl;
  CostInt max_cost = max_elem;
  HungarianAlgorithm hungarian_algorithm(target_indexes.size(), cost_matrix);
  hungarian_algorithm.Start(assignments, costs);
  std::for_each(costs.begin(),
                costs.end(),
                [&](CostInt &c)
                {
                  c = CostInt((max_cost - c) / costs_order_of_magnitude_);
                });
}

void KalmanFilterExperimental::PerformTrackConnecting(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                                      std::map<int, std::vector<int>> &timestamps,
                                                      std::vector<int> &target_indexes,
                                                      std::vector<int> &assignments,
                                                      std::vector<CostInt> &costs,
                                                      int delta,
                                                      int tau)
{
  std::cout << "unification of tracklings" << std::endl;
  int max_allowed_distance = 15 * std::min(delta, tau);

  // check the distance
  for (int i = 0; i < costs.size(); ++i)
  {
    if (costs[i] > max_allowed_distance)
    {
      continue;
    }

    int min_idx = std::min(target_indexes[i], assignments[i]);
    int max_idx = std::max(target_indexes[i], assignments[i]);

    std::map<int, std::vector<Eigen::VectorXf>>::iterator outer_trajectory_iter;
    std::map<int, std::vector<Eigen::VectorXf>>::iterator inner_trajectory_iter;
    std::map<int, std::vector<int>>::iterator outer_timestamps_iter;
    std::map<int, std::vector<int>>::iterator inner_timestamps_iter;

    std::vector<int> trial;

    outer_trajectory_iter = trajectories.find(min_idx);
    inner_trajectory_iter = trajectories.find(max_idx);
    outer_timestamps_iter = timestamps.find(min_idx);
    inner_timestamps_iter = timestamps.find(max_idx);

    int first_trj_idx = outer_trajectory_iter->first;
    int second_trj_idx = inner_trajectory_iter->first;

    int Ti_e = timestamps[first_trj_idx][timestamps[first_trj_idx].size() - 1];
    int Tj_b = timestamps[second_trj_idx][0];

    int s = Ti_e - Tj_b; // TODO: whether trajectories intersect; whether trajectories time-lapsed

    // creating new trajectory by connecting previous two
    std::vector<Eigen::VectorXf> new_trajectory;
    std::vector<int> new_timestamp;
    for (int pre_intersection_time = 0; pre_intersection_time < outer_trajectory_iter->second.size() - 1 - s;
         ++pre_intersection_time)
    {
      new_trajectory.push_back(outer_trajectory_iter->second[pre_intersection_time]);
      new_timestamp.push_back(outer_timestamps_iter->second[pre_intersection_time]);
    } // pre_intersection_time

    for (int intersection_time = 0; intersection_time <= s; ++intersection_time)
    {
      Eigen::VectorXf new_trajectory_part(kNumOfExtractedFeatures);
      for (int state_element = 0; state_element < kNumOfExtractedFeatures; ++state_element)
      {
        new_trajectory_part(state_element) =
            (outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1 - s + intersection_time](
                state_element) + inner_trajectory_iter->second[intersection_time](state_element)) / 2;
      }
      new_trajectory.push_back(new_trajectory_part);
      new_timestamp.push_back(outer_timestamps_iter->second[
                                  outer_trajectory_iter->second.size() - 1 - s + intersection_time]);
    } // intersection_time

    for (int post_intersection_time = s + 1; post_intersection_time < inner_trajectory_iter->second.size();
         ++post_intersection_time)
    {
      new_trajectory.push_back(inner_trajectory_iter->second[post_intersection_time]);
      new_timestamp.push_back(inner_timestamps_iter->second[post_intersection_time]);
    } // post_intersection_time

    // removing old unnecessary trajectories from map
    outer_trajectory_iter = trajectories.find(min_idx);
    if (outer_trajectory_iter != trajectories.end())
    {
      trajectories.erase(outer_trajectory_iter);
    }
    outer_trajectory_iter = trajectories.find(max_idx);
    if (outer_trajectory_iter != trajectories.end())
    {
      trajectories.erase(outer_trajectory_iter);
    }
    // creating new trajectory in a map
    trajectories[min_idx] = new_trajectory;

    // removing old unnecessary timestamps from map
    outer_timestamps_iter = timestamps.find(min_idx);
    if (outer_timestamps_iter != timestamps.end())
    {
      timestamps.erase(outer_timestamps_iter);
    }
    outer_timestamps_iter = timestamps.find(max_idx);
    if (outer_timestamps_iter != timestamps.end())
    {
      timestamps.erase(outer_timestamps_iter);
    }
    // creating new timestampin a map
    timestamps[min_idx] = new_timestamp;

    std::replace(assignments.begin(), assignments.end(), max_idx, min_idx);
    std::replace(target_indexes.begin(), target_indexes.end(), max_idx, min_idx);
  } // i
}

void KalmanFilterExperimental::SaveTargets(std::ofstream &file,
                                           int image_idx,
                                           const std::map<int, Eigen::VectorXf> &targets)
{
  Eigen::VectorXf x_i;
  file << image_idx << " " << targets.size() << " ";
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
  {
    x_i = it->second;
    file << it->first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3) << " " << x_i(4) << " "
         << x_i(5) << " " << x_i(6) << " " << x_i(7) << " ";
  }
  file << std::endl;
}

void KalmanFilterExperimental::SaveTargetsMatlab(std::ofstream &file,
                                                 int image_idx,
                                                 const std::map<int, Eigen::VectorXf> &targets)
{
  Eigen::VectorXf x_i;
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
  {
    x_i = it->second;
    file << image_idx << " " << it->first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3) << " "
         << x_i(4) << " " << x_i(5) << " " << x_i(6) << " " << x_i(7) << std::endl;
  }
}

void KalmanFilterExperimental::SaveTrajectories(std::ofstream &file,
                                                std::map<int, std::vector<Eigen::VectorXf>> &trajectories)
{
  // TODO: save time-points and number of targets per time-point
  std::map<int, std::vector<Eigen::VectorXf>>::iterator it = trajectories.begin();
  for (; it != trajectories.end(); ++it)
  {
    file << it->first << " ";
    for (int i = 0; i < trajectories[it->first].size(); ++i)
    {
      file << it->second[i](0) << " "
           << it->second[i](1) << " "
           << it->second[i](2) << " "
           << it->second[i](3) << " "
           << it->second[i](4) << " "
           << it->second[i](5) << " "
           << it->second[i](6) << " "
           << it->second[i](7) << " ";
    }
    file << std::endl;
  }
}

void KalmanFilterExperimental::SaveTrajectoriesMatlab(std::ofstream &file,
                                                      std::map<int, std::vector<Eigen::VectorXf>> &trajectories)
{
  // TODO: save time-points and number of targets per time-point
  std::map<int, std::vector<Eigen::VectorXf>>::iterator it = trajectories.begin();
  for (; it != trajectories.end(); ++it)
  {
    for (int i = 0; i < trajectories[it->first].size(); ++i)
    {
      file << it->first << " "
           << it->second[i](0) << " "
           << it->second[i](1) << " "
           << it->second[i](2) << " "
           << it->second[i](3) << " "
           << it->second[i](4) << " "
           << it->second[i](5) << " "
           << it->second[i](6) << " "
           << it->second[i](7) << std::endl;
    }
  }
}

void KalmanFilterExperimental::SaveImages(int image_idx,
                                          const std::map<int, Eigen::VectorXf> &targets,
                                          const Eigen::MatrixXf &P)
{
  cv::Mat image;
  image = image_processing_engine_.GetSourceImage();

  Eigen::VectorXf x_i;
  cv::Point2f center;
  cv::Scalar color(255, 127, 0);
  Real length = 0.0f;

  for (std::map<int, Eigen::VectorXf>::const_iterator cit = targets.begin(); cit != targets.end(); ++cit)
  {
    x_i = cit->second;
    center = cv::Point2f(x_i(0), x_i(1));
    cv::circle(image, center, 3, color, -1, 8);
    cv::putText(image, std::to_string(cit->first), center, cv::FONT_HERSHEY_DUPLEX, 0.4, color);
//		cv::Point2f pt = cv::Point2f(std::cosf(x_i(5)), std::sinf(x_i(5)));
    length = std::max(x_i(6), x_i(7));
    cv::line(image,
             center,
             center + cv::Point2f(x_i(2), x_i(3)),
             cv::Scalar(0, 255, 0));
    cv::line(image,
             center,
             center + cv::Point2f(std::cosf(x_i(5)), std::sinf(x_i(5))) * length / 2.0f,
             cv::Scalar(255, 0, 0));
//		std::cout << "(" << center.x << "," << center.y << ") -> (" << center.x + std::cosf(x_i(5)) * x_i(4) / 10.0f << "," << center.y + std::sinf(x_i(5)) * x_i(4) / 10.0f << ")" << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, kNumOfStateVars, kNumOfStateVars>> s(P);
    Eigen::Matrix<std::complex<float>, kNumOfStateVars, 1> eigenvalues = s.eigenvalues();
    Eigen::Matrix<std::complex<float>, kNumOfStateVars, kNumOfStateVars> eigenvectors = s.eigenvectors();

    float angle = std::atan2(std::real(eigenvectors(1, 0)), std::real(eigenvectors(0, 0))) * 180 / M_PI;
    cv::ellipse(image,
                center,
                cv::Size(3 * std::real(eigenvalues(0)), 3 * std::real(eigenvalues(1))),
                std::atan2(std::real(eigenvectors(1, 0)), std::real(eigenvectors(0, 0))) * 180 / M_PI,
                0,
                360,
                cv::Scalar(255, 172, 0));
  }

  std::ostringstream output_image_name_buf;
  output_image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetKalmanFilterSubfolder()
                        << parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << image_idx
                        << parameter_handler_.GetFileName1();
  std::string output_image_name = output_image_name_buf.str();
  cv::imwrite(output_image_name, image);
}

CostInt KalmanFilterExperimental::InitializeCostMatrix(const std::map<int, Eigen::VectorXf> &targets,
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
  Real d_x = 0.0, d_y = 0.0;
  Real dist = 0.0;
//  Real max_dist = Real(std::sqrt(parameter_handler_.GetSubimageXSize() * parameter_handler_.GetSubimageXSize()
//                                     + parameter_handler_.GetSubimageYSize() * parameter_handler_.GetSubimageYSize()));
  Real area_increase = 0.0;
  for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it, ++i)
  {
    target_indexes.push_back(it->first);
    target = it->second;

    for (int j = 0; j < detections.size(); ++j)
    {
      detection = detections[j];
      d_x = (target(0) - detection(0));
      d_y = (target(1) - detection(1));
      dist = std::sqrt(d_x * d_x + d_y * d_y);
      area_increase = std::max(target(4), detection(4)) / std::min(target(4), detection(4));
//      // put only close assignment costs in the cost matrix
//      if (dist <= parameter_handler_.GetDataAssociationCost())
//      {
//        cost = dist; // Euclidean norm from a target to a detection
//      } else
//      {
//        cost = max_dist;
//      }
      cost = dist * area_increase;

      cost_matrix[i][j] = CostInt(cost * costs_order_of_magnitude_);
      if (max_cost < cost)
      {
        max_cost = cost;
      }
    }
  }

  // turn min cost problem into max cost problem
  for (int i = 0; i < targets.size(); ++i)
  {
    for (int j = 0; j < detections.size(); ++j)
    {
      cost_matrix[i][j] = CostInt(max_cost * costs_order_of_magnitude_) - cost_matrix[i][j];
    }
  }
  // the complementary values are left zero as needed for the max cost problem

  return CostInt(max_cost * costs_order_of_magnitude_);
}

CostInt KalmanFilterExperimental::InitializeCostMatrixForTrackLinking(std::map<int,
                                                                               std::vector<Eigen::VectorXf>> &trajectories,
                                                                      std::map<int, std::vector<int>> &timestamps,
                                                                      double &max_elem,
                                                                      std::vector<std::vector<CostInt>> &cost_matrix,
                                                                      std::vector<int> &target_indexes)
{
  std::cout << "cost matrix initialization for track linking" << std::endl;
  target_indexes.clear();

  std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_outer = trajectories.begin();
  for (; iter_trj_outer != trajectories.end(); ++iter_trj_outer)
  {
    target_indexes.push_back(iter_trj_outer->first);
    std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_inner = trajectories.begin();
    for (; iter_trj_inner != trajectories.end(); ++iter_trj_inner)
    {
      int first_trj_idx = iter_trj_outer->first;
      int second_trj_idx = iter_trj_inner->first;

      if (cost_matrix[first_trj_idx][second_trj_idx] < 0)
      {
        cost_matrix[first_trj_idx][second_trj_idx] = CostInt(max_elem);
      }
    } // iter_trj_inner
  } // iter_trj_outer

  // turn min cost problem into max cost problem
  for (int i = 0; i < cost_matrix.size(); ++i)
  {
    for (int j = 0; j < cost_matrix.size(); ++j)
    {
      cost_matrix[i][j] = CostInt(max_elem) - cost_matrix[i][j];
    }
  }

  return CostInt(max_elem);
}
