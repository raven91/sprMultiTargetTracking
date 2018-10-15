//
// Created by Nikita Kruk on 27.09.18.
//

#include "TrajectoryLinker.hpp"
#include "HungarianAlgorithm.hpp"

#include <cassert>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <iterator>   // std::next
#include <algorithm>
#include <cmath>
#include <numeric>    // std::iota

TrajectoryLinker::TrajectoryLinker(ParameterHandlerExperimental &parameter_handler,
                                   ImageProcessingEngine &image_processing_engine) :
    parameter_handler_(parameter_handler),
    image_processing_engine_(image_processing_engine),
    costs_order_of_magnitude_(1000.0)
{

}

TrajectoryLinker::~TrajectoryLinker()
{
  CloseTrackLinkingOutputFiles();
}

void TrajectoryLinker::CreateTrackLinkingOutputFiles()
{
  std::string track_linking_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
          + parameter_handler_.GetTrackLinkingOutputFileName();
  track_linking_output_file_.open(track_linking_output_file_name, std::ios::out | std::ios::trunc);
  assert(track_linking_output_file_.is_open());

  std::string track_linking_matlab_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
          + parameter_handler_.GetTrackLinkingMatlabOutputFileName();
  track_linking_matlab_output_file_.open(track_linking_matlab_output_file_name, std::ios::out | std::ios::trunc);
  assert(track_linking_matlab_output_file_.is_open());
}

void TrajectoryLinker::CloseTrackLinkingOutputFiles()
{
  if (track_linking_output_file_.is_open())
  {
    track_linking_output_file_.close();
  }
  if (track_linking_matlab_output_file_.is_open())
  {
    track_linking_matlab_output_file_.close();
  }
}

void TrajectoryLinker::InitializeTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                              std::map<int, std::vector<int>> &timestamps)
{
  std::string kalman_filter_output_file_name =
      parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
          + parameter_handler_.GetKalmanFilterOutputFileName();
  std::ifstream kalman_filter_output_file(kalman_filter_output_file_name, std::ios::in);
  assert(kalman_filter_output_file.is_open());

  Eigen::VectorXd new_target = Eigen::MatrixXd::Zero(kNumOfExtractedFeatures, 1);
  int time_idx = 0;
  int target_idx = 0;
  int number_of_targets = 0;

  while (kalman_filter_output_file >> time_idx >> number_of_targets)
  {
    if (time_idx > parameter_handler_.GetLastImage())
    {
      break;
    }
    for (int b = 0; b < number_of_targets; ++b)
    {
      kalman_filter_output_file >> target_idx
                                >> new_target(0) >> new_target(1) >> new_target(2) >> new_target(3)
                                >> new_target(4) >> new_target(5) >> new_target(6) >> new_target(7);

      if (trajectories.find(target_idx) == trajectories.end())
      {
        trajectories[target_idx] = std::vector<Eigen::VectorXd>();
        timestamps[target_idx] = std::vector<int>();
      }
      trajectories[target_idx].push_back(new_target);
      timestamps[target_idx].push_back(time_idx);
    }
  }
}

void TrajectoryLinker::PerformTrackLinking(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                           std::map<int, std::vector<int>> &timestamps)
{
  std::cout << "trajectory linking" << std::endl;

  DeleteShortTrajectories(trajectories, timestamps);

  int n_max_dim = (int) trajectories.size();
  std::vector<int> target_indexes;
  std::vector<int> assignments(n_max_dim, -1);
  std::vector<CostInt> costs(n_max_dim);

  PerformDataAssociationForTrackLinking(trajectories,
                                        timestamps,
                                        n_max_dim,
                                        target_indexes,
                                        assignments,
                                        costs);
  UnassignUnrealisticAssociations(assignments, costs);
  ConnectBrokenTrajectories(trajectories, timestamps, target_indexes, assignments, costs);
  ImposeSuccessiveLabeling(trajectories, timestamps);

  std::cout << "trajectory linking | saving of trajectories" << std::endl;
  SaveTrajectories(track_linking_output_file_, trajectories, timestamps);
  SaveTrajectoriesMatlab(track_linking_matlab_output_file_, trajectories, timestamps);
  SaveImagesWithVectors(trajectories, timestamps);
}

void TrajectoryLinker::PerformDataAssociationForTrackLinking(std::map<int,
                                                                      std::vector<Eigen::VectorXd>> &trajectories,
                                                             std::map<int, std::vector<int>> &timestamps,
                                                             int n_max_dim,
                                                             std::vector<int> &target_indexes,
                                                             std::vector<int> &assignments,
                                                             std::vector<CostInt> &costs)
{
  std::cout << "track linking | data association" << std::endl;
  std::vector<std::vector<CostInt>> cost_matrix(n_max_dim, std::vector<CostInt>(n_max_dim, -1));
  CostInt max_cost = InitializeCostMatrixForTrackLinking(trajectories,
                                                         timestamps,
                                                         cost_matrix,
                                                         target_indexes);

  std::cout << "track linking | hungarian algorithm" << std::endl;
  HungarianAlgorithm hungarian_algorithm(n_max_dim, cost_matrix);
  hungarian_algorithm.Start(assignments, costs);
  std::for_each(costs.begin(),
                costs.end(),
                [&](CostInt &c)
                {
                  c = CostInt((max_cost - c) / costs_order_of_magnitude_);
                });
}

CostInt TrajectoryLinker::InitializeCostMatrixForTrackLinking(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                                              std::map<int, std::vector<int>> &timestamps,
                                                              std::vector<std::vector<CostInt>> &cost_matrix,
                                                              std::vector<int> &target_indexes)
{
  std::cout << "track linking | cost matrix initialization" << std::endl;

  target_indexes.clear();

  Real cost = 0.0;
  Real max_cost = 0.0;

  // the trajectories are not guaranteed to have successive labels
  // row and column indexes are introduced to account for that
  int row = 0;
  for (std::map<int, std::vector<Eigen::VectorXd>>::iterator outer_trj_it = trajectories.begin();
       outer_trj_it != trajectories.end(); ++outer_trj_it, ++row)
  {
    target_indexes.push_back(outer_trj_it->first);

    int column = 0;
    for (std::map<int, std::vector<Eigen::VectorXd>>::iterator inner_trj_it = trajectories.begin();
         inner_trj_it != trajectories.end(); ++inner_trj_it, ++column)
    {
      int outer_trj_idx = outer_trj_it->first;
      int inner_trj_idx = inner_trj_it->first;

//      int outer_trj_begin_time = timestamps[outer_trj_idx][0];
      int outer_trj_end_time = timestamps[outer_trj_idx][timestamps[outer_trj_idx].size() - 1];
      int inner_trj_begin_time = timestamps[inner_trj_idx][0];
//      int inner_trj_end_time = timestamps[inner_trj_idx][timestamps[inner_trj_idx].size() - 1];

      if (inner_trj_it->first == outer_trj_it->first)
      {
        continue; // TODO: is it required?
      }

      // trajectories do not intersect: outer, inner (in time)
      if ((inner_trj_begin_time - outer_trj_end_time >= 1)
          && (inner_trj_begin_time - outer_trj_end_time <= parameter_handler_.GetTrackLinkingLagTime()))
      {
        if (!IsLinkingNearBoundary(outer_trj_it->second[outer_trj_it->second.size() - 1], inner_trj_it->second[0]))
        {
          int s = inner_trj_begin_time - outer_trj_end_time;
          cost = ComputeCostMatrixEntryWithoutIntersection(outer_trj_it, inner_trj_it, s);
          if (max_cost < cost)
          {
            max_cost = cost;
          }
          cost_matrix[row][column] = CostInt(cost * costs_order_of_magnitude_);
//          continue;
        }
      }

/*      // trajectories do not intersect: inner, outer (in time)
      if ((outer_trj_begin_time - inner_trj_end_time >= 1)
          && (outer_trj_begin_time - inner_trj_end_time <= parameter_handler_.GetTrackLinkingLagTime()))
      {
        if (!IsLinkingNearBoundary(outer_trj_it->second[0], inner_trj_it->second[inner_trj_it->second.size() - 1]))
        {
          int s = outer_trj_begin_time - inner_trj_end_time;
          cost = ComputeCostMatrixEntryWithoutIntersection(inner_trj_it, outer_trj_it, s);
          if (max_cost < cost)
          {
            max_cost = cost;
          }
          cost_matrix[row][column] = CostInt(cost * costs_order_of_magnitude_);
          continue;
        }
      }
      */

      // trajectories intersect: outer, inner (in time)
      if ((outer_trj_end_time - inner_trj_begin_time >= 0)
          && (outer_trj_end_time - inner_trj_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime()))
      {
        if (!IsLinkingNearBoundary(outer_trj_it->second[outer_trj_it->second.size() - 1], inner_trj_it->second[0]))
        {
          int s = outer_trj_end_time - inner_trj_begin_time;
          cost = ComputeCostMatrixEntryWithIntersection(outer_trj_it,
                                                        inner_trj_it,
                                                        s);
          if (max_cost < cost)
          {
            max_cost = cost;
          }
          cost_matrix[row][column] = CostInt(cost * costs_order_of_magnitude_);
//          continue;
        }
      }

/*      // trajectories intersect: inner, outer (in time)
      if ((inner_trj_end_time - outer_trj_begin_time >= 0)
          && (inner_trj_end_time - outer_trj_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime()))
      {
        if (!IsLinkingNearBoundary(outer_trj_it->second[0], inner_trj_it->second[inner_trj_it->second.size() - 1]))
        {
          int s = inner_trj_end_time - outer_trj_begin_time;
          cost = ComputeCostMatrixEntryWithIntersection(inner_trj_it,
                                                        outer_trj_it,
                                                        s);
          if (max_cost < cost)
          {
            max_cost = cost;
          }
          cost_matrix[row][column] = CostInt(cost * costs_order_of_magnitude_);
          continue;
        }
      }
      */
    } // inner_trj_it
  } // outer_trj_it

  // turn min cost problem into max cost problem
  for (int row = 0; row < cost_matrix.size(); ++row)
  {
    for (int column = 0; column < cost_matrix.size(); ++column)
    {
      // the complementary values (initialized as -1) are put to zero as needed for the max cost problem
      if (cost_matrix[row][column] < 0)
      {
        cost_matrix[row][column] = 0;
      } else
      {
        cost_matrix[row][column] = CostInt(max_cost * costs_order_of_magnitude_) - cost_matrix[row][column];
      }
    } // column
  } // row

  return CostInt(max_cost * costs_order_of_magnitude_);
}

bool TrajectoryLinker::IsLinkingNearBoundary(const Eigen::VectorXd &outer_trajectory_point,
                                             const Eigen::VectorXd &inner_trajectory_point)
{
  Real x_left = parameter_handler_.GetTrackLinkingRoiMargin();
  Real x_right = parameter_handler_.GetSubimageXSize() - parameter_handler_.GetTrackLinkingRoiMargin();
  Real y_top = parameter_handler_.GetTrackLinkingRoiMargin();
  Real y_bottom = parameter_handler_.GetSubimageYSize() - parameter_handler_.GetTrackLinkingRoiMargin();

  if ((outer_trajectory_point(0) < x_left) && (inner_trajectory_point(0) < x_left))
  {
    return true;
  } else if ((outer_trajectory_point(1) < y_top) && (inner_trajectory_point(1) < y_top))
  {
    return true;
  } else if ((outer_trajectory_point(0) > x_right) && (inner_trajectory_point(0) > x_right))
  {
    return true;
  } else if ((outer_trajectory_point(1) > y_bottom) && (inner_trajectory_point(1) > y_bottom))
  {
    return true;
  } else
  {
    return false;
  }
}

/**
 * outer, inner (in time)
 * @param outer_trj_it
 * @param inner_trj_it
 * @param s
 * @return
 */
Real TrajectoryLinker::ComputeCostMatrixEntryWithoutIntersection(
    const std::map<int, std::vector<Eigen::VectorXd>>::iterator &outer_trj_it,
    const std::map<int, std::vector<Eigen::VectorXd>>::iterator &inner_trj_it,
    int s)
{
  Eigen::VectorXd outer_target_at_last_time = outer_trj_it->second[outer_trj_it->second.size() - 1];
  Eigen::VectorXd
      inner_target_at_first_time = inner_trj_it->second[1]; // note: velocity must be taken from the 1st entry, not 0th

  // predict forward outer trajectory
  std::vector<Eigen::VectorXd> augmented_outer_trajectory;
  Eigen::VectorXd augmented_outer_position = outer_target_at_last_time;
  augmented_outer_trajectory.push_back(augmented_outer_position);
  for (int continuation_time = 1; continuation_time <= s; ++continuation_time)
  {
    augmented_outer_position.head(2) += outer_target_at_last_time.segment(2, 2);
    augmented_outer_trajectory.push_back(augmented_outer_position);
  }

  // building beginning of inner trajectory
  // predict backward inner trajectory
  std::vector<Eigen::VectorXd> augmented_inner_trajectory;
  Eigen::VectorXd augmented_inner_position = inner_target_at_first_time;
  augmented_inner_trajectory.push_back(augmented_inner_position);
  for (int continuation_time = 1; continuation_time <= s; ++continuation_time)
  {
    augmented_inner_position.head(2) -= inner_target_at_first_time.segment(2, 2);
    augmented_inner_trajectory.push_back(augmented_inner_position);
  }
  std::reverse(augmented_inner_trajectory.begin(), augmented_inner_trajectory.end());

  Real cost = 0;
  for (int continuation_time = 0; continuation_time <= s; ++continuation_time)
  {
    cost += (augmented_outer_trajectory[continuation_time].head(2)
        - augmented_inner_trajectory[continuation_time].head(2)).norm();
  }
  return cost / (s + 1);
}

/**
 * outer, inner (in time)
 * @param outer_trj_it
 * @param inner_trj_it
 * @param outer_trj_end_time
 * @param inner_trj_begin_time
 * @return
 */
Real TrajectoryLinker::ComputeCostMatrixEntryWithIntersection(
    const std::map<int, std::vector<Eigen::VectorXd>>::iterator &outer_trj_it,
    const std::map<int, std::vector<Eigen::VectorXd>>::iterator &inner_trj_it,
    int s)
{
  Real cost = 0;
  for (int intersection_time = 0; intersection_time <= s; ++intersection_time)
  {
    Eigen::VectorXd outer_target = outer_trj_it->second[outer_trj_it->second.size() - 1 - s + intersection_time];
    Eigen::VectorXd inner_target = inner_trj_it->second[intersection_time];
    cost += (outer_target.head(2) - inner_target.head(2)).norm();
  }
  return cost / (s + 1);
}

void TrajectoryLinker::UnassignUnrealisticAssociations(std::vector<int> &assignments, const std::vector<CostInt> &costs)
{
  for (int i = 0; i < costs.size(); ++i)
  {
    if (costs[i] > parameter_handler_.GetTrackLinkingDataAssociationCost())
    {
      assignments[i] = -1;
    }
  }
}

void TrajectoryLinker::ConnectBrokenTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                                 std::map<int, std::vector<int>> &timestamps,
                                                 std::vector<int> &target_indexes,
                                                 std::vector<int> &assignments,
                                                 std::vector<CostInt> &costs)
{
  std::cout << "track linking | unification of tracklings" << std::endl;

  int number_of_unifications = 0;
  for (int i = 0; i < assignments.size(); ++i)
  {
    if (assignments[i] != -1)
    {
      int target_idx = target_indexes[i];
      int assignment_idx = target_indexes[assignments[i]];

//      int target_begin_time = timestamps[target_idx][0];
      int target_end_time = timestamps[target_idx][timestamps[target_idx].size() - 1];
      int assignment_begin_time = timestamps[assignment_idx][0];
//      int assignment_end_time = timestamps[assignment_idx][timestamps[assignment_idx].size() - 1];

      std::vector<Eigen::VectorXd> unified_trajectory;
      std::vector<int> unified_timestamp;
      // trajectories do not intersect: target, assignment (in time)
      if ((assignment_begin_time - target_end_time >= 1)
          && (assignment_begin_time - target_end_time <= parameter_handler_.GetTrackLinkingLagTime()))
      {
        int s = assignment_begin_time - target_end_time;
        UnifyNonintersectingTrajectories(trajectories[target_idx],
                                         trajectories[assignment_idx],
                                         timestamps[target_idx],
                                         timestamps[assignment_idx],
                                         s,
                                         unified_trajectory,
                                         unified_timestamp);
      }
/*      // trajectories do not intersect: assignment, target (in time)
      if ((target_begin_time - assignment_end_time >= 1)
          && (target_begin_time - assignment_end_time <= parameter_handler_.GetTrackLinkingLagTime()))
      {
        int s = target_begin_time - assignment_end_time;
        UnifyNonintersectingTrajectories(trajectories[assignment_idx],
                                         trajectories[target_idx],
                                         timestamps[assignment_idx],
                                         timestamps[target_idx],
                                         s,
                                         unified_trajectory,
                                         unified_timestamp);
      }
      */
      // trajectories intersect: target, assignment (in time)
      if ((target_end_time - assignment_begin_time >= 0)
          && (target_end_time - assignment_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime()))
      {
        int s = target_end_time - assignment_begin_time;
        UnifyIntersectingTrajectories(trajectories[target_idx],
                                      trajectories[assignment_idx],
                                      timestamps[target_idx],
                                      timestamps[assignment_idx],
                                      s,
                                      unified_trajectory,
                                      unified_timestamp);
      }
/*      // trajectories intersect: assignment, target (in time)
      if ((assignment_end_time - target_begin_time >= 0)
          && (assignment_end_time - target_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime()))
      {
        int s = assignment_end_time - target_begin_time;
        UnifyIntersectingTrajectories(trajectories[assignment_idx],
                                      trajectories[target_idx],
                                      timestamps[assignment_idx],
                                      timestamps[target_idx],
                                      s,
                                      unified_trajectory,
                                      unified_timestamp);
      }
      */

      trajectories.erase(target_idx);
      trajectories.erase(assignment_idx);
      timestamps.erase(target_idx);
      timestamps.erase(assignment_idx);
      trajectories[target_idx] = unified_trajectory;
      timestamps[target_idx] = unified_timestamp;
      // relabel target indexes because the mapped index can occur later
      std::replace(target_indexes.begin(), target_indexes.end(), assignment_idx, target_idx);
//      std::cout << target_idx << "<->" << assignment_idx << " : " << assignment_begin_time << " : " << costs[i]
//                << std::endl;
      ++number_of_unifications;
    }
  } // i

  std::cout << "track linking | trajectories unified: " << number_of_unifications << std::endl;
}

/**
 * outer, inner (in time)
 * @param outer_trajectory
 * @param inner_trajectory
 * @param outer_timestamps
 * @param inner_timestamps
 * @param s
 * @param unified_trajectory
 */
void TrajectoryLinker::UnifyNonintersectingTrajectories(const std::vector<Eigen::VectorXd> &outer_trajectory,
                                                        const std::vector<Eigen::VectorXd> &inner_trajectory,
                                                        const std::vector<int> &outer_timestamps,
                                                        const std::vector<int> &inner_timestamps,
                                                        int s,
                                                        std::vector<Eigen::VectorXd> &unified_trajectory,
                                                        std::vector<int> &unified_timestamp)
{
  Eigen::VectorXd outer_target_at_last_time = outer_trajectory[outer_trajectory.size() - 1];
  Eigen::VectorXd
      inner_target_at_first_time = inner_trajectory[1]; // note: velocity must be taken from the 1st entry, not 0th

  // predict forward outer trajectory
  std::vector<Eigen::VectorXd> augmented_outer_trajectory;
  Eigen::VectorXd augmented_outer_position = outer_target_at_last_time;
  augmented_outer_trajectory.push_back(augmented_outer_position);
  for (int continuation_time = 1; continuation_time <= s; ++continuation_time)
  {
    augmented_outer_position.head(2) += outer_target_at_last_time.segment(2, 2);
    augmented_outer_trajectory.push_back(augmented_outer_position);
  }

  // predict backward inner trajectory
  std::vector<Eigen::VectorXd> augmented_inner_trajectory;
  Eigen::VectorXd augmented_inner_position = inner_target_at_first_time;
  augmented_inner_trajectory.push_back(augmented_inner_position);
  for (int continuation_time = 1; continuation_time <= s; ++continuation_time)
  {
    augmented_inner_position.head(2) -= inner_target_at_first_time.segment(2, 2);
    augmented_inner_trajectory.push_back(augmented_inner_position);
  }
  std::reverse(augmented_inner_trajectory.begin(), augmented_inner_trajectory.end());

  // combine the original and augmented trajectories
  for (int outer_trj_time_idx = 0; outer_trj_time_idx < outer_trajectory.size() - 1; ++outer_trj_time_idx)
  {
    unified_trajectory.push_back(outer_trajectory[outer_trj_time_idx]);
  }
  for (int unification_time_idx = 0; unification_time_idx <= s; ++unification_time_idx)
  {
    unified_trajectory.push_back(
        (augmented_outer_trajectory[unification_time_idx] + augmented_inner_trajectory[unification_time_idx]) / 2.0);
  }
  for (int inner_trj_time_idx = 1; inner_trj_time_idx < inner_trajectory.size(); ++inner_trj_time_idx)
  {
    unified_trajectory.push_back(inner_trajectory[inner_trj_time_idx]);
  }
  unified_timestamp.resize(unified_trajectory.size());
  std::iota(unified_timestamp.begin(), unified_timestamp.end(), outer_timestamps[0]);
}

/**
 * outer, inner (in time)
 * @param outer_trajectory
 * @param inner_trajectory
 * @param outer_timestamps
 * @param inner_timestamps
 * @param s
 * @param unified_trajectory
 * @param unified_timestamp
 */
void TrajectoryLinker::UnifyIntersectingTrajectories(const std::vector<Eigen::VectorXd> &outer_trajectory,
                                                     const std::vector<Eigen::VectorXd> &inner_trajectory,
                                                     const std::vector<int> &outer_timestamps,
                                                     const std::vector<int> &inner_timestamps,
                                                     int s,
                                                     std::vector<Eigen::VectorXd> &unified_trajectory,
                                                     std::vector<int> &unified_timestamp)
{
  // combine the original and augmented trajectories
  for (int outer_trj_time_idx = 0; outer_trj_time_idx < outer_trajectory.size() - 1 - s; ++outer_trj_time_idx)
  {
    unified_trajectory.push_back(outer_trajectory[outer_trj_time_idx]);
  }
  for (int unification_time_idx = 0; unification_time_idx <= s; ++unification_time_idx)
  {
    unified_trajectory.push_back((outer_trajectory[outer_trajectory.size() - 1 - s + unification_time_idx]
        + inner_trajectory[unification_time_idx]) / 2.0);
  }
  for (int inner_trj_time_idx = s + 1; inner_trj_time_idx < inner_trajectory.size(); ++inner_trj_time_idx)
  {
    unified_trajectory.push_back(inner_trajectory[inner_trj_time_idx]);
  }
  unified_timestamp.resize(unified_trajectory.size());
  std::iota(unified_timestamp.begin(), unified_timestamp.end(), outer_timestamps[0]);
}

void TrajectoryLinker::DeleteShortTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                               std::map<int, std::vector<int>> &timestamps)
{
  int num_deleted_trajectories = 0;
  for (std::map<int, std::vector<Eigen::VectorXd>>::iterator trj_it = trajectories.begin();
       trj_it != trajectories.end();)
  {
    if (trj_it->second.size() <= parameter_handler_.GetMinTrajectoryLength())
    {
      timestamps.erase(trj_it->first);
      trj_it = trajectories.erase(trj_it);
      ++num_deleted_trajectories;
    } else
    {
      ++trj_it;
    }
  }
  std::cout << "track linking | short trajectories deleted: " << num_deleted_trajectories << std::endl;
}

void TrajectoryLinker::ImposeSuccessiveLabeling(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                                std::map<int, std::vector<int>> &timestamps)
{
  std::cout << "track linking | relabeling" << std::endl;

  int label = 0;
  for (std::map<int, std::vector<Eigen::VectorXd>>::iterator trj_it = trajectories.begin();
       trj_it != trajectories.end(); ++label)
  {
    if (trj_it->first != label)
    {
      trajectories[label] = trajectories[trj_it->first];
      timestamps[label] = timestamps[trj_it->first];
      timestamps.erase(trj_it->first);
      trj_it = trajectories.erase(trj_it);
    } else
    {
      ++trj_it;
    }
  }
}

void TrajectoryLinker::SaveTrajectories(std::ofstream &file,
                                        const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                        const std::map<int, std::vector<int>> &timestamps)
{
  for (int img_idx = parameter_handler_.GetFirstImage(); img_idx <= parameter_handler_.GetLastImage(); ++img_idx)
  {
    // go through all trajectories in order to calculate the number of them per image
    int number_of_targets_per_image = 0;
    for (std::map<int, std::vector<int>>::const_iterator cit = timestamps.begin(); cit != timestamps.end(); ++cit)
    {
      if (std::find(cit->second.begin(), cit->second.end(), img_idx) != cit->second.end())
      {
        ++number_of_targets_per_image;
      }
    }

    file << img_idx << " " << number_of_targets_per_image << " ";
    for (std::map<int, std::vector<int>>::const_iterator cit = timestamps.begin(); cit != timestamps.end(); ++cit)
    {
      std::vector<int>::const_iterator trj_it = std::find(cit->second.begin(), cit->second.end(), img_idx);
      if (trj_it != cit->second.end())
      {
        int trj_idx = std::distance(cit->second.begin(), trj_it);
        Eigen::VectorXd target = trajectories.at(cit->first)[trj_idx];
        file << cit->first << " "
             << target(0) << " " << target(1) << " " << target(2) << " " << target(3) << " "
             << target(4) << " " << target(5) << " " << target(6) << " " << target(7) << " ";
      }
    }
    file << std::endl;
  }
}

void TrajectoryLinker::SaveTrajectoriesMatlab(std::ofstream &file,
                                              const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                              const std::map<int, std::vector<int>> &timestamps)
{
  for (std::map<int, std::vector<Eigen::VectorXd>>::const_iterator cit = trajectories.begin();
       cit != trajectories.end();
       ++cit)
  {
    int trajectory_label = cit->first;
    for (int trj_idx = 0; trj_idx < cit->second.size(); ++trj_idx)
    {
      Eigen::VectorXd target(cit->second[trj_idx]);
      file << timestamps.at(trajectory_label)[trj_idx] << " "
           << trajectory_label << " "
           << target(0) << " " << target(1) << " " << target(2) << " " << target(3) << " "
           << target(4) << " " << target(5) << " " << target(6) << " " << target(7)
           << std::endl;
    }
  } // cit
}

void TrajectoryLinker::SaveImagesWithVectors(const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                             const std::map<int, std::vector<int>> &timestamps)
{
  cv::Scalar text_color(255, 127, 0);
  cv::Scalar velocity_color(0, 255, 0);
  cv::Scalar orientation_color(255, 0, 0);

  for (int image_idx = parameter_handler_.GetFirstImage(); image_idx <= parameter_handler_.GetLastImage(); ++image_idx)
  {
    cv::Mat image;
    image = image_processing_engine_.GetSourceImage(image_idx);

    for (std::map<int, std::vector<int>>::const_iterator cit = timestamps.begin(); cit != timestamps.end(); ++cit)
    {
      std::vector<int>::const_iterator trj_it = std::find(cit->second.begin(), cit->second.end(), image_idx);
      if (trj_it != cit->second.end())
      {
        int trj_idx = std::distance(cit->second.begin(), trj_it);
        Eigen::VectorXd target = trajectories.at(cit->first)[trj_idx];

        cv::Point2f center = cv::Point2f(target(0), target(1));
        cv::putText(image, std::to_string(cit->first), center, cv::FONT_HERSHEY_DUPLEX, 0.4, text_color);
        Real length = std::max(target(6), target(7));
        cv::line(image,
                 center,
                 center + cv::Point2f(target(2), target(3)),
                 velocity_color);
        cv::line(image,
                 center,
                 center + cv::Point2f(std::cosf(target(5)), std::sinf(target(5))) * length / 2.0,
                 orientation_color);
      }
    } // cit

    std::ostringstream output_image_name_buf;
    output_image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetTrackLinkingSubfolder()
                          << parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << image_idx
                          << parameter_handler_.GetFileName1();
    std::string output_image_name = output_image_name_buf.str();
    cv::imwrite(output_image_name, image);
  } // image_idx
}

void TrajectoryLinker::SaveImagesWithRectangles(const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                                const std::map<int, std::vector<int>> &timestamps)
{
  for (int image_idx = parameter_handler_.GetFirstImage(); image_idx <= parameter_handler_.GetLastImage(); ++image_idx)
  {
    cv::Mat image;
    image_processing_engine_.ComposeImageForFilterOutput(image_idx, image);

    for (std::map<int, std::vector<int>>::const_iterator cit = timestamps.begin(); cit != timestamps.end(); ++cit)
    {
      std::vector<int>::const_iterator trj_it = std::find(cit->second.begin(), cit->second.end(), image_idx);
      if (trj_it != cit->second.end())
      {
        int trj_idx = std::distance(cit->second.begin(), trj_it);
        Eigen::VectorXd target = trajectories.at(cit->first)[trj_idx];

        cv::Scalar color;
        if (targets_colors_.find(cit->first) != targets_colors_.end())
        {
          color = targets_colors_[cit->first];
        } else
        {
          color = cv::Scalar(rng_.uniform(0, 255), rng_.uniform(0, 255), rng_.uniform(0, 255));
          targets_colors_[cit->first] = color;
        }

        cv::Point2f center(target(0), target(1));
        Real length = std::max(target(6), target(7));
        Real width = std::min(target(6), target(7));
        cv::Point2f lengthwise_vec = cv::Point2f(std::cosf(target(5)), std::sinf(target(5)));
        cv::Point2f widthwise_vec = RotatePoint(lengthwise_vec, M_PI / 2.0);
        lengthwise_vec *= length / 2.0;
        widthwise_vec *= width / 2.0;
        cv::line(image, center + lengthwise_vec + widthwise_vec, center + lengthwise_vec - widthwise_vec, color, 2, 8);
        cv::line(image, center + lengthwise_vec - widthwise_vec, center - lengthwise_vec - widthwise_vec, color, 2, 8);
        cv::line(image, center - lengthwise_vec - widthwise_vec, center - lengthwise_vec + widthwise_vec, color, 2, 8);
        cv::line(image, center - lengthwise_vec + widthwise_vec, center + lengthwise_vec + widthwise_vec, color, 2, 8);
      }
    } // cit

    std::ostringstream output_image_name_buf;
    output_image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetTrackLinkingSubfolder()
                          << parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << image_idx
                          << parameter_handler_.GetFileName1();
    std::string output_image_name = output_image_name_buf.str();
    cv::imwrite(output_image_name, image);
  } // image_idx
}

cv::Point2f TrajectoryLinker::RotatePoint(const cv::Point2f &p, float rad)
{
  const float x = std::cos(rad) * p.x - std::sin(rad) * p.y;
  const float y = std::sin(rad) * p.x + std::cos(rad) * p.y;

  const cv::Point2f rot_p(x, y);
  return rot_p;
}