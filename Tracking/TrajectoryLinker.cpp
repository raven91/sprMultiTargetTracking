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

TrajectoryLinker::TrajectoryLinker(ParameterHandlerExperimental &parameter_handler,
                                   ImageProcessingEngine &image_processing_engine) :
    parameter_handler_(parameter_handler),
    image_processing_engine_(image_processing_engine),
    costs_order_of_magnitude_(1000.0)
{

}

TrajectoryLinker::~TrajectoryLinker()
{
  track_linking_output_file_.close();
  track_linking_matlab_output_file_.close();
}

void TrajectoryLinker::CreateNewTrackLinkingOutputFiles(ParameterHandlerExperimental &parameter_handler)
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

void TrajectoryLinker::InitializeTrajectories(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
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
  }
}

void TrajectoryLinker::PerformTrackLinking(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                           std::map<int, std::vector<int>> &timestamps)
{
  std::cout << "trajectory linking started" << std::endl;

  int counter = 0;
  const int delta = 2;
  const int tau = 2;
  int n_max_dim = 0;
  double max_elem = 0;

  n_max_dim = (int) trajectories.size();
  std::vector<int> target_indexes;
  std::vector<std::vector<CostInt>> cost_matrix(n_max_dim, std::vector<CostInt>(n_max_dim, 0));

  std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_outer;
  for (iter_trj_outer = trajectories.begin(); iter_trj_outer != trajectories.end(); ++iter_trj_outer)
  {
    std::cout << "outer_trajectory #" << iter_trj_outer->first << std::endl;

    std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_inner;
    for (iter_trj_inner = trajectories.begin(); iter_trj_inner != trajectories.end(); ++iter_trj_inner)
    {
      int first_trj_idx = iter_trj_outer->first;
      int second_trj_idx = iter_trj_inner->first;

      int Ti_e = timestamps[first_trj_idx][timestamps[first_trj_idx].size() - 1];
      int Tj_b = timestamps[second_trj_idx][0];

      int Ti_b = timestamps[first_trj_idx][0];
      int Tj_e = timestamps[second_trj_idx][timestamps[second_trj_idx].size() - 1];

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

      // trajectories NOT intersect Ti goes BEFORE Tj
      if (Tj_b - Ti_e >= 1 && Tj_b - Ti_e <= delta)
      {
        if (!CheckDistance(iter_trj_outer, iter_trj_inner))
        {
          cost_matrix[iter_trj_outer->first][iter_trj_inner->first] = -1;
          continue;
        }
        int s = Tj_b - Ti_e;
        cost_matrix[first_trj_idx][second_trj_idx] =
            CountCostMatrixElementNOIntersection(iter_trj_outer, iter_trj_inner, s);
        if (max_elem < cost_matrix[first_trj_idx][second_trj_idx])
        {
          max_elem = cost_matrix[first_trj_idx][second_trj_idx];
        }
      }

      // if trajectories do NOT intersect	Ti goes AFTER Tj
      if (Ti_b - Tj_e > 0 && Tj_e - Ti_b <= delta)
      {
        if (!CheckDistance(iter_trj_inner, iter_trj_outer))
        {
          cost_matrix[iter_trj_outer->first][iter_trj_inner->first] = -1;
          continue;
        }
        int s = Ti_b - Tj_e;
        cost_matrix[first_trj_idx][second_trj_idx] =
            CountCostMatrixElementNOIntersection(iter_trj_inner, iter_trj_outer, s);
        if (max_elem < cost_matrix[first_trj_idx][second_trj_idx])
        {
          max_elem = cost_matrix[first_trj_idx][second_trj_idx];
        }
      }

      // if trajectories intersect	Ti goes BEFORE Tj
      if ((Ti_e - Tj_b >= 0) && (Ti_e - Tj_b <= tau))
      {
        if (!CheckDistance(iter_trj_outer, iter_trj_inner))
        {
          cost_matrix[iter_trj_outer->first][iter_trj_inner->first] = -1;
          continue;
        }
        cost_matrix[first_trj_idx][second_trj_idx] =
            CountCostMatrixElementIntersection(iter_trj_outer, iter_trj_inner, Ti_e, Tj_b);
        if (max_elem < cost_matrix[first_trj_idx][second_trj_idx])
        {
          max_elem = cost_matrix[first_trj_idx][second_trj_idx];
        }
      }

      // trajectories intersect	Ti goes AFTER Tj
      if (Tj_e - Ti_b >= 0 && Tj_e - Ti_b <= tau)
      {
        if (CheckDistance(iter_trj_inner, iter_trj_outer) == false)
        {
          cost_matrix[iter_trj_outer->first][iter_trj_inner->first] = -1;
          continue;
        }
        cost_matrix[first_trj_idx][second_trj_idx] =
            CountCostMatrixElementIntersection(iter_trj_inner, iter_trj_outer, Tj_e, Ti_b);
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
  DeleteShortTrajectories(trajectories, timestamps);
  FillHolesInMaps(trajectories, timestamps);
  SaveTrajectories(track_linking_output_file_, trajectories, timestamps);
  SaveTrajectoriesMatlab(track_linking_matlab_output_file_, trajectories, timestamps);
}

bool TrajectoryLinker::CheckDistance(const std::map<int,
                                                    std::vector<Eigen::VectorXf>>::iterator &iter_trj_outer,
                                     const std::map<int,
                                                    std::vector<Eigen::VectorXf>>::iterator &iter_trj_inner)
{
  int sigma = 25;
  if (((iter_trj_outer->second[iter_trj_outer->second.size() - 1](0)) < sigma)
      && ((iter_trj_inner->second[0](0)) < sigma))
  {
    return false;
  }
  if (((iter_trj_outer->second[iter_trj_outer->second.size() - 1](1)) < sigma)
      && ((iter_trj_inner->second[0](1)) < sigma))
  {
    return false;
  }
  if ((parameter_handler_.GetSubimageXSize() - (iter_trj_outer->second[iter_trj_outer->second.size() - 1](0)) < sigma)
      && (parameter_handler_.GetSubimageXSize() - (iter_trj_inner->second[0](0)) < sigma))
  {
    return false;
  }
  if ((parameter_handler_.GetSubimageYSize() - (iter_trj_outer->second[iter_trj_outer->second.size() - 1](1)) < sigma)
      && (parameter_handler_.GetSubimageYSize() - (iter_trj_inner->second[0](1)) < sigma))
  {
    return false;
  }
  return true;
}

CostInt TrajectoryLinker::CountCostMatrixElementNOIntersection(
    const std::map<int, std::vector<Eigen::VectorXf>>::iterator &iter_trj_outer,
    const std::map<int, std::vector<Eigen::VectorXf>>::iterator &iter_trj_inner,
    int s)
{
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
  return res / (s + 1) * costs_order_of_magnitude_;
}

CostInt TrajectoryLinker::CountCostMatrixElementIntersection(
    const std::map<int, std::vector<Eigen::VectorXf>>::iterator &iter_trj_outer,
    const std::map<int, std::vector<Eigen::VectorXf>>::iterator &iter_trj_inner,
    int Ti_e,
    int Tj_b)
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
  return res / (s + 1) * costs_order_of_magnitude_;
}

void TrajectoryLinker::PerformDataAssociationForTrackLinking(std::map<int,
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

void TrajectoryLinker::PerformTrackConnecting(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
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
    int Ti_b = timestamps[first_trj_idx][0];
    int Tj_e = timestamps[second_trj_idx][timestamps[second_trj_idx].size() - 1];

    int s = 0; // TODO: whether trajectories intersect; whether trajectories time-lapsed

    if (Ti_e - Tj_b >= 0 && Ti_e - Tj_b <= tau)
    {
      s = Ti_e - Tj_b; // trajectories intersect Ti goes BEFORE Tj
    }
    if (Tj_e - Ti_b >= 0 && Tj_e - Ti_b <= tau)
    {
      s = Tj_e - Ti_b; // trajectories intersect Ti goes AFTER Tj
      outer_trajectory_iter = trajectories.find(max_idx);
      inner_trajectory_iter = trajectories.find(min_idx);
      outer_timestamps_iter = timestamps.find(max_idx);
      inner_timestamps_iter = timestamps.find(min_idx);
    }
    if (Tj_b - Ti_e > 0 && Tj_b - Ti_e <= delta)
    {
      s = Tj_b - Ti_e; // trajectories NOT intersect Ti goes BEFORE Tj
      /// HERE perform continuation of trajectories
      PerformTrajectoryContinuation(outer_trajectory_iter,
                                    inner_trajectory_iter,
                                    outer_timestamps_iter,
                                    inner_timestamps_iter,
                                    s);
    }
    if (Ti_b - Tj_e > 0 && Tj_e - Ti_b <= delta)
    {
      s = Ti_b - Tj_e; // trajectories NOT intersect Ti goes AFTER Tj
      /// HERE perform continuation of trajectories
      PerformTrajectoryContinuation(inner_trajectory_iter,
                                    outer_trajectory_iter,
                                    inner_timestamps_iter,
                                    outer_timestamps_iter,
                                    s);
    }

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
      std::cout << "At " << timestamps.find(max_idx)->second[0] << "th second " << "beginning trajectory index = "
                << outer_trajectory_iter->first;
      trajectories.erase(outer_trajectory_iter);
    }
    outer_trajectory_iter = trajectories.find(max_idx);
    if (outer_trajectory_iter != trajectories.end())
    {
      std::cout << " | | ending trajectory index = " << outer_trajectory_iter->first << "and they are now connected"
                << std::endl;
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
    // creating new timestamp in a map
    timestamps[min_idx] = new_timestamp;

    std::replace(assignments.begin(), assignments.end(), max_idx, min_idx);
    std::replace(target_indexes.begin(), target_indexes.end(), max_idx, min_idx);
  } // i
}

void TrajectoryLinker::DeleteShortTrajectories(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                               std::map<int, std::vector<int>> &timestamps)
{
  int min_traj_length = 2;
  int counter = 0;
  for (std::map<int, std::vector<Eigen::VectorXf>>::iterator traj_it = trajectories.begin();
       traj_it != trajectories.end();)
  {
    if (traj_it->second.size() <= min_traj_length)
    {
      std::map<int, std::vector<int>>::iterator time_it_del = timestamps.find(traj_it->first);
      timestamps.erase(time_it_del);
      traj_it = trajectories.erase(traj_it);
      ++counter;
    } else
    {
      ++traj_it;
    }
  }
  std::cout << "# deleted short trajectories: " << counter << std::endl;
}

void TrajectoryLinker::PerformTrajectoryContinuation(
    const std::map<int, std::vector<Eigen::VectorXf>>::iterator &outer_trajectory_iter,
    const std::map<int, std::vector<Eigen::VectorXf>>::iterator &inner_trajectory_iter,
    const std::map<int, std::vector<int>>::iterator &outer_timestamps_iter,
    const std::map<int, std::vector<int>>::iterator &inner_timestamps_iter,
    int s)
{
  Real v_t_x_outer = outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](0)
      - outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 2](0);
  Real v_t_y_outer = outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](1)
      - outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 2](1);

  Real v_t_x_inner = inner_trajectory_iter->second[1](0) - (inner_trajectory_iter->second[0](0));
  Real v_t_y_inner = inner_trajectory_iter->second[1](1) - (inner_trajectory_iter->second[0](1));

  // building continuation of outer trajectory
  Real continued_trj_x = outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](0);
  Real continued_trj_y = outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](1);
  for (int continuation_time = 0; continuation_time <= s; ++continuation_time)
  {
    if (continuation_time == 0) // It's the last element of outer trajectory
    {
      continue;// just skip it
    } else // push other elements
    {
      Eigen::VectorXf new_outer_point;
      continued_trj_x += v_t_x_outer;
      continued_trj_y += v_t_y_outer;
      new_outer_point << continued_trj_x, continued_trj_y, v_t_x_outer, v_t_y_outer,
          outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](4),
          outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](5),
          outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](6),
          outer_trajectory_iter->second[outer_trajectory_iter->second.size() - 1](7);

      outer_trajectory_iter->second.push_back(new_outer_point);
      outer_timestamps_iter->second.push_back(
          outer_timestamps_iter->second[outer_timestamps_iter->second.size() - 1] + 1);
    }
  }

  // building beginning of inner trajectory
  continued_trj_x = inner_trajectory_iter->second[0](0);
  continued_trj_y = inner_trajectory_iter->second[0](1);
  for (int continuation_time = 0; continuation_time <= s; ++continuation_time)
  {
    if (continuation_time == 0) // It's the first element of inner trajectory
    {
      continue;// just skip it
    } else // push other elements
    {
      Eigen::VectorXf new_inner_point;
      continued_trj_x -= v_t_x_inner;
      continued_trj_y -= v_t_y_inner;
      new_inner_point << continued_trj_x, continued_trj_y, v_t_x_inner, v_t_y_inner,
          inner_trajectory_iter->second[0](4),
          inner_trajectory_iter->second[0](5),
          inner_trajectory_iter->second[0](6),
          inner_trajectory_iter->second[0](7);

      inner_trajectory_iter->second.insert(inner_trajectory_iter->second.begin(), new_inner_point);
      inner_timestamps_iter->second.insert(inner_timestamps_iter->second.begin(), inner_timestamps_iter->second[0] - 1);
    }
  }
}

void TrajectoryLinker::FillHolesInMaps(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                       std::map<int, std::vector<int>> &timestamps)
{
  std::map<int, std::vector<Eigen::VectorXf>>::iterator traj_it;
  for (traj_it = trajectories.begin(); traj_it != trajectories.end(); ++traj_it)
  {
    //auto cur_traj_it = traj_it;
    auto next_traj_it = std::next(traj_it);
    if (next_traj_it != trajectories.end())
    {
      if (next_traj_it->first - traj_it->first > 1)
      {
        //std::cout << "first  key " << traj_it->first << "  " << trajectories[traj_it->first][0] << std::endl;
        //std::cout << "next first  key " << next_traj_it->first << "  " << trajectories[next_traj_it->first][0] << std::endl;
        //std::cout << "last   key " << std::prev(trajectories.end())->first << "  " << trajectories[std::prev(trajectories.end())->first][0] << std::endl;
        trajectories[traj_it->first + 1] = std::prev(trajectories.end())->second;
        timestamps[traj_it->first + 1] = std::prev(timestamps.end())->second;
        trajectories.erase(std::prev(trajectories.end()));
        timestamps.erase(std::prev(timestamps.end()));
        //std::cout << "second key " << (std::next(traj_it))->first << "  " << trajectories[(std::next(traj_it))->first][0] << std::endl;
      }
    }
  }
  if (trajectories.begin()->first > 0)
  {
    for (int i = 0; i < trajectories.begin()->first; ++i)
    {
      trajectories[i] = std::prev(trajectories.end())->second;
      timestamps[i] = std::prev(timestamps.end())->second;
      trajectories.erase(std::prev(trajectories.end()));
      timestamps.erase(std::prev(timestamps.end()));
    }
  }
}

void TrajectoryLinker::SaveTrajectories(std::ofstream &file,
                                        const std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                        const std::map<int, std::vector<int>> &timestamps)
{
  // TODO: save time-points and number of targets per time-point
  for (int i = parameter_handler_.GetFirstImage(); i <= parameter_handler_.GetLastImage(); ++i)
  {
    int counter = 0;

    cv::Mat image;
    image = image_processing_engine_.GetSourceImage(i);
    cv::Point2f center;
    cv::Scalar color(255, 127, 0);

    std::map<int, std::vector<int>>::const_iterator time;
    for (time = timestamps.begin(); time != timestamps.end(); ++time)
    {
      if (std::find(time->second.begin(), time->second.end(), i) != time->second.end())
        ++counter;
    }
    file
        << i << " "
        << counter << " ";
    for (time = timestamps.begin(); time != timestamps.end(); ++time)
    {
      if (std::find(time->second.begin(), time->second.end(), i) != time->second.end())
      {

        auto ts_iter = std::find(time->second.begin(), time->second.end(), i);
        if (time->second.size() == trajectories.at(time->first).size())
        {
          auto bactery_data = trajectories.at(time->first)[std::distance(time->second.begin(), ts_iter)];
          file
              << time->first << " "
              << bactery_data(0) << " "
              << bactery_data(1) << " "
              << bactery_data(2) << " "
              << bactery_data(3) << " "
              << bactery_data(4) << " "
              << bactery_data(5) << " "
              << bactery_data(6) << " "
              << bactery_data(7) << " ";

          center = cv::Point2f(bactery_data(0), bactery_data(1));
          cv::circle(image, center, 3, color, -1, 8);
          cv::putText(image, std::to_string(time->first), center, cv::FONT_HERSHEY_DUPLEX, 0.4, color);
          cv::line(image,
                   center,
                   center + cv::Point2f(bactery_data(2), bactery_data(3)),
                   cv::Scalar(255, 0, 0));
        }
      }
    }
    file << std::endl;
    std::ostringstream output_image_name_buf;
    output_image_name_buf << parameter_handler_.GetInputFolder()
                          << parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << i
                          << parameter_handler_.GetFileName1();
    std::string output_image_name = output_image_name_buf.str();
    cv::imwrite(output_image_name, image);
  }
}

void TrajectoryLinker::SaveTrajectoriesMatlab(std::ofstream &file,
                                              const std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                              const std::map<int, std::vector<int>> &timestamps)
{
  // TODO: save time-points and number of targets per time-point
  std::map<int, std::vector<Eigen::VectorXf>>::const_iterator it;
  std::map<int, std::vector<int>>::const_iterator time = timestamps.begin();
  for (it = trajectories.begin(); it != trajectories.end(); ++it)
  {
    for (int i = 0; i < trajectories.at(it->first).size(); ++i)
    {
      file
          << time->second[i] << " "   // index of every image
          << it->first << " "
          << it->second[i](0) << " "
          << it->second[i](1) << " "
          << it->second[i](2) << " "
          << it->second[i](3) << " "
          << it->second[i](4) << " "
          << it->second[i](5) << " "
          << it->second[i](6) << " "
          << it->second[i](7) << std::endl;
    }
    ++time;
  }
}

CostInt TrajectoryLinker::InitializeCostMatrixForTrackLinking(std::map<int,
                                                                       std::vector<Eigen::VectorXf>> &trajectories,
                                                              std::map<int, std::vector<int>> &timestamps,
                                                              double &max_elem,
                                                              std::vector<std::vector<CostInt>> &cost_matrix,
                                                              std::vector<int> &target_indexes)
{
  std::cout << "cost matrix initialization for track linking" << std::endl;

  target_indexes.clear();

  std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_outer;
  for (iter_trj_outer = trajectories.begin(); iter_trj_outer != trajectories.end(); ++iter_trj_outer)
  {
    target_indexes.push_back(iter_trj_outer->first);
    std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_inner;
    for (iter_trj_inner = trajectories.begin(); iter_trj_inner != trajectories.end(); ++iter_trj_inner)
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