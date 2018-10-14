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

void TrajectoryLinker::InitializeTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                              std::map<int, std::vector<int>> &timestamps,
                                              std::ifstream &file)
{
  int last_index = 0;
  Eigen::VectorXd new_target = Eigen::MatrixXd::Zero(kNumOfExtractedFeatures, 1);
  int time_idx = 0;
  int target_idx = 0;
  int number_of_targets = 0;

  while (file >> time_idx >> number_of_targets)
  {
    for (int b = 0; b < number_of_targets; ++b)
    {
      file >> target_idx
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
  DeleteShortTrajectories(trajectories, timestamps);
  FillHolesInMaps(trajectories, timestamps);
  SaveTrajectories(track_linking_output_file_, trajectories, timestamps);
  SaveTrajectoriesMatlab(track_linking_matlab_output_file_, trajectories, timestamps);
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
  HungarianAlgorithm hungarian_algorithm(n_max_dim, cost_matrix);
  hungarian_algorithm.Start(assignments, costs);
  std::for_each(costs.begin(),
                costs.end(),
                [&](CostInt &c)
                {
                  c = CostInt((max_cost - c) / costs_order_of_magnitude_);
                });
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

  // check the distance
  for (int i = 0; i < costs.size(); ++i)
  {
    if (assignments[i] != -1)
    {
      int min_idx = std::min(target_indexes[i], assignments[i]);
      int max_idx = std::max(target_indexes[i], assignments[i]);

      std::map<int, std::vector<Eigen::VectorXd>>::iterator outer_trajectory_iter(trajectories.find(min_idx));
      std::map<int, std::vector<Eigen::VectorXd>>::iterator inner_trajectory_iter(trajectories.find(max_idx));
      std::map<int, std::vector<int>>::iterator outer_timestamps_iter(timestamps.find(min_idx));
      std::map<int, std::vector<int>>::iterator inner_timestamps_iter(timestamps.find(max_idx));

      std::vector<int> trial;

      int first_trj_idx = outer_trajectory_iter->first;
      int second_trj_idx = inner_trajectory_iter->first;

      int outer_trj_end_time = timestamps[first_trj_idx][timestamps[first_trj_idx].size() - 1];
      int inner_trj_begin_time = timestamps[second_trj_idx][0];
      int outer_trj_begin_time = timestamps[first_trj_idx][0];
      int inner_trj_end_time = timestamps[second_trj_idx][timestamps[second_trj_idx].size() - 1];

      int s = 0; // TODO: whether trajectories intersect; whether trajectories time-lapsed

      if (outer_trj_end_time - inner_trj_begin_time >= 0
          && outer_trj_end_time - inner_trj_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime())
      {
        s = outer_trj_end_time - inner_trj_begin_time; // trajectories intersect Ti goes BEFORE Tj
      }
      if (inner_trj_end_time - outer_trj_begin_time >= 0
          && inner_trj_end_time - outer_trj_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime())
      {
        s = inner_trj_end_time - outer_trj_begin_time; // trajectories intersect Ti goes AFTER Tj
        outer_trajectory_iter = trajectories.find(max_idx);
        inner_trajectory_iter = trajectories.find(min_idx);
        outer_timestamps_iter = timestamps.find(max_idx);
        inner_timestamps_iter = timestamps.find(min_idx);
      }
      if (inner_trj_begin_time - outer_trj_end_time > 0
          && inner_trj_begin_time - outer_trj_end_time <= parameter_handler_.GetTrackLinkingLagTime())
      {
        s = inner_trj_begin_time - outer_trj_end_time; // trajectories NOT intersect Ti goes BEFORE Tj
        /// HERE perform continuation of trajectories
        PerformTrajectoryContinuation(outer_trajectory_iter,
                                      inner_trajectory_iter,
                                      outer_timestamps_iter,
                                      inner_timestamps_iter,
                                      s);
      }
      if (outer_trj_begin_time - inner_trj_end_time > 0
          && inner_trj_end_time - outer_trj_begin_time <= parameter_handler_.GetTrackLinkingLagTime())
      {
        s = outer_trj_begin_time - inner_trj_end_time; // trajectories NOT intersect Ti goes AFTER Tj
        /// HERE perform continuation of trajectories
        PerformTrajectoryContinuation(inner_trajectory_iter,
                                      outer_trajectory_iter,
                                      inner_timestamps_iter,
                                      outer_timestamps_iter,
                                      s);
      }

      // creating new trajectory by connecting previous two
      std::vector<Eigen::VectorXd> new_trajectory;
      std::vector<int> new_timestamp;
      for (int pre_intersection_time = 0; pre_intersection_time < outer_trajectory_iter->second.size() - 1 - s;
           ++pre_intersection_time)
      {
        new_trajectory.push_back(outer_trajectory_iter->second[pre_intersection_time]);
        new_timestamp.push_back(outer_timestamps_iter->second[pre_intersection_time]);
      } // pre_intersection_time

      for (int intersection_time = 0; intersection_time <= s; ++intersection_time)
      {
        Eigen::VectorXd new_trajectory_part(kNumOfExtractedFeatures);
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
    }
  } // i
}

void TrajectoryLinker::DeleteShortTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                               std::map<int, std::vector<int>> &timestamps)
{
  int min_traj_length = 2;
  int counter = 0;
  for (std::map<int, std::vector<Eigen::VectorXd>>::iterator traj_it = trajectories.begin();
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
    const std::map<int, std::vector<Eigen::VectorXd>>::iterator &outer_trajectory_iter,
    const std::map<int, std::vector<Eigen::VectorXd>>::iterator &inner_trajectory_iter,
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
      Eigen::VectorXd new_outer_point;
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
      Eigen::VectorXd new_inner_point;
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

void TrajectoryLinker::FillHolesInMaps(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                       std::map<int, std::vector<int>> &timestamps)
{
  std::map<int, std::vector<Eigen::VectorXd>>::iterator traj_it;
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
                                        const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
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
                                              const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                              const std::map<int, std::vector<int>> &timestamps)
{
  // TODO: save time-points and number of targets per time-point
  std::map<int, std::vector<Eigen::VectorXd>>::const_iterator it;
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

CostInt TrajectoryLinker::InitializeCostMatrixForTrackLinking(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                                              std::map<int, std::vector<int>> &timestamps,
                                                              std::vector<std::vector<CostInt>> &cost_matrix,
                                                              std::vector<int> &target_indexes)
{
  std::cout << "track linking | cost matrix initialization" << std::endl;

  target_indexes.clear();

  Real cost = 0.0;
  Real max_cost = 0.0;

  for (std::map<int, std::vector<Eigen::VectorXd>>::iterator outer_trj_it = trajectories.begin();
       outer_trj_it != trajectories.end(); ++outer_trj_it)
  {
    target_indexes.push_back(outer_trj_it->first);
    std::cout << "outer_trajectory #" << outer_trj_it->first << std::endl;

    for (std::map<int, std::vector<Eigen::VectorXd>>::iterator inner_trj_it = trajectories.begin();
         inner_trj_it != trajectories.end(); ++inner_trj_it)
    {
      int outer_trj_idx = outer_trj_it->first;
      int inner_trj_idx = inner_trj_it->first;

      int outer_trj_begin_time = timestamps[outer_trj_idx][0];
      int outer_trj_end_time = timestamps[outer_trj_idx][timestamps[outer_trj_idx].size() - 1];
      int inner_trj_begin_time = timestamps[inner_trj_idx][0];
      int inner_trj_end_time = timestamps[inner_trj_idx][timestamps[inner_trj_idx].size() - 1];

//      // EXCLUDING trajectories with length < tau + 1
//      if ((outer_trj_it->second.size() <= tau) || (inner_trj_it->second.size() <= tau))
//      {
//        cost_matrix[outer_trj_it->first][inner_trj_it->first] = -1; // TODO: make sure the indexing is correct
//        continue;
//      }

      if (inner_trj_it->first == outer_trj_it->first)
      {
//        cost_matrix[outer_trj_it->first][inner_trj_it->first] = -1;
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
          cost_matrix[outer_trj_idx][inner_trj_idx] = CostInt(cost * costs_order_of_magnitude_);
          continue;
        }
//        if (!CheckDistance(outer_trj_it, inner_trj_it))
//        {
//          cost_matrix[outer_trj_it->first][inner_trj_it->first] = -1;
//          continue;
//        }
      }

      // trajectories do not intersect: inner, outer (in time)
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
          cost_matrix[outer_trj_idx][inner_trj_idx] = CostInt(cost * costs_order_of_magnitude_);
          continue;
        }
//        if (!CheckDistance(inner_trj_it, outer_trj_it))
//        {
//          cost_matrix[outer_trj_it->first][inner_trj_it->first] = -1;
//          continue;
//        }
      }

      // trajectories intersect: outer, inner (in time)
      if ((outer_trj_end_time - inner_trj_begin_time >= 0)
          && (outer_trj_end_time - inner_trj_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime()))
      {
        if (!IsLinkingNearBoundary(outer_trj_it->second[outer_trj_it->second.size() - 1], inner_trj_it->second[0]))
        {
          cost = ComputeCostMatrixEntryWithIntersection(outer_trj_it,
                                                        inner_trj_it,
                                                        outer_trj_end_time,
                                                        inner_trj_begin_time);
          if (max_cost < cost)
          {
            max_cost = cost;
          }
          cost_matrix[outer_trj_idx][inner_trj_idx] = CostInt(cost * costs_order_of_magnitude_);
          continue;
        }
//        if (!CheckDistance(outer_trj_it, inner_trj_it))
//        {
//          cost_matrix[outer_trj_it->first][inner_trj_it->first] = -1;
//          continue;
//        }
      }

      // trajectories intersect: inner, outer (in time)
      if ((inner_trj_end_time - outer_trj_begin_time >= 0)
          && (inner_trj_end_time - outer_trj_begin_time <= parameter_handler_.GetTrackLinkingIntersectionTime()))
      {
        if (!IsLinkingNearBoundary(outer_trj_it->second[0], inner_trj_it->second[inner_trj_it->second.size() - 1]))
        {
          cost = ComputeCostMatrixEntryWithIntersection(inner_trj_it,
                                                        outer_trj_it,
                                                        inner_trj_end_time,
                                                        outer_trj_begin_time);
          if (max_cost < cost)
          {
            max_cost = cost;
          }
          cost_matrix[outer_trj_idx][inner_trj_idx] = CostInt(cost * costs_order_of_magnitude_);
          continue;
        }
//        if (!CheckDistance(inner_trj_it, outer_trj_it))
//        {
//          cost_matrix[outer_trj_it->first][inner_trj_it->first] = -1;
//          continue;
//        }
      }
    } // inner_trj_it
  } // outer_trj_it

  // turn min cost problem into max cost problem
  for (int i = 0; i < cost_matrix.size(); ++i)
  {
    for (int j = 0; j < cost_matrix.size(); ++j)
    {
      // the complementary values (initialized as -1) are put to zero as needed for the max cost problem
      if (cost_matrix[i][j] < 0)
      {
        cost_matrix[i][j] = 0;
      } else
      {
        cost_matrix[i][j] = CostInt(max_cost * costs_order_of_magnitude_) - cost_matrix[i][j];
      }
    }
  }

  return CostInt(max_cost * costs_order_of_magnitude_);
}

///**
// * outer, inner (in time)
// * @param outer_trj_it
// * @param inner_trj_it
// * @return
// */
//bool TrajectoryLinker::CheckDistance(const std::map<int,
//                                                    std::vector<Eigen::VectorXd>>::iterator &outer_trj_it,
//                                     const std::map<int,
//                                                    std::vector<Eigen::VectorXd>>::iterator &inner_trj_it)
//{
////  int sigma = 25;
//  if (((outer_trj_it->second[outer_trj_it->second.size() - 1](0)) < sigma)
//      && ((inner_trj_it->second[0](0)) < sigma))
//  {
//    return false;
//  }
//  if (((outer_trj_it->second[outer_trj_it->second.size() - 1](1)) < sigma)
//      && ((inner_trj_it->second[0](1)) < sigma))
//  {
//    return false;
//  }
//  if ((parameter_handler_.GetSubimageXSize() - (outer_trj_it->second[outer_trj_it->second.size() - 1](0)) < sigma)
//      && (parameter_handler_.GetSubimageXSize() - (inner_trj_it->second[0](0)) < sigma))
//  {
//    return false;
//  }
//  if ((parameter_handler_.GetSubimageYSize() - (outer_trj_it->second[outer_trj_it->second.size() - 1](1)) < sigma)
//      && (parameter_handler_.GetSubimageYSize() - (inner_trj_it->second[0](1)) < sigma))
//  {
//    return false;
//  }
//  return true;
//}

bool TrajectoryLinker::IsLinkingNearBoundary(const Eigen::VectorXd &outer_trajectory_point,
                                             const Eigen::VectorXd &inner_trajectory_point)
{
  if ((outer_trajectory_point(0) < parameter_handler_.GetTrackLinkingRoiMargin())
      && (inner_trajectory_point(0) < parameter_handler_.GetTrackLinkingRoiMargin()))
  {
    return true;
  } else if ((outer_trajectory_point(1) < parameter_handler_.GetTrackLinkingRoiMargin())
      && (inner_trajectory_point(1) < parameter_handler_.GetTrackLinkingRoiMargin()))
  {
    return true;
  } else if ((outer_trajectory_point(0)
      > parameter_handler_.GetSubimageXSize() - parameter_handler_.GetTrackLinkingRoiMargin())
      && (inner_trajectory_point(0)
          > parameter_handler_.GetSubimageXSize() - parameter_handler_.GetTrackLinkingRoiMargin()))
  {
    return true;
  } else if ((outer_trajectory_point(1)
      > parameter_handler_.GetSubimageYSize() - parameter_handler_.GetTrackLinkingRoiMargin())
      && (inner_trajectory_point(1)
          > parameter_handler_.GetSubimageYSize() - parameter_handler_.GetTrackLinkingRoiMargin()))
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
//  Real v_t_x_outer = iter_trj_outer->second[iter_trj_outer->second.size() - 1](0)
//      - iter_trj_outer->second[iter_trj_outer->second.size() - 2](0);
//  Real v_t_y_outer = iter_trj_outer->second[iter_trj_outer->second.size() - 1](1)
//      - iter_trj_outer->second[iter_trj_outer->second.size() - 2](1);
//  Real v_t_x_inner = iter_trj_inner->second[1](0) - (iter_trj_inner->second[0](0));
//  Real v_t_y_inner = iter_trj_inner->second[1](1) - (iter_trj_inner->second[0](1));
  Eigen::VectorXd outer_target_at_last_time = outer_trj_it->second[outer_trj_it->second.size() - 1];
  Eigen::VectorXd inner_target_at_first_time = inner_trj_it->second[inner_trj_it->second.size() - 1];

//  std::vector<std::vector<Real>> outer_vect;
//  std::vector<std::vector<Real>> inner_vect;
  // continuation of outer trajectory
//  Real continued_trj_x = iter_trj_outer->second[iter_trj_outer->second.size() - 1](0);
//  Real continued_trj_y = iter_trj_outer->second[iter_trj_outer->second.size() - 1](1);
  std::vector<Eigen::VectorXd> augmented_outer_trajectory;
  Eigen::VectorXd augmented_outer_position = outer_target_at_last_time;
  augmented_outer_trajectory.push_back(augmented_outer_position);
  for (int continuation_time = 1; continuation_time <= s; ++continuation_time)
  {
//    continued_trj_x += v_t_x_outer;
//    continued_trj_y += v_t_y_outer;
    augmented_outer_position.head(2) += outer_target_at_last_time.segment(2, 2);

//    std::vector<Real> continued_trj{continued_trj_x, continued_trj_y};
//    outer_vect.push_back(continued_trj);
    augmented_outer_trajectory.push_back(augmented_outer_position);
  }

  // building beginning of inner trajectory
//  continued_trj_x = iter_trj_inner->second[0](0);
//  continued_trj_y = iter_trj_inner->second[0](1);
  std::vector<Eigen::VectorXd> augmented_inner_trajectory;
  Eigen::VectorXd augmented_inner_position = inner_target_at_first_time;
  augmented_inner_trajectory.push_back(augmented_inner_position);
  for (int continuation_time = 1; continuation_time <= s; ++continuation_time)
  {
//    continued_trj_x -= v_t_x_inner;
//    continued_trj_y -= v_t_y_inner;
    augmented_inner_position.head(2) += inner_target_at_first_time.segment(2, 2);

//    std::vector<Real> continued_trj{continued_trj_x, continued_trj_y};
//    inner_vect.push_back(continued_trj);
    augmented_inner_trajectory.push_back(augmented_inner_position);
  }
//  std::reverse(inner_vect.begin(), inner_vect.end());
  std::reverse(augmented_inner_trajectory.begin(), augmented_inner_trajectory.end());

  Real cost = 0;
  for (int continuation_time = 0; continuation_time <= s; ++continuation_time)
  {
//    cost += std::sqrt(std::pow((outer_vect[continuation_time][0] - inner_vect[continuation_time][0]), 2) +
//        std::pow((outer_vect[continuation_time][1] - inner_vect[continuation_time][1]), 2));
    cost += (augmented_outer_trajectory[continuation_time].head(2)
        - augmented_inner_trajectory[continuation_time].head(2)).norm();
  }
  return cost / (s + 1);// * costs_order_of_magnitude_;
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
    int outer_trj_end_time,
    int inner_trj_begin_time)
{
  Real cost = 0;
  int s = outer_trj_end_time - inner_trj_begin_time;

  for (int intersection_time = 0; intersection_time <= s; ++intersection_time)
  {
//    cost +=
//        std::sqrt(std::pow((outer_trj_it->second[outer_trj_it->second.size() - 1 - s + intersection_time](0)
//            - inner_trj_it->second[intersection_time](0)), 2) +
//            std::pow((outer_trj_it->second[outer_trj_it->second.size() - 1 - s + intersection_time](1)
//                - inner_trj_it->second[intersection_time](1)), 2));
    Eigen::VectorXd outer_target = outer_trj_it->second[outer_trj_it->second.size() - 1 - s + intersection_time];
    Eigen::VectorXd inner_target = inner_trj_it->second[intersection_time];
    cost += (outer_target.head(2) - inner_target.head(2)).norm();
  }
  return cost / (s + 1);// * costs_order_of_magnitude_;
}