//
// Created by Nikita Kruk on 27.09.18.
//

#ifndef SPRMULTITARGETTRACKING_TRAJECTORYLINKER_HPP
#define SPRMULTITARGETTRACKING_TRAJECTORYLINKER_HPP

#include "../Definitions.hpp"
#include "../Parameters/ParameterHandlerExperimental.hpp"
#include "../ImageProcessing/ImageProcessingEngine.hpp"

#include <fstream>
#include <map>
#include <vector>

#include <eigen3/Eigen/Dense>

class TrajectoryLinker
{
 public:

  explicit TrajectoryLinker(ParameterHandlerExperimental &parameter_handler,
                            ImageProcessingEngine &image_processing_engine);
  ~TrajectoryLinker();

  void CreateNewTrackLinkingOutputFiles(ParameterHandlerExperimental &parameter_handler);
  void InitializeTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                              std::map<int, std::vector<int>> &timestamps,
                              std::ifstream &file);
  void PerformTrackLinking(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                           std::map<int, std::vector<int>> &timestamps);

 private:

  ParameterHandlerExperimental &parameter_handler_;
  ImageProcessingEngine &image_processing_engine_;

  std::ofstream track_linking_output_file_;
  std::ofstream track_linking_matlab_output_file_;

  Real costs_order_of_magnitude_;

  void PerformDataAssociationForTrackLinking(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                             std::map<int, std::vector<int>> &timestamps,
                                             int n_max_dim,
                                             std::vector<int> &target_indexes,
                                             std::vector<int> &assignments,
                                             std::vector<CostInt> &costs);
  void UnassignUnrealisticAssociations(std::vector<int> &assignments,
                                       const std::vector<CostInt> &costs);
  void ConnectBrokenTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                 std::map<int, std::vector<int>> &timestamps,
                                 std::vector<int> &target_indexes,
                                 std::vector<int> &assignments,
                                 std::vector<CostInt> &costs);
  void DeleteShortTrajectories(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                               std::map<int, std::vector<int>> &timestamps);
  void PerformTrajectoryContinuation(const std::map<int, std::vector<Eigen::VectorXd>>::iterator &outer_trajectory_iter,
                                     const std::map<int, std::vector<Eigen::VectorXd>>::iterator &inner_trajectory_iter,
                                     const std::map<int, std::vector<int>>::iterator &outer_timestamps_iter,
                                     const std::map<int, std::vector<int>>::iterator &inner_timestamps_iter,
                                     int s);
  void FillHolesInMaps(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                       std::map<int, std::vector<int>> &timestamps);

  void SaveTrajectories(std::ofstream &file,
                        const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                        const std::map<int, std::vector<int>> &timestamps);
  void SaveTrajectoriesMatlab(std::ofstream &file,
                              const std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                              const std::map<int, std::vector<int>> &timestamps);

  CostInt InitializeCostMatrixForTrackLinking(std::map<int, std::vector<Eigen::VectorXd>> &trajectories,
                                              std::map<int, std::vector<int>> &timestamps,
                                              std::vector<std::vector<CostInt>> &cost_matrix,
                                              std::vector<int> &target_indexes);
//  bool CheckDistance(const std::map<int, std::vector<Eigen::VectorXd>>::iterator &outer_trj_it,
//                     const std::map<int, std::vector<Eigen::VectorXd>>::iterator &inner_trj_it);
  bool IsLinkingNearBoundary(const Eigen::VectorXd &outer_trajectory_point,
                             const Eigen::VectorXd &inner_trajectory_point);
  Real ComputeCostMatrixEntryWithoutIntersection(const std::map<int,
                                                                std::vector<Eigen::VectorXd>>::iterator &outer_trj_it,
                                                 const std::map<int,
                                                                std::vector<Eigen::VectorXd>>::iterator &inner_trj_it,
                                                 int s);
  Real ComputeCostMatrixEntryWithIntersection(const std::map<int,
                                                             std::vector<Eigen::VectorXd>>::iterator &outer_trj_it,
                                              const std::map<int,
                                                             std::vector<Eigen::VectorXd>>::iterator &inner_trj_it,
                                              int outer_trj_end_time,
                                              int inner_trj_begin_time);

};

#endif //SPRMULTITARGETTRACKING_TRAJECTORYLINKER_HPP
