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
  void InitializeTrajectories(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                              std::map<int, std::vector<int>> &timestamps,
                              std::ifstream &file);
  void PerformTrackLinking(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                           std::map<int, std::vector<int>> &timestamps);
  bool CheckDistance(const std::map<int, std::vector<Eigen::VectorXf>>::iterator &iter_trj_outer,
                     const std::map<int, std::vector<Eigen::VectorXf>>::iterator &iter_trj_inner);
  CostInt CountCostMatrixElementNOIntersection(const std::map<int,
                                                              std::vector<Eigen::VectorXf>>::iterator &iter_trj_outer,
                                               const std::map<int,
                                                              std::vector<Eigen::VectorXf>>::iterator &iter_trj_inner,
                                               int s);
  CostInt CountCostMatrixElementIntersection(const std::map<int,
                                                            std::vector<Eigen::VectorXf>>::iterator &iter_trj_outer,
                                             const std::map<int,
                                                            std::vector<Eigen::VectorXf>>::iterator &iter_trj_inner,
                                             int Ti_e,
                                             int Tj_b);

 private:

  ParameterHandlerExperimental &parameter_handler_;
  ImageProcessingEngine &image_processing_engine_;

  std::ofstream track_linking_output_file_;
  std::ofstream track_linking_matlab_output_file_;

  Real costs_order_of_magnitude_;

  void PerformDataAssociationForTrackLinking(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                             std::map<int, std::vector<int>> &timestamps,
                                             double &max_elem,
                                             std::vector<int> &target_indexes,
                                             std::vector<std::vector<CostInt>> &cost_matrix,
                                             std::vector<int> &assignments,
                                             std::vector<CostInt> &costs);
  void PerformTrackConnecting(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                              std::map<int, std::vector<int>> &timestamps,
                              std::vector<int> &target_indexes,
                              std::vector<int> &assignments,
                              std::vector<CostInt> &costs,
                              int delta,
                              int tau);
  void DeleteShortTrajectories(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                               std::map<int, std::vector<int>> &timestamps);
  void PerformTrajectoryContinuation(const std::map<int, std::vector<Eigen::VectorXf>>::iterator &outer_trajectory_iter,
                                     const std::map<int, std::vector<Eigen::VectorXf>>::iterator &inner_trajectory_iter,
                                     const std::map<int, std::vector<int>>::iterator &outer_timestamps_iter,
                                     const std::map<int, std::vector<int>>::iterator &inner_timestamps_iter,
                                     int s);
  void FillHolesInMaps(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                       std::map<int, std::vector<int>> &timestamps);
  void SaveTrajectories(std::ofstream &file,
                        const std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                        const std::map<int, std::vector<int>> &timestamps);
  void SaveTrajectoriesMatlab(std::ofstream &file,
                              const std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                              const std::map<int, std::vector<int>> &timestamps);
  CostInt InitializeCostMatrixForTrackLinking(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
                                              std::map<int, std::vector<int>> &timestamps,
                                              double &max_elem,
                                              std::vector<std::vector<CostInt>> &cost_matrix,
                                              std::vector<int> &target_indexes);

};

#endif //SPRMULTITARGETTRACKING_TRAJECTORYLINKER_HPP
