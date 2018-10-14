//
// Created by Nikita Kruk on 03.07.18.
//

#ifndef SPRMULTITARGETTRACKING_KALMANFILTERSYNTHETIC_HPP
#define SPRMULTITARGETTRACKING_KALMANFILTERSYNTHETIC_HPP

#include "../Definitions.hpp"
#include "../Parameters/ParameterHandlerSynthetic.hpp"
#include "../Parameters/PeriodicBoundaryConditionsConfiguration.hpp"
#include "TrackingStatistics.hpp"

#include <vector>
#include <fstream>
#include <map>

#include <eigen3/Eigen/Dense>

class KalmanFilterSynthetic
{
 public:

  explicit KalmanFilterSynthetic(ParameterHandlerSynthetic &parameter_handler,
                                 PeriodicBoundaryConditionsConfiguration &pbc_config);
  ~KalmanFilterSynthetic();

  void InitializeTargets(std::map<int, Eigen::VectorXd> &targets, std::ifstream &file);
  void ObtainNewDetections(std::vector<Eigen::VectorXd> &detections, std::ifstream &file);
  void PerformEstimation(int image_idx,
                         std::map<int, Eigen::VectorXd> &targets,
                         const std::vector<Eigen::VectorXd> &detections);
  void WriteTrackingStatisticsIntoFile();

 private:

  ParameterHandlerSynthetic &parameter_handler_;
  PeriodicBoundaryConditionsConfiguration &pbc_config_;
  TrackingStatistics tracking_statistics_;
  std::ofstream kalman_filter_output_file_;
  std::ofstream kalman_filter_matlab_output_file_;
  std::map<int, int> unmatched_;
  int max_prediction_time_;
  int max_target_index_;
  Real costs_order_of_magnitude_;

  void ComputePriorEstimate(std::map<int, Eigen::VectorXd> &targets,
                            Eigen::MatrixXd &P_estimate,
                            const Eigen::MatrixXd &A,
                            const Eigen::MatrixXd &W,
                            const Eigen::MatrixXd &H);
  void ComputeKalmanGainMatrix(Eigen::MatrixXd &K,
                               const Eigen::MatrixXd &P_estimate,
                               const Eigen::MatrixXd &H,
                               const Eigen::MatrixXd &Q);
  void PerformDataAssociation(const std::map<int, Eigen::VectorXd> &targets,
                              const std::vector<Eigen::VectorXd> &detections,
                              int n_max_dim,
                              std::vector<int> &target_indexes,
                              std::vector<std::vector<CostInt>> &cost_matrix,
                              std::vector<int> &assignments,
                              std::vector<CostInt> &costs);
  void UnassignUnrealisticTargets(const std::map<int, Eigen::VectorXd> &targets,
                                  const std::vector<Eigen::VectorXd> &detections,
                                  int n_max_dim,
                                  std::vector<int> &assignments,
                                  std::vector<CostInt> &costs);
  void ComputePosteriorEstimate(std::map<int, Eigen::VectorXd> &targets,
                                const std::vector<Eigen::VectorXd> &detections,
                                Eigen::MatrixXd &P_estimate,
                                const Eigen::MatrixXd &K,
                                const Eigen::MatrixXd &H,
                                const std::vector<int> &assignments,
                                const std::vector<int> &target_indexes);
  void MarkLostTargetsAsUnmatched(std::map<int, Eigen::VectorXd> &targets,
                                  const std::vector<int> &assignments,
                                  const std::vector<int> &target_indexes);
  void MarkAllTargetsAsUnmatched(std::map<int, Eigen::VectorXd> &targets);
  void RemoveRecapturedTargetsFromStrikes(std::map<int, Eigen::VectorXd> &targets,
                                          const std::vector<int> &assignments,
                                          const std::vector<int> &target_indexes);
  void AddNewTargets(std::map<int, Eigen::VectorXd> &targets,
                     const std::vector<Eigen::VectorXd> &detections,
                     const std::vector<int> &assignments);
  void DeleteLongLostTargets(std::map<int, Eigen::VectorXd> &targets);
  void SaveTargets(std::ofstream &file, int time_idx, const std::map<int, Eigen::VectorXd> &targets);
  void SaveTargetsMatlab(std::ofstream &file, int time_idx, const std::map<int, Eigen::VectorXd> &targets);
  void SaveTargetsBinary(int time_idx, const std::map<int, Eigen::VectorXd> &targets);

  CostInt InitializeCostMatrix(const std::map<int, Eigen::VectorXd> &targets,
                               const std::vector<Eigen::VectorXd> &detections,
                               std::vector<std::vector<CostInt>> &cost_matrix,
                               std::vector<int> &target_indexes);
};

#endif //SPRMULTITARGETTRACKING_KALMANFILTERSYNTHETIC_HPP
