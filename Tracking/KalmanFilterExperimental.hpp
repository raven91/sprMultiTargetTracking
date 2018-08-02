//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_KALMANFILTER_HPP
#define SPRMULTITARGETTRACKING_KALMANFILTER_HPP

#include "../Definitions.hpp"
#include "../Parameters/ParameterHandlerExperimental.hpp"
#include "../ImageProcessing/ImageProcessingEngine.hpp"

#include <vector>
#include <fstream>
#include <map>

#include <eigen3/Eigen/Dense>

class KalmanFilterExperimental
{
 public:

  explicit KalmanFilterExperimental(ParameterHandlerExperimental &parameter_handler,
                                    ImageProcessingEngine &image_processing_engine);
  ~KalmanFilterExperimental();

  void InitializeTargets(std::map<int, Eigen::VectorXf> &targets, const std::vector<Eigen::VectorXf> &detections);
  void PerformEstimation(int image_idx,
                         std::map<int, Eigen::VectorXf> &targets,
                         const std::vector<Eigen::VectorXf> &detections);

 private:

  ParameterHandlerExperimental &parameter_handler_;
  ImageProcessingEngine &image_processing_engine_;
  std::ofstream kalman_filter_output_file_;
  std::ofstream kalman_filter_matlab_output_file_;
  std::map<int, int> unmatched_;
  int max_prediction_time_;
  int max_target_index_;
  Real costs_order_of_magnitude_;

  void ComputePriorEstimate(std::map<int, Eigen::VectorXf> &targets,
                            Eigen::MatrixXf &P_estimate,
                            const Eigen::MatrixXf &A,
                            const Eigen::MatrixXf &W,
                            const Eigen::MatrixXf &H);
  void ComputeKalmanGainMatrix(Eigen::MatrixXf &K,
                               const Eigen::MatrixXf &P_estimate,
                               const Eigen::MatrixXf &H,
                               const Eigen::MatrixXf &Q);
  void PerformDataAssociation(const std::map<int, Eigen::VectorXf> &targets,
                              const std::vector<Eigen::VectorXf> &detections,
                              int n_max_dim,
                              std::vector<int> &target_indexes,
                              std::vector<std::vector<CostInt>> &cost_matrix,
                              std::vector<int> &assignments,
                              std::vector<CostInt> &costs);
  void UnassignUnrealisticTargets(const std::map<int, Eigen::VectorXf> &targets,
                                  const std::vector<Eigen::VectorXf> &detections,
                                  int n_max_dim,
                                  std::vector<int> &assignments,
                                  std::vector<CostInt> &costs);
  void ComputePosteriorEstimate(std::map<int, Eigen::VectorXf> &targets,
                                const std::vector<Eigen::VectorXf> &detections,
                                Eigen::MatrixXf &P_estimate,
                                const Eigen::MatrixXf &K,
                                const Eigen::MatrixXf &H,
                                const std::vector<int> &assignments,
                                const std::vector<int> &target_indexes);
  void MarkLostTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets,
                                  const std::vector<int> &assignments,
                                  const std::vector<int> &target_indexes);
  void MarkAllTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets);
  void RemoveRecapturedTargetsFromStrikes(std::map<int, Eigen::VectorXf> &targets,
                                          const std::vector<int> &assignments,
                                          const std::vector<int> &target_indexes);
  void AddNewTargets(std::map<int, Eigen::VectorXf> &targets,
                     const std::vector<Eigen::VectorXf> &detections,
                     const std::vector<int> &assignments);
  void DeleteLongLostTargets(std::map<int, Eigen::VectorXf> &targets);
  void CorrectForOrientationUniqueness(std::map<int, Eigen::VectorXf> &targets);
  void SaveTargets(std::ofstream &file, int image_idx, const std::map<int, Eigen::VectorXf> &targets);
  void SaveTargetsMatlab(std::ofstream &file, int image_idx, const std::map<int, Eigen::VectorXf> &targets);
  void SaveImages(int image_idx, const std::map<int, Eigen::VectorXf> &targets);

  CostInt InitializeCostMatrix(const std::map<int, Eigen::VectorXf> &targets,
                               const std::vector<Eigen::VectorXf> &detections,
                               std::vector<std::vector<CostInt>> &cost_matrix,
                               std::vector<int> &target_indexes);
  CostInt InitializeSecondaryCostMatrix(const std::map<int, Eigen::VectorXf> &targets,
                                        const std::vector<Eigen::VectorXf> &detections,
                                        std::vector<std::vector<CostInt>> &cost_matrix,
                                        std::vector<int> &target_indexes);

};

#endif //SPRMULTITARGETTRACKING_KALMANFILTER_HPP
