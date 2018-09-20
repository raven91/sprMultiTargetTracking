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

	void CreateNewKalmanFilterOutputFiles(ParameterHandlerExperimental &parameter_handler);
	void CreateNewTrackLinkingOutputFiles(ParameterHandlerExperimental &parameter_handler);
	void InitializeTargets(std::map<int, Eigen::VectorXf> &targets, const std::vector<Eigen::VectorXf> &detections);
	void InitializeTargets(std::map<int, Eigen::VectorXf> &targets, std::ifstream &file);
	void ObtainNewDetections(std::vector<Eigen::VectorXf> &detections, std::ifstream &file);
	void InitializeTrajectories(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
		std::map<int, std::vector<int>> &timestamps,
		std::ifstream &file);
	void PerformEstimation(int image_idx,
		std::map<int, Eigen::VectorXf> &targets,
		const std::vector<Eigen::VectorXf> &detections);
	void PerformTrackLinking(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
		std::map<int, std::vector<int>> &timestamps);

	bool CheckDistance(std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_outer, std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_inner);

	CostInt CountCostMatrixElementNOIntersection(std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_outer, std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_inner, int s);

	CostInt CountCostMatrixElementIntersection(std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_outer, std::map<int, std::vector<Eigen::VectorXf>>::iterator iter_trj_inner, int Ti_e, int Tj_b);
	


private:

	ParameterHandlerExperimental &parameter_handler_;
	ImageProcessingEngine &image_processing_engine_;

	std::ofstream kalman_filter_output_file_;
	std::ofstream kalman_filter_matlab_output_file_;
	std::ofstream track_linking_output_file_;
	std::ofstream track_linking_matlab_output_file_;

	std::map<int, int> unmatched_;
	int max_prediction_time_;
	int max_target_index_;
	Real costs_order_of_magnitude_;
	Eigen::MatrixXf I_;
	Eigen::MatrixXf A_;
	Eigen::MatrixXf W_;
	Eigen::MatrixXf H_;
	Eigen::MatrixXf Q_;
	Eigen::MatrixXf P_;
	Eigen::MatrixXf K_;

	void FillHolesInMaps(std::map<int, std::vector<Eigen::VectorXf>>& trajectories, std::map<int, std::vector<int>>& timestamps);

	void ComputePriorEstimate(std::map<int, Eigen::VectorXf> &targets);
	void ComputeKalmanGainMatrix();
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
		std::vector<CostInt> &costs,
		const std::vector<int> &target_indexes);
	void ComputePosteriorEstimate(std::map<int, Eigen::VectorXf> &targets,
		const std::vector<Eigen::VectorXf> &detections,
		const std::vector<int> &assignments,
		const std::vector<int> &target_indexes);
	void MarkLostTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets,
		const std::vector<int> &assignments,
		const std::vector<int> &target_indexes);
	void MarkAllTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets);
	void RemoveRecapturedTargetsFromUnmatched(std::map<int, Eigen::VectorXf> &targets,
		const std::vector<int> &assignments,
		const std::vector<int> &target_indexes);
	void AddNewTargets(std::map<int, Eigen::VectorXf> &targets,
		const std::vector<Eigen::VectorXf> &detections,
		const std::vector<int> &assignments);
	void DeleteLongLostTargets(std::map<int, Eigen::VectorXf> &targets);
	void CorrectForOrientationUniqueness(std::map<int, Eigen::VectorXf> &targets);
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

	void PerformTrajectoryContinuation(std::map<int, std::vector<Eigen::VectorXf>>::iterator outer_trajectory_iter, std::map<int, std::vector<Eigen::VectorXf>>::iterator inner_trajectory_iter, std::map<int, std::vector<int>>::iterator outer_timestamps_iter, std::map<int, std::vector<int>>::iterator inner_timestamps_iter, int s);

	void SaveTargets(std::ofstream &file, int image_idx, const std::map<int, Eigen::VectorXf> &targets);
	void SaveTargetsMatlab(std::ofstream &file, int image_idx, const std::map<int, Eigen::VectorXf> &targets);
	void SaveTrajectories(std::ofstream & file, std::map<int, std::vector<Eigen::VectorXf>>& trajectories, std::map<int, std::vector<int>>& timestamps);
	void SaveTrajectoriesMatlab(std::ofstream & file, std::map<int, std::vector<Eigen::VectorXf>>& trajectories, std::map<int, std::vector<int>>& timestamps);
	void SaveImages(int image_idx, const std::map<int, Eigen::VectorXf> &targets);


	CostInt InitializeCostMatrix(const std::map<int, Eigen::VectorXf> &targets,
		const std::vector<Eigen::VectorXf> &detections,
		std::vector<std::vector<CostInt>> &cost_matrix,
		std::vector<int> &target_indexes);
	CostInt InitializeCostMatrixForTrackLinking(std::map<int, std::vector<Eigen::VectorXf>> &trajectories,
		std::map<int, std::vector<int>> &timestamps,
		double &max_elem,
		std::vector<std::vector<CostInt>> &cost_matrix,
		std::vector<int> &target_indexes);

};

#endif //SPRMULTITARGETTRACKING_KALMANFILTER_HPP
