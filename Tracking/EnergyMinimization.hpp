//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_ENERGYMINIMIZATION_HPP
#define SPRMULTITARGETTRACKING_ENERGYMINIMIZATION_HPP

#include "../Definitions.hpp"

#include <vector>
#include <map>
#include <utility>//std::pair

class EnergyMinimization
{
public:

	EnergyMinimization();
	~EnergyMinimization();

	void Perform();

private:

	int number_of_timepoints_;
	int image_size_x_;
	int image_size_y_;
	std::map<int, std::vector<int>> trajectories_times_;
	std::map<int, std::vector<Real>> trajectories_;
	std::vector<std::vector<Real>> detections_;
	std::map<int, std::vector<Real>> energy_gradient_;

	Real target_size_;
	Real delta_E_crit_;
	Real alpha_;
	Real beta_;
	Real gamma_;
	Real delta_;

	void InitializeTrajectoriesAndDetections();
	void SaveTrajectories();

	Real ComputeTotalEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times);
	Real ComputeObservationModelEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times);
	Real ComputeDynamicModelEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times);
	Real ComputeMutualExclusionEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times);
	Real ComputeTrajectoryPersistenceEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times);

	void PerformConjugateGradientDescent();
	void ComputeEnergyGradient();
	Real ComputeEnergyGradientNorm();
	Real PerformBacktrackingArmijoLineSearch();

	void TryMerging(int i, std::pair<int, Real> &index_energy);
	void PerformMerging(int i, int j);
	void TrySplitting(int i, std::pair<int, Real> &time_energy);
	void PerformSplitting(int i, int t);
	void TryAdding();
	void PerformAdding();
	void TryRemoving(int i, std::pair<int, Real> &index_energy);
	void PerformRemoving(int i);
	void TryGrowing();
	void PerformGrowing();
	void TryShrinking(int i, std::pair<int, Real> &index_energy, bool forward);
	void PerformShrinking(int i, bool forward);

};

#endif //SPRMULTITARGETTRACKING_ENERGYMINIMIZATION_HPP

