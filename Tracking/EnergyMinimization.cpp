//
// Created by Nikita Kruk on 27.11.17.
//

#include "EnergyMinimization.hpp"

#include <vector>
#include <fstream>
#include <iterator>//std::istream_iterator, std::distance, std::prev
#include <algorithm>//std::copy, std::max, std::min, std::max_element
#include <cmath>
#include <limits>//std::numeric_limits
#include <cassert>

EnergyMinimization::EnergyMinimization() :
	number_of_timepoints_(20),
	image_size_x_(256),
	image_size_y_(256),
	trajectories_times_(),
	trajectories_(),
	detections_(number_of_timepoints_, std::vector<Real>()),
	energy_gradient_(),
	target_size_(20.0),
	delta_E_crit_(0.0),
	alpha_(1.0),
	beta_(0.01),
	gamma_(0.5),
	delta_(0.02)
{

}

EnergyMinimization::~EnergyMinimization()
{

}

void EnergyMinimization::Perform()
{
	InitializeTrajectoriesAndDetections();

	//	for (std::map<int, std::vector<int>>::iterator time_iterator = trajectories_times_.begin(); time_iterator != trajectories_times_.end(); )
	//	{
	//		if ((time_iterator->second).size() < 3)
	//		{
	//			trajectories_.erase(time_iterator->first);
	//			trajectories_times_.erase(time_iterator++);
	//		}
	//		else
	//		{
	//			++time_iterator;
	//		}
	//	}

	//	bool convergence_achieved = false;
	//	Real E_new = 0.0, E_prev = 0.0;
	for (int k = 0; k < 50; ++k)
	{
		std::cout << "minimization step: " << k << ", E_total: " << ComputeTotalEnergy(trajectories_, trajectories_times_) << std::endl;

		//Growing
		//Shrinking
		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
		{
			std::pair<int, Real> min_energy(-1, std::numeric_limits<Real>::max());
			if (trajectories_times_[trajectory_iterator->first].size() >= 3)
			{
				TryShrinking(trajectory_iterator->first, min_energy, false);
			}
			if (min_energy.first != -1)
			{
				std::cout << "shrinking trajectory " << trajectory_iterator->first << std::endl;
				PerformShrinking(trajectory_iterator->first, false);
			}
		}
		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
		{
			std::pair<int, Real> min_energy(-1, std::numeric_limits<Real>::max());
			if (trajectories_times_[trajectory_iterator->first].size() >= 3)
			{
				TryShrinking(trajectory_iterator->first, min_energy, true);
			}
			if (min_energy.first != -1)
			{
				std::cout << "shrinking trajectory " << trajectory_iterator->first << std::endl;
				PerformShrinking(trajectory_iterator->first, true);
			}
		}

		//Merging
		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
		{
			std::pair<int, Real> min_energy(-1, std::numeric_limits<Real>::max());
			TryMerging(trajectory_iterator->first, min_energy);

			if (min_energy.first != -1)
			{
				std::cout << "merging trajectory " << trajectory_iterator->first << " with trajectory " << min_energy.first << std::endl;
				PerformMerging(trajectory_iterator->first, min_energy.first);
			}
		}

		//Splitting
		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
		{
			std::pair<int, Real> min_energy(-1, std::numeric_limits<Real>::max());
			TrySplitting(trajectory_iterator->first, min_energy);

			if (min_energy.first != -1)
			{
				std::cout << "splitting trajectory " << trajectory_iterator->first << " at time " << min_energy.first << " into a new trajectory " << (std::prev(trajectories_.end()))->first + 1 << std::endl;
				PerformSplitting(trajectory_iterator->first, min_energy.first);
			}
		}

		//Adding
		//Removing
		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); )
		{
			std::pair<int, Real> min_energy(-1, std::numeric_limits<Real>::max());
			TryRemoving(trajectory_iterator->first, min_energy);

			if (min_energy.first != -1)
			{
				std::cout << "removing trajectory " << trajectory_iterator->first << std::endl;
				PerformRemoving((trajectory_iterator++)->first);
			}
			else
			{
				++trajectory_iterator;
			}
		}

		PerformConjugateGradientDescent();
	}

	////	E_new = ComputeTotalEnergy(trajectories_, trajectories_times_);
	//	for (int k = 0; k < 50; ++k)
	////	while (!convergence_achieved)
	//	{
	//		std::cout << "minimization step: " << k << ", E_total: " << ComputeTotalEnergy(trajectories_, trajectories_times_) << std::endl;
	//
	//		//Growing
	//		//Shrinking
	//		std::map<int, std::vector<Real>>::iterator last_trajectory_iterator = std::prev(trajectories_.end());
	//		std::vector<std::pair<int, Real>> min_energies(last_trajectory_iterator->first + 1, std::pair<int, Real>(-1, std::numeric_limits<Real>::max()));
	//		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
	//		{
	//			if (trajectories_times_[trajectory_iterator->first].size() >= 3)
	//			{
	//				TryShrinking(trajectory_iterator->first, min_energies[trajectory_iterator->first], false);
	//			}
	//		}
	//
	//		std::vector<std::pair<int, Real>>::iterator min_energies_iterator = std::min_element(min_energies.begin(), min_energies.end(), [](const std::pair<int, Real> &p1, const std::pair<int, Real> &p2) { return p1.second < p2.second; } );
	//		size_t trajectory_index = std::distance(min_energies.begin(), min_energies_iterator);
	////		if (min_energies_iterator->first != -1)
	////		{
	////			std::cout << "shrinking trajectory " << trajectory_index << std::endl;
	////			PerformShrinking((int)trajectory_index, false);
	////		}
	//		for (std::vector<std::pair<int, Real>>::iterator min_energies_iterator = min_energies.begin(); min_energies_iterator != min_energies.end(); ++min_energies_iterator)
	//		{
	//			if (min_energies_iterator->first != -1)
	//			{
	//				std::cout << "shrinking trajectory " << std::distance(min_energies.begin(), min_energies_iterator) << std::endl;
	//			}
	//		}
	//
	//		//Merging
	//		last_trajectory_iterator = std::prev(trajectories_.end());
	//		min_energies = std::vector<std::pair<int, Real>>(last_trajectory_iterator->first + 1, std::pair<int, Real>(-1, std::numeric_limits<Real>::max()));
	//		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
	//		{
	//			TryMerging(trajectory_iterator->first, min_energies[trajectory_iterator->first]);
	//		}
	//
	//		min_energies_iterator = std::min_element(min_energies.begin(), min_energies.end(), [](const std::pair<int, Real> &p1, const std::pair<int, Real> &p2) { return p1.second < p2.second; } );
	//		trajectory_index = std::distance(min_energies.begin(), min_energies_iterator);
	//		if (min_energies_iterator->first != -1)
	//		{
	//			std::cout << "merging trajectory " << trajectory_index << " with trajectory " << min_energies_iterator->first << std::endl;
	//			PerformMerging((int)trajectory_index, min_energies_iterator->first);
	//		}
	//
	//		//Splitting
	//		last_trajectory_iterator = std::prev(trajectories_.end());
	//		min_energies = std::vector<std::pair<int, Real>>(last_trajectory_iterator->first + 1, std::pair<int, Real>(-1, std::numeric_limits<Real>::max()));
	//		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
	//		{
	//			TrySplitting(trajectory_iterator->first, min_energies[trajectory_iterator->first]);
	//		}
	//
	////		min_energies_iterator = std::min_element(min_energies.begin(), min_energies.end(), [](const std::pair<int, Real> &p1, const std::pair<int, Real> &p2) { return p1.second < p2.second; } );
	////		trajectory_index = std::distance(min_energies.begin(), min_energies_iterator);
	////		if (min_energies_iterator->first != -1)
	////		{
	////			std::cout << "splitting trajectory " << trajectory_index << " at time " << min_energies_iterator->first << " into a new trajectory " << (std::prev(trajectories_.end()))->first + 1 << std::endl;
	////			PerformSplitting((int)trajectory_index, min_energies_iterator->first);
	////		}
	//		for (std::vector<std::pair<int, Real>>::iterator min_energies_iterator = min_energies.begin(); min_energies_iterator != min_energies.end(); ++min_energies_iterator)
	//		{
	//			if (min_energies_iterator->first != -1)
	//			{
	//				std::cout << "splitting trajectory " << std::distance(min_energies.begin(), min_energies_iterator) << " at time " << min_energies_iterator->first << " into a new trajectory " << (std::prev(trajectories_.end()))->first + 1 << std::endl;
	//				PerformSplitting((int)std::distance(min_energies.begin(), min_energies_iterator), min_energies_iterator->first);
	//			}
	//		}
	//
	//		//Adding
	//		//Removing
	//		last_trajectory_iterator = std::prev(trajectories_.end());
	//		min_energies = std::vector<std::pair<int, Real>>(last_trajectory_iterator->first + 1, std::pair<int, Real>(-1, std::numeric_limits<Real>::max()));
	//		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
	//		{
	//			TryRemoving(trajectory_iterator->first, min_energies[trajectory_iterator->first]);
	//		}
	//
	////		min_energies_iterator = std::min_element(min_energies.begin(), min_energies.end(), [](const std::pair<int, Real> &p1, const std::pair<int, Real> &p2) { return p1.second < p2.second; } );
	////		trajectory_index = std::distance(min_energies.begin(), min_energies_iterator);
	////		if (min_energies_iterator->first != -1)
	////		{
	////			std::cout << "removing trajectory " << trajectory_index << std::endl;
	////			PerformRemoving((int)trajectory_index);
	////		}
	//		for (std::vector<std::pair<int, Real>>::iterator min_energies_iterator = min_energies.begin(); min_energies_iterator != min_energies.end(); ++min_energies_iterator)
	//		{
	//			if (min_energies_iterator->first != -1)
	//			{
	//				std::cout << "removing trajectory " << trajectory_index << std::endl;
	//				PerformRemoving((int)min_energies_iterator->first);
	//			}
	//		}
	//
	//		//remove trajectories of the length <= 2
	//		for (std::map<int, std::vector<int>>::iterator time_iterator = trajectories_times_.begin(); time_iterator != trajectories_times_.end(); )
	//		{
	//			if ((time_iterator->second).size() <= 2)
	//			{
	//				trajectories_.erase(time_iterator->first);
	//				trajectories_times_.erase(time_iterator++);
	//			}
	//			else
	//			{
	//				++time_iterator;
	//			}
	//		}
	//
	//		PerformConjugateGradientDescent();
	//	}

	SaveTrajectories();
}

void EnergyMinimization::InitializeTrajectoriesAndDetections()
{
	std::ifstream estimated_state_file("/Users/nikita/Documents/spr/20160920/100x_10BF_1ms_v1/Pos0/output_256/KalmanFilterState.txt", std::ios::in);
	std::ifstream detection_file("/Users/nikita/Documents/spr/20160920/100x_10BF_1ms_v1/Pos0/output_256/KalmanFilterDetection.txt", std::ios::in);
	int number_of_targets = 0, number_of_detections = 0;
	int tau = 0;
	int index = 0;
	std::vector<Real> x_i(4, 0.0);

	//	estimated_state_file >> tau >> number_of_targets;
	//	for (int i = 0; i < number_of_targets; ++i)
	//	{
	//		estimated_state_file >> index >> x_i[0] >> x_i[1] >> x_i[2] >> x_i[3];
	//		if (trajectories_.find(index) == trajectories_.end())
	//		{
	//			trajectories_[index] = x_i;
	//			trajectories_times_[index] = std::vector<int>(1, tau);
	//		}
	//		else
	//		{
	//			trajectories_[index].insert(trajectories_[index].end(), x_i.begin(), x_i.end());
	//			trajectories_times_[index].push_back(tau);
	//		}
	//	}

	for (int t = 0; t < number_of_timepoints_; ++t)
	{
		estimated_state_file >> tau >> number_of_targets;

		for (int i = 0; i < number_of_targets; ++i)
		{
			estimated_state_file >> index >> x_i[0] >> x_i[1] >> x_i[2] >> x_i[3];
			if (trajectories_.find(index) == trajectories_.end())
			{
				trajectories_[index] = x_i;
				trajectories_times_[index] = std::vector<int>(1, tau);
			}
			else
			{
				trajectories_[index].insert(trajectories_[index].end(), x_i.begin(), x_i.end());
				trajectories_times_[index].push_back(tau);
			}
		}

		detection_file >> tau >> number_of_detections;
		detections_[t].resize(2 * number_of_detections);

		for (int j = 0; j < number_of_detections; ++j)
		{
			detection_file >> detections_[t][2 * j] >> detections_[t][2 * j + 1];
		}
	}

	estimated_state_file.close();
	detection_file.close();
}

Real EnergyMinimization::ComputeTotalEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times)
{
	return alpha_ * ComputeObservationModelEnergy(trajectories, trajectories_times) + beta_ * ComputeDynamicModelEnergy(trajectories, trajectories_times) + gamma_ * ComputeMutualExclusionEnergy(trajectories, trajectories_times) + delta_ * ComputeTrajectoryPersistenceEnergy(trajectories, trajectories_times);
}

Real EnergyMinimization::ComputeObservationModelEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times)
{
	Real E_det = 0.0, sum_over_detections = 0.0;
	Real lambda = 0.0, omega = 1.0;
	std::vector<Real> x_i(2, 0.0);
	std::vector<Real> d_g(2, 0.0);
	int t = 0;

	for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories.begin(); trajectory_iterator != trajectories.end(); ++trajectory_iterator)
	{
		t = 0;
		for (std::vector<int>::const_iterator trajectory_time_iterator = trajectories_times[trajectory_iterator->first].begin(); trajectory_time_iterator != trajectories_times[trajectory_iterator->first].end(); ++trajectory_time_iterator, ++t)
		{
			x_i[0] = (trajectory_iterator->second)[kNumOfStateVars * t];
			x_i[1] = (trajectory_iterator->second)[kNumOfStateVars * t + 1];

			sum_over_detections = 0.0;
			for (std::vector<Real>::const_iterator detection_iterator = detections_[*trajectory_time_iterator].begin(); detection_iterator != detections_[*trajectory_time_iterator].end(); detection_iterator += 2)
			{
				d_g[0] = *detection_iterator;
				d_g[1] = *(detection_iterator + 1);

				sum_over_detections += omega * target_size_ * target_size_ / ((x_i[0] - d_g[0]) * (x_i[0] - d_g[0]) + (x_i[1] - d_g[1]) * (x_i[1] - d_g[1]) + target_size_ * target_size_);
			}
			E_det += lambda - sum_over_detections;
		}
	}

	return E_det;
}

Real EnergyMinimization::ComputeDynamicModelEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times)
{
	Real E_dyn = 0.0;
	std::vector<Real> x_0(2, 0.0), x_1(2, 0.0), x_2(2, 0.0);
	std::vector<Real> x_dyn(2, 0.0);

	for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories.begin(); trajectory_iterator != trajectories.end(); ++trajectory_iterator)
	{
		for (int t = 0; t < int(trajectories_times[trajectory_iterator->first].size()) - 2; ++t)
		{
			x_0[0] = (trajectory_iterator->second)[kNumOfStateVars * t];
			x_0[1] = (trajectory_iterator->second)[kNumOfStateVars * t + 1];
			x_1[0] = (trajectory_iterator->second)[kNumOfStateVars * (t + 1)];
			x_1[1] = (trajectory_iterator->second)[kNumOfStateVars * (t + 1) + 1];
			x_2[0] = (trajectory_iterator->second)[kNumOfStateVars * (t + 2)];
			x_2[1] = (trajectory_iterator->second)[kNumOfStateVars * (t + 2) + 1];

			x_dyn[0] = x_0[0] - 2.0 * x_1[0] + x_2[0];
			x_dyn[1] = x_0[1] - 2.0 * x_1[1] + x_2[1];

			E_dyn += x_dyn[0] * x_dyn[0] + x_dyn[1] * x_dyn[1];
		}
	}

	return E_dyn;
}

//Real EnergyMinimization::ComputeMutualExclusionEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times)
//{
//	Real E_exc = 0.0;
//	std::vector<Real> x_i(2, 0.0), x_j(2, 0.0);
//
//	for (std::map<int, std::vector<int>>::const_iterator time_iterator_i = trajectories_times.begin(); time_iterator_i != trajectories_times.end(); ++time_iterator_i)
//	{
//		for (std::map<int, std::vector<int>>::const_iterator time_iterator_j = trajectories_times.begin(); time_iterator_j != trajectories_times.end(); ++time_iterator_j)
//		{
//			if (time_iterator_j->first != time_iterator_i->first)
//			{
//				for (int t_i = 0; t_i < (time_iterator_i->second).size(); ++t_i)
//				{
//					for (int t_j = 0; t_j < (time_iterator_j->second).size(); ++t_j)
//					{
//						if ((time_iterator_i->second)[t_i] == (time_iterator_j->second)[t_j])
//						{
//							x_i[0] = trajectories[time_iterator_i->first][kNumOfStateVars * t_i];
//							x_i[1] = trajectories[time_iterator_i->first][kNumOfStateVars * t_i + 1];
//
//							x_j[0] = trajectories[time_iterator_j->first][kNumOfStateVars * t_j];
//							x_j[1] = trajectories[time_iterator_j->first][kNumOfStateVars * t_j + 1];
//
//							E_exc += target_size_ / (x_i[0] - x_j[0]) * (x_i[0] - x_j[0]) + (x_i[1] - x_j[1]) * (x_i[1] - x_j[1]);
//						}
//					}
//				}
//			}
//		}
//	}
//
//	return E_exc;
//}
Real EnergyMinimization::ComputeMutualExclusionEnergy(std::map<int, std::vector<Real> > &trajectories, std::map<int, std::vector<int> > &trajectories_times)
{
	Real E_exc = 0.0;
	std::vector<Real> x_i(2, 0.0), x_j(2, 0.0);

	for (std::map<int, std::vector<int>>::iterator time_iterator_i = trajectories_times.begin(); time_iterator_i != trajectories_times.end(); ++time_iterator_i)
	{
		int target_id_i = time_iterator_i->first;

		for (int t_i = 0; t_i < (time_iterator_i->second).size(); ++t_i)
		{
			x_i[0] = trajectories[target_id_i][kNumOfStateVars * t_i];
			x_i[1] = trajectories[target_id_i][kNumOfStateVars * t_i + 1];

			for (std::map<int, std::vector<int>>::iterator time_iterator_j = trajectories_times.begin(); time_iterator_j != trajectories_times.end(); ++time_iterator_j)
			{
				int target_id_j = time_iterator_j->first;

				if (target_id_i != target_id_j)
				{
					for (int t_j = 0; t_j < (time_iterator_j->second).size(); ++t_j)
					{
						if ((time_iterator_i->second)[t_i] == (time_iterator_j->second)[t_j])
						{
							x_j[0] = trajectories[target_id_j][kNumOfStateVars * t_j];
							x_j[1] = trajectories[target_id_j][kNumOfStateVars * t_j + 1];

							E_exc += target_size_ * target_size_ / ((x_i[0] - x_j[0]) * (x_i[0] - x_j[0]) + (x_i[1] - x_j[1]) * (x_i[1] - x_j[1]) + target_size_ * target_size_) - 1.0;
						}
					}
				}
			}
		}
	}

	return E_exc;
}

Real EnergyMinimization::ComputeTrajectoryPersistenceEnergy(std::map<int, std::vector<Real>> &trajectories, std::map<int, std::vector<int>> &trajectories_times)
{
	Real E_per = 0.0;
	Real q = 1.0 / target_size_;
	std::vector<Real> x_i_start(2, 0.0), x_i_end(2, 0.0);
	std::vector<Real> min_border_dist(2, 0.0);
	int target_id = 0;

	for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories.begin(); trajectory_iterator != trajectories.end(); ++trajectory_iterator)
	{
		target_id = trajectory_iterator->first;

		if (trajectories_times[target_id][0] != 0)
		{
			x_i_start[0] = (trajectory_iterator->second)[0];
			x_i_start[1] = (trajectory_iterator->second)[1];

			min_border_dist[0] = std::min(x_i_start[0], image_size_x_ - x_i_start[0]);
			min_border_dist[1] = std::min(x_i_start[1], image_size_y_ - x_i_start[1]);

			//			E_per += 1.0 / (1.0 + std::exp(target_size_ / 2.0 - std::min(min_border_dist[0], min_border_dist[1])));
			E_per += 1.0 / (1.0 + std::exp(-q * std::min(min_border_dist[0], min_border_dist[1]) + 1.0));
		}

		if (trajectories_times[target_id][trajectories_times[target_id].size() - 1] != (number_of_timepoints_ - 1))
		{
			x_i_end[0] = (trajectory_iterator->second)[kNumOfStateVars * ((trajectory_iterator->second).size() / kNumOfStateVars - 1)];
			x_i_end[1] = (trajectory_iterator->second)[kNumOfStateVars * ((trajectory_iterator->second).size() / kNumOfStateVars - 1) + 1];

			min_border_dist[0] = std::min(x_i_end[0], image_size_x_ - x_i_end[0]);
			min_border_dist[1] = std::min(x_i_end[1], image_size_y_ - x_i_end[1]);

			//			E_per += 1.0 / (1.0 + std::exp(target_size_ / 2.0 - std::min(min_border_dist[0], min_border_dist[1])));
			E_per += 1.0 / (1.0 + std::exp(-q * std::min(min_border_dist[0], min_border_dist[1]) + 1.0));
		}
	}

	return E_per;
}

void EnergyMinimization::ComputeEnergyGradient()
{
	int t = 0;
	Real q = 1.0 / target_size_;
	std::vector<Real> grad_E(4, 0.0);
	std::vector<Real> x_i(4, 0.0), x_j(4, 0.0), x_i_start(2, 0.0), x_i_end(2, 0.0);
	std::vector<Real> d_g(2, 0.0);
	Real omega = 1.0;
	int target_id = 0;
	Real observation_model_denominator = 0.0, mutual_exclusion_denominator = 0.0, trajcetory_persistance_exponent = 0.0;

	//Create and insert initial elements into the energy gradient, which is of the same size as the state vector
	energy_gradient_.clear();
	for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
	{
		target_id = trajectory_iterator->first;

		energy_gradient_[target_id] = std::vector<Real>((trajectory_iterator->second).size(), 0.0);
	}

	//Observational Model Energy
	for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
	{
		target_id = trajectory_iterator->first;
		t = 0;
		for (std::vector<int>::iterator trajectory_time_iterator = trajectories_times_[target_id].begin(); trajectory_time_iterator != trajectories_times_[target_id].end(); ++trajectory_time_iterator, ++t)
		{
			x_i[0] = (trajectory_iterator->second)[kNumOfStateVars * t];
			x_i[1] = (trajectory_iterator->second)[kNumOfStateVars * t + 1];

			grad_E[0] = 0.0;
			grad_E[1] = 0.0;
			for (std::vector<Real>::iterator detection_iterator = detections_[*trajectory_time_iterator].begin(); detection_iterator != detections_[*trajectory_time_iterator].end(); detection_iterator += 2)
			{
				d_g[0] = *(detection_iterator);
				d_g[1] = *(detection_iterator + 1);

				observation_model_denominator = (x_i[0] - d_g[0]) * (x_i[0] - d_g[0]) + (x_i[1] - d_g[1]) * (x_i[1] - d_g[1]) + target_size_ * target_size_;
				observation_model_denominator *= observation_model_denominator;
				grad_E[0] += 2.0 * omega * target_size_ * target_size_ * (x_i[0] - d_g[0]) / observation_model_denominator;
				grad_E[1] += 2.0 * omega * target_size_ * target_size_ * (x_i[1] - d_g[1]) / observation_model_denominator;
			}
			grad_E[0] *= alpha_;
			grad_E[1] *= alpha_;

			energy_gradient_[target_id][kNumOfStateVars * t] += grad_E[0];
			energy_gradient_[target_id][kNumOfStateVars * t + 1] += grad_E[1];

			//			//Create and insert elements into the energy gradient
			//			if (energy_gradient_.find(target_id) == energy_gradient_.end())
			//			{
			//				energy_gradient_[target_id] = grad_E;
			//			}
			//			else
			//			{
			//				energy_gradient_[target_id].insert(energy_gradient_[target_id].end(), grad_E.begin(), grad_E.end());
			//			}
		}
	}

	//Dynamic Model Energy
	for (std::map<int, std::vector<int>>::iterator trajectory_time_iterator = trajectories_times_.begin(); trajectory_time_iterator != trajectories_times_.end(); ++trajectory_time_iterator)
	{
		target_id = trajectory_time_iterator->first;

		for (t = 0; t < int((trajectory_time_iterator->second).size()) - 2; ++t)
		{
			grad_E[0] = 2.0 * (trajectories_[target_id][kNumOfStateVars * t] - 2.0 * trajectories_[target_id][kNumOfStateVars * (t + 1)] + trajectories_[target_id][kNumOfStateVars * (t + 2)]);
			grad_E[1] = 2.0 * (trajectories_[target_id][kNumOfStateVars * t + 1] - 2.0 * trajectories_[target_id][kNumOfStateVars * (t + 1) + 1] + trajectories_[target_id][kNumOfStateVars * (t + 2) + 1]);

			grad_E[0] *= beta_;
			grad_E[1] *= beta_;

			energy_gradient_[target_id][kNumOfStateVars * t] += grad_E[0];
			energy_gradient_[target_id][kNumOfStateVars * t + 1] += grad_E[1];

			energy_gradient_[target_id][kNumOfStateVars * (t + 1)] += (-2.0) * grad_E[0];
			energy_gradient_[target_id][kNumOfStateVars * (t + 1) + 1] += (-2.0) * grad_E[1];

			energy_gradient_[target_id][kNumOfStateVars * (t + 2)] += grad_E[0];
			energy_gradient_[target_id][kNumOfStateVars * (t + 2) + 1] += grad_E[1];
		}
	}

	//Mutual Exclusion Energy
	//	for (std::map<int, std::vector<int>>::iterator time_iterator_i = trajectories_times_.begin(); time_iterator_i != trajectories_times_.end(); ++time_iterator_i)
	//	{
	//		int target_id_i = time_iterator_i->first;
	//
	//		for (int t_i = 0; t_i < (time_iterator_i->second).size(); ++t_i)
	//		{
	//			x_i[0] = trajectories_[target_id_i][kNumOfStateVars * t_i];
	//			x_i[1] = trajectories_[target_id_i][kNumOfStateVars * t_i + 1];
	//
	//			grad_E[0] = 0.0;
	//			grad_E[1] = 0.0;
	//			for (std::map<int, std::vector<int>>::iterator time_iterator_j = trajectories_times_.begin(); time_iterator_j != trajectories_times_.end(); ++time_iterator_j)
	//			{
	//				int target_id_j = time_iterator_j->first;
	//
	//				if (target_id_i != target_id_j)
	//				{
	//					for (int t_j = 0; t_j < (time_iterator_j->second).size(); ++t_j)
	//					{
	//						if ((time_iterator_i->second)[t_i] == (time_iterator_j->second)[t_j])
	//						{
	//							x_j[0] = trajectories_[target_id_j][kNumOfStateVars * t_j];
	//							x_j[1] = trajectories_[target_id_j][kNumOfStateVars * t_j + 1];
	//
	//							mutual_exclusion_denominator = (x_i[0] - x_j[0]) * (x_i[0] - x_j[0]) + (x_i[1] - x_j[1]) * (x_i[1] - x_j[1]);
	//							mutual_exclusion_denominator *= mutual_exclusion_denominator;
	//							grad_E[0] -= 4.0 * target_size_ * (x_i[0] - x_j[0]) / mutual_exclusion_denominator;
	//							grad_E[1] -= 4.0 * target_size_ * (x_i[1] - x_j[1]) / mutual_exclusion_denominator;
	//						}
	//					}
	//				}
	//			}
	//			grad_E[0] *= gamma_;
	//			grad_E[1] *= gamma_;
	//
	//			energy_gradient_[target_id_i][kNumOfStateVars * t_i] += grad_E[0];
	//			energy_gradient_[target_id_i][kNumOfStateVars * t_i + 1] += grad_E[1];
	//		}
	//	}
	for (std::map<int, std::vector<int>>::iterator time_iterator_i = trajectories_times_.begin(); time_iterator_i != trajectories_times_.end(); ++time_iterator_i)
	{
		int target_id_i = time_iterator_i->first;

		for (int t_i = 0; t_i < (time_iterator_i->second).size(); ++t_i)
		{
			x_i[0] = trajectories_[target_id_i][kNumOfStateVars * t_i];
			x_i[1] = trajectories_[target_id_i][kNumOfStateVars * t_i + 1];

			grad_E[0] = 0.0;
			grad_E[1] = 0.0;
			for (std::map<int, std::vector<int>>::iterator time_iterator_j = trajectories_times_.begin(); time_iterator_j != trajectories_times_.end(); ++time_iterator_j)
			{
				int target_id_j = time_iterator_j->first;

				if (target_id_i != target_id_j)
				{
					for (int t_j = 0; t_j < (time_iterator_j->second).size(); ++t_j)
					{
						if ((time_iterator_i->second)[t_i] == (time_iterator_j->second)[t_j])
						{
							x_j[0] = trajectories_[target_id_j][kNumOfStateVars * t_j];
							x_j[1] = trajectories_[target_id_j][kNumOfStateVars * t_j + 1];

							mutual_exclusion_denominator = (x_i[0] - x_j[0]) * (x_i[0] - x_j[0]) + (x_i[1] - x_j[1]) * (x_i[1] - x_j[1]) + target_size_ * target_size_;
							mutual_exclusion_denominator *= mutual_exclusion_denominator;
							grad_E[0] -= 4.0 * target_size_ * target_size_ * (x_i[0] - x_j[0]) / mutual_exclusion_denominator;
							grad_E[1] -= 4.0 * target_size_ * target_size_ * (x_i[1] - x_j[1]) / mutual_exclusion_denominator;
						}
					}
				}
			}
			grad_E[0] *= gamma_;
			grad_E[1] *= gamma_;

			energy_gradient_[target_id_i][kNumOfStateVars * t_i] += grad_E[0];
			energy_gradient_[target_id_i][kNumOfStateVars * t_i + 1] += grad_E[1];
		}
	}

	//Trajectory Persistence Energy
	std::vector<Real> min_border_dist(4, 0.0), min_border_dist_grad_x = { 1.0, -1.0, 0.0, 0.0 }, min_border_dist_grad_y = { 0.0, 0.0, 1.0, -1.0 };
	size_t i_min = 0;
	for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
	{
		target_id = trajectory_iterator->first;

		if (trajectories_times_[target_id][0] != 0)
		{
			x_i_start[0] = (trajectory_iterator->second)[0];
			x_i_start[1] = (trajectory_iterator->second)[1];

			min_border_dist[0] = x_i_start[0];
			min_border_dist[1] = image_size_x_ - x_i_start[0];
			min_border_dist[2] = x_i_start[1];
			min_border_dist[3] = image_size_y_ - x_i_start[1];

			i_min = std::distance(min_border_dist.begin(), std::min_element(min_border_dist.begin(), min_border_dist.end()));

			trajcetory_persistance_exponent = std::exp(target_size_ / 2.0 - min_border_dist[i_min]);
			//			grad_E[0] = trajcetory_persistance_exponent / (1.0 + trajcetory_persistance_exponent) / (1.0 + trajcetory_persistance_exponent) * min_border_dist_grad_x[i_min];
			//			grad_E[1] = trajcetory_persistance_exponent / (1.0 + trajcetory_persistance_exponent) / (1.0 + trajcetory_persistance_exponent) * min_border_dist_grad_y[i_min];
			grad_E[0] = q * std::exp(-q * min_border_dist[i_min] + 1.0) / (1.0 + std::exp(-q * min_border_dist[i_min] + 1.0)) / (1.0 * std::exp(-q * min_border_dist[i_min] + 1.0)) * min_border_dist_grad_x[i_min];
			grad_E[1] = q * std::exp(-q * min_border_dist[i_min] + 1.0) / (1.0 + std::exp(-q * min_border_dist[i_min] + 1.0)) / (1.0 * std::exp(-q * min_border_dist[i_min] + 1.0)) * min_border_dist_grad_y[i_min];

			energy_gradient_[target_id][0] += grad_E[0] * delta_;
			energy_gradient_[target_id][1] += grad_E[1] * delta_;
		}

		if (trajectories_times_[target_id][trajectories_times_[target_id].size() - 1] != (number_of_timepoints_ - 1))
		{
			x_i_end[0] = (trajectory_iterator->second)[kNumOfStateVars * ((trajectory_iterator->second).size() / kNumOfStateVars - 1)];
			x_i_end[1] = (trajectory_iterator->second)[kNumOfStateVars * ((trajectory_iterator->second).size() / kNumOfStateVars - 1) + 1];

			min_border_dist[0] = x_i_end[0];
			min_border_dist[1] = image_size_x_ - x_i_end[0];
			min_border_dist[2] = x_i_end[1];
			min_border_dist[3] = image_size_y_ - x_i_end[1];

			i_min = std::distance(min_border_dist.begin(), std::min_element(min_border_dist.begin(), min_border_dist.end()));

			trajcetory_persistance_exponent = std::exp(target_size_ / 2.0 - min_border_dist[i_min]);
			//			grad_E[0] = trajcetory_persistance_exponent / (1.0 + trajcetory_persistance_exponent) / (1.0 + trajcetory_persistance_exponent) * min_border_dist_grad_x[i_min];
			//			grad_E[1] = trajcetory_persistance_exponent / (1.0 + trajcetory_persistance_exponent) / (1.0 + trajcetory_persistance_exponent) * min_border_dist_grad_y[i_min];
			grad_E[0] = q * std::exp(-q * min_border_dist[i_min] + 1.0) / (1.0 + std::exp(-q * min_border_dist[i_min] + 1.0)) / (1.0 + std::exp(-q * min_border_dist[i_min] + 1.0)) * min_border_dist_grad_x[i_min];
			grad_E[1] = q * std::exp(-q * min_border_dist[i_min] + 1.0) / (1.0 + std::exp(-q * min_border_dist[i_min] + 1.0)) / (1.0 + std::exp(-q * min_border_dist[i_min] + 1.0)) * min_border_dist_grad_y[i_min];

			energy_gradient_[target_id][kNumOfStateVars * ((trajectory_iterator->second).size() / kNumOfStateVars - 1)] += grad_E[0] * delta_;
			energy_gradient_[target_id][kNumOfStateVars * ((trajectory_iterator->second).size() / kNumOfStateVars - 1) + 1] += grad_E[1] * delta_;
		}
	}
}

Real EnergyMinimization::ComputeEnergyGradientNorm()
{
	Real norm = 0.0;

	for (std::map<int, std::vector<Real>>::iterator energy_gradient_iterator = energy_gradient_.begin(); energy_gradient_iterator != energy_gradient_.end(); ++energy_gradient_iterator)
	{
		for (std::vector<Real>::iterator trajectory_iterator = (energy_gradient_iterator->second).begin(); trajectory_iterator != (energy_gradient_iterator->second).end(); ++trajectory_iterator)
		{
			norm += (*trajectory_iterator) * (*trajectory_iterator);
		}
	}

	return std::sqrt(norm);
}

Real EnergyMinimization::PerformBacktrackingArmijoLineSearch()
{
	Real lambda = 1000.0, epsilon = 0.001, tau = 0.5;
	Real lhs = 0.0, rhs = 0.0;

	do
	{
		//X^k - \lambda^l \nabla E(X^k) = X^k + \lambda^l d^k
		std::map<int, std::vector<Real>> temp_traj_0(trajectories_);
		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = temp_traj_0.begin(); trajectory_iterator != temp_traj_0.end(); ++trajectory_iterator)
		{
			int trajectory_index = trajectory_iterator->first;
			for (int i = 0; i < (trajectory_iterator->second).size(); ++i)
			{
				(trajectory_iterator->second)[i] -= lambda * energy_gradient_[trajectory_index][i];
			}
		}

		//E(X^k - \lambda^l \nabla E(X^k))
		lhs = ComputeTotalEnergy(temp_traj_0, trajectories_times_);
		//E(X^k) - \lambda^l\epsilon \nabla_{X^k} E(X^k)\cdot\nabla E(X^k)
		rhs = ComputeTotalEnergy(trajectories_, trajectories_times_);
		for (std::map<int, std::vector<Real>>::iterator energy_gradient_iterator = energy_gradient_.begin(); energy_gradient_iterator != energy_gradient_.end(); ++energy_gradient_iterator)
		{
			for (int i = 0; i < (energy_gradient_iterator->second).size(); ++i)
			{
				rhs -= lambda * epsilon * (energy_gradient_iterator->second)[i] * (energy_gradient_iterator->second)[i];
			}
		}

		lambda *= tau;
	} while (lhs > rhs);//(lhs - rhs > 0.001)

	return lambda / tau;
}

void EnergyMinimization::PerformConjugateGradientDescent()
{
	Real epsilon = 0.1;
	Real lambda = 0.0;

	ComputeEnergyGradient();
	Real energy_gradient_norm = ComputeEnergyGradientNorm();
	while (energy_gradient_norm > epsilon)
	{
		lambda = PerformBacktrackingArmijoLineSearch();
		assert(lambda <= 1000.0);

		//X^{k+1} = X^k - \lambda^k \nabla E(X^k)
		for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator)
		{
			for (int i = 0; i < (trajectory_iterator->second).size(); ++i)
			{
				(trajectory_iterator->second)[i] -= lambda * energy_gradient_[trajectory_iterator->first][i];
			}
		}

		ComputeEnergyGradient();
		energy_gradient_norm = ComputeEnergyGradientNorm();
	}
}

void EnergyMinimization::SaveTrajectories()
{
	std::ofstream optimized_state_file("/Users/nikita/Documents/spr/20160920/100x_10BF_1ms_v1/Pos0/output_256/OptimizedState.txt", std::ios::out | std::ios::trunc);

	int trajectory_index = 0;
	for (int t = 0; t < number_of_timepoints_; ++t)
	{
		for (std::map<int, std::vector<int>>::iterator trajectory_time_iterator = trajectories_times_.begin(); trajectory_time_iterator != trajectories_times_.end(); ++trajectory_time_iterator)
		{
			trajectory_index = trajectory_time_iterator->first;

			int time_index = 0;
			for (std::vector<int>::iterator times_iterator = (trajectory_time_iterator->second).begin(); times_iterator != (trajectory_time_iterator->second).end(); ++times_iterator, ++time_index)
			{
				if (*times_iterator == t)
				{
					optimized_state_file << t << " " << trajectory_index << " " << trajectories_[trajectory_index][kNumOfStateVars * time_index] << " " << trajectories_[trajectory_index][kNumOfStateVars * time_index + 1] << " " << trajectories_[trajectory_index][kNumOfStateVars * time_index + 2] << " " << trajectories_[trajectory_index][kNumOfStateVars * time_index + 3] << std::endl;
					break;
				}
			}
		}
	}

	optimized_state_file.close();
}

void EnergyMinimization::TryMerging(int i, std::pair<int, Real> &index_energy)
{
	Real E_new = 0.0, E_prev = 0.0;
	E_prev = ComputeTotalEnergy(trajectories_, trajectories_times_);
	std::vector<Real> min_energies(trajectories_.size(), std::numeric_limits<Real>::max());
	bool perform_jump_move = false;

	size_t j = 0;
	for (std::map<int, std::vector<Real>>::iterator trajectory_iterator = trajectories_.begin(); trajectory_iterator != trajectories_.end(); ++trajectory_iterator, ++j)
	{
		if ((i != trajectory_iterator->first) && (*(std::prev(trajectories_times_[i].end())) + 1 == trajectories_times_[trajectory_iterator->first][0]))
		{
			std::map<int, std::vector<Real>> new_trajectories(trajectories_);
			std::map<int, std::vector<int>> new_trajectories_times(trajectories_times_);

			new_trajectories[i].insert(new_trajectories[i].end(), (trajectory_iterator->second).begin(), (trajectory_iterator->second).end());
			new_trajectories.erase(trajectory_iterator->first);

			new_trajectories_times[i].insert(new_trajectories_times[i].end(), trajectories_times_[trajectory_iterator->first].begin(), trajectories_times_[trajectory_iterator->first].end());
			new_trajectories_times.erase(trajectory_iterator->first);

			E_new = ComputeTotalEnergy(new_trajectories, new_trajectories_times);
			if (E_prev - E_new > delta_E_crit_)
				//			if (E_new < E_prev)
			{
				min_energies[j] = E_new;
				perform_jump_move = true;
			}
		}
	}

	if (perform_jump_move)
	{
		std::vector<Real>::iterator min_energy_iterator = std::min_element(min_energies.begin(), min_energies.end());
		int j = int(std::distance(min_energies.begin(), min_energy_iterator));
		std::map<int, std::vector<Real>>::iterator trajectory_j = trajectories_.begin();
		std::advance(trajectory_j, j);

		index_energy.first = trajectory_j->first;
		index_energy.second = *min_energy_iterator;
	}
	else
	{
		index_energy.first = -1;
		index_energy.second = std::numeric_limits<Real>::max();
	}
}

void EnergyMinimization::PerformMerging(int i, int j)
{
	trajectories_[i].insert(trajectories_[i].end(), trajectories_[j].begin(), trajectories_[j].end());
	trajectories_.erase(j);

	trajectories_times_[i].insert(trajectories_times_[i].end(), trajectories_times_[j].begin(), trajectories_times_[j].end());
	trajectories_times_.erase(j);
}

void EnergyMinimization::TrySplitting(int i, std::pair<int, Real> &time_energy)
{
	Real E_new = 0.0, E_prev = 0.0;
	E_prev = ComputeTotalEnergy(trajectories_, trajectories_times_);
	std::vector<Real> min_energies(trajectories_times_[i].size(), std::numeric_limits<Real>::max());
	bool perform_jump_move = false;

	for (int t = 2; t < int(trajectories_times_[i].size()) - 3; ++t)
		//	for (int t = 0; t < trajectories_times_[i].size() - 1; ++t)
	{
		std::map<int, std::vector<Real>> new_trajectories(trajectories_);
		std::map<int, std::vector<int>> new_trajectories_times(trajectories_times_);
		std::map<int, std::vector<Real>>::iterator last_it = std::prev(new_trajectories.end());

		new_trajectories[last_it->first + 1] = std::vector<Real>(new_trajectories[i].begin() + kNumOfStateVars * (t + 1), new_trajectories[i].end());
		new_trajectories[i].erase(new_trajectories[i].begin() + kNumOfStateVars * (t + 1), new_trajectories[i].end());

		new_trajectories_times[last_it->first + 1] = std::vector<int>(new_trajectories_times[i].begin() + t + 1, new_trajectories_times[i].end());
		new_trajectories_times[i].erase(new_trajectories_times[i].begin() + t + 1, new_trajectories_times[i].end());

		E_new = ComputeTotalEnergy(new_trajectories, new_trajectories_times);
		if (E_prev - E_new > delta_E_crit_)
			//		if (E_new < E_prev)
		{
			min_energies[t] = E_new;
			perform_jump_move = true;
		}
	}

	if (perform_jump_move)
	{
		std::vector<Real>::iterator min_energy_iterator = std::min_element(min_energies.begin(), min_energies.end());
		int t = int(std::distance(min_energies.begin(), min_energy_iterator));

		time_energy.first = t;
		time_energy.second = *min_energy_iterator;
	}
	else
	{
		time_energy.first = -1;
		time_energy.second = std::numeric_limits<Real>::max();
	}
}

void EnergyMinimization::PerformSplitting(int i, int t)
{
	std::map<int, std::vector<Real>>::iterator last_it = std::prev(trajectories_.end());

	trajectories_[last_it->first + 1] = std::vector<Real>(trajectories_[i].begin() + kNumOfStateVars * (t + 1), trajectories_[i].end());
	trajectories_[i].erase(trajectories_[i].begin() + kNumOfStateVars * (t + 1), trajectories_[i].end());

	trajectories_times_[last_it->first + 1] = std::vector<int>(trajectories_times_[i].begin() + t + 1, trajectories_times_[i].end());
	trajectories_times_[i].erase(trajectories_times_[i].begin() + t + 1, trajectories_times_[i].end());
}

void EnergyMinimization::TryRemoving(int i, std::pair<int, Real> &index_energy)
{
	Real E_new = 0.0, E_prev = 0.0;
	E_prev = ComputeTotalEnergy(trajectories_, trajectories_times_);

	std::map<int, std::vector<Real>> new_trajectories(trajectories_);
	std::map<int, std::vector<int>> new_trajectories_times(trajectories_times_);

	new_trajectories.erase(i);
	new_trajectories_times.erase(i);

	E_new = ComputeTotalEnergy(new_trajectories, new_trajectories_times);
	if (E_prev - E_new > delta_E_crit_)
		//	if (E_new < E_prev)
	{
		index_energy.first = i;
		index_energy.second = E_new;
	}
	else
	{
		index_energy.first = -1;
		index_energy.second = std::numeric_limits<Real>::max();
	}
}

void EnergyMinimization::PerformRemoving(int i)
{
	trajectories_.erase(i);
	trajectories_times_.erase(i);
}

void EnergyMinimization::TryShrinking(int i, std::pair<int, Real> &index_energy, bool forward)
{
	Real E_new = 0.0, E_prev = 0.0;
	E_prev = ComputeTotalEnergy(trajectories_, trajectories_times_);

	std::map<int, std::vector<Real>> new_trajectories(trajectories_);
	std::map<int, std::vector<int>> new_trajectories_times(trajectories_times_);

	if (forward)
	{
		new_trajectories[i].erase(new_trajectories[i].begin() + kNumOfStateVars * (new_trajectories[i].size() - kNumOfStateVars), new_trajectories[i].end());
		new_trajectories_times[i].erase(std::prev(new_trajectories_times[i].end()));
	}
	else
	{
		new_trajectories[i].erase(new_trajectories[i].begin(), new_trajectories[i].begin() + kNumOfStateVars);
		new_trajectories_times[i].erase(new_trajectories_times[i].begin());
	}

	if (new_trajectories[i].empty())
	{
		new_trajectories.erase(i);
		new_trajectories_times.erase(i);
	}

	E_new = ComputeTotalEnergy(new_trajectories, new_trajectories_times);
	if (E_prev - E_new > delta_E_crit_)
		//	if (E_new < E_prev)
	{
		index_energy.first = i;
		index_energy.second = E_new;
	}
	else
	{
		index_energy.first = -1;
		index_energy.second = std::numeric_limits<Real>::max();
	}
}

void EnergyMinimization::PerformShrinking(int i, bool forward)
{
	if (forward)
	{
		trajectories_[i].erase(trajectories_[i].begin() + kNumOfStateVars * (trajectories_[i].size() - kNumOfStateVars), trajectories_[i].end());
		trajectories_times_[i].erase(std::prev(trajectories_times_[i].end()));
	}
	else
	{
		trajectories_[i].erase(trajectories_[i].begin(), trajectories_[i].begin() + kNumOfStateVars);
		trajectories_times_[i].erase(trajectories_times_[i].begin());
	}

	if (trajectories_[i].empty())
	{
		trajectories_.erase(i);
		trajectories_times_.erase(i);
	}
}