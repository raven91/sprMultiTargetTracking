//
// Created by Nikita Kruk on 23.07.18.
//

#ifndef SPRMULTITARGETTRACKING_STATISTICS_HPP
#define SPRMULTITARGETTRACKING_STATISTICS_HPP

#include "../Definitions.hpp"

#include <vector>
#include <fstream>

class TrackingStatistics
{
public:

	TrackingStatistics();
	~TrackingStatistics();

	int GetNumberOfSuspendedTargets();
	int GetNumberOfMissedTargets();
	int GetNumberOfDeletedTargets();
	int GetNumberOfNewTargets();
	int GetNumberOfRecapturedTargets();
	int GetNumberOfTimePoints();

	void IncrementNumberOfSuspendedTargets(int increment);
	void IncrementNumberOfMissedTargets(int increment);
	void IncrementNumberOfDeletedTargets(int increment);
	void IncrementNumberOfNewTargets(int increment);
	void IncrementNumberOfRecapturedTargets(int increment);
	void IncrementNumberOfTimePoints(int increment);

private:

	int number_of_suspended_targets_;
	int number_of_missed_targets_;
	int number_of_deleted_targets_;
	int number_of_new_targets_;
	int number_of_recaptured_targets_;
	int number_of_time_points_;

};

#endif //SPRMULTITARGETTRACKING_STATISTICS_HPP
