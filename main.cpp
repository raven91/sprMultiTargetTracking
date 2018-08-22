#include "Tracking/MultiTargetTracker.hpp"

int main(int argc, const char *argv[])
{
	//  const Real phi = 0.1, a = 4.0, U0 = 250.0, kappa = 0.0, percentage_of_misdetections = 0.1;
	MultitargetTracker multitarget_tracker;
	//multitarget_tracker.StartOnExperimentalData(); 

	// FOR IMAGE PROCESSING
	//multitarget_tracker.StartImageProcessingORTrackingAndFilteringForMultipleExperiments('1'); 

	// FOR TRACKING & KALMAN FILTERING
	//multitarget_tracker.StartImageProcessingORTrackingAndFilteringForMultipleExperiments('2');

	// FOR TRACK LINKING
	multitarget_tracker.StartImageProcessingORTrackingAndFilteringForMultipleExperiments('3');


	//system("pause")
	;
	//  multitarget_tracker.StartOnSyntheticData(phi, a, U0, kappa, percentage_of_misdetections);
	//  multitarget_tracker.StartOnSyntheticDataForDifferentParameters();

	return 0;
}

//void TestHungarianAlgorithm()
//{
//  std::vector<std::vector<int>> cost(3, std::vector<int>(3, 0));
//  cost[0][0] = 7;
//  cost[0][1] = 4;
//  cost[0][2] = 3;
//  cost[1][0] = 3;
//  cost[1][1] = 1;
//  cost[1][2] = 2;
//  cost[2][0] = 3;
//  cost[2][1] = 0;
//  cost[2][2] = 0;
//  HungarianAlgorithm ha(3, cost);
//  std::vector<int> assgn(3, 0), costs(3, 0);
//  ha.Start(assgn, costs);
//}