#include "Tracking/MultiTargetTracker.hpp"

int main(int argc, const char *argv[])
{
  MultitargetTracker multitarget_tracker;
  std::string experimental_configuration("/Volumes/Kruk/Swarming/20170705/100x_01-BF0_1to1600_yepd_1ms_6.2nm_noautofocus_v_2/ConfigExperimental.cfg");
//  multitarget_tracker.PerformImageProcessingForOneExperiment(experimental_configuration);
//  multitarget_tracker.StartFilteringWithoutImageProcessingForOneExperiment(experimental_configuration);
  multitarget_tracker.StartTrackLinkingViaTemporalAssignment(experimental_configuration);

  // FOR IMAGE PROCESSING
//  multitarget_tracker.StartImageProcessingOrFilteringForMultipleExperiments('1');
  // FOR TRACKING & KALMAN FILTERING
  //multitarget_tracker.StartImageProcessingOrFilteringForMultipleExperiments('2');

  // FOR TRACK LINKING
//  multitarget_tracker.StartImageProcessingOrFilteringForMultipleExperiments('3');

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