#include "Tracking/MultiTargetTracker.hpp"

int main(int argc, const char *argv[])
{
  MultitargetTracker multitarget_tracker;
//  std::string configuration_file_name("/Users/nikita/CLionProjects/sprMultiTargetTracking/Parameters/ConfigExperimental.cfg");
//  multitarget_tracker.PerformTrackingForOneExperiment(configuration_file_name);
//  multitarget_tracker.PerformOnlyTrackLinkingForOneExperiment(configuration_file_name);
  multitarget_tracker.PerformActionForMultipleExperiments(0, std::string("/Volumes/Kruk/Swarming/"));

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