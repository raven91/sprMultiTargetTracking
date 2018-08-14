//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_MULTITARGETTRACKER_HPP
#define SPRMULTITARGETTRACKING_MULTITARGETTRACKER_HPP

#include "../Definitions.hpp"
#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

#include <vector>
#include <map>

#include <eigen3/Eigen/Dense>


class MultitargetTracker
{
public:

    MultitargetTracker();
    ~MultitargetTracker();

    void StartOnExperimentalData();
	void PerformImageProcessingForOneExperiment(const std::string & file_name);
	void PerformImageProcessingForMultipleExperiments();
	void StartOnExperimentalDataWithoutFiltering();
	void StartOnReceivedDataWithoutImageProcessing();
    void StartOnSyntheticData(Real phi, Real a, Real U0, Real kappa, Real percentage_of_misdetections);
    void StartOnSyntheticDataForDifferentParameters();

private:

    std::map<int, Eigen::VectorXf> targets_;    // i -> x_i y_i v_x_i v_y_i area_i slope_i width_i height_i
    std::vector<Eigen::VectorXf> detections_;   // observations

};

#endif //SPRMULTITARGETTRACKING_MULTITARGETTRACKER_HPP
