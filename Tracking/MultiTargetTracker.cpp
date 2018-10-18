//
// Created by Nikita Kruk on 27.11.17.
//

#include "MultiTargetTracker.hpp"
#include "../Parameters/ParameterHandlerExperimental.hpp"
#include "../Parameters/ParameterHandlerSynthetic.hpp"
#include "../ImageProcessing/ImageProcessingEngine.hpp"
#include "KalmanFilterExperimental.hpp"
#include "KalmanFilterSynthetic.hpp"
#include "../Parameters/PeriodicBoundaryConditionsConfiguration.hpp"
#include "TrajectoryLinker.hpp"

#include <chrono>
#include <regex>

#include <boost/filesystem.hpp>

MultitargetTracker::MultitargetTracker() :
    targets_(),
    detections_(),
    trajectories_(),
    timestamps_()
{
  std::cout << "multi-target tracking started" << std::endl;
}

MultitargetTracker::~MultitargetTracker()
{
  std::cout << "multi-target tracking ended" << std::endl;
}

/**
 * Tracking includes Image processing, Filtering, and Track linking
 * @param configuration_file_name
 */
void MultitargetTracker::PerformTrackingForOneExperiment(const std::string &configuration_file_name)
{
  ParameterHandlerExperimental parameter_handler(configuration_file_name);
  ImageProcessingEngine image_processing_engine(parameter_handler);
  image_processing_engine.CreateImageProcessingOutputFile();
  KalmanFilterExperimental kalman_filter(parameter_handler, image_processing_engine);
  kalman_filter.CreateKalmanFilterOutputFiles();

  image_processing_engine.RetrieveBacterialData(parameter_handler.GetFirstImage(), detections_);
  kalman_filter.InitializeTargets(targets_, detections_);
  for (int i = parameter_handler.GetFirstImage() + 1; i <= parameter_handler.GetLastImage(); ++i)
  {
//    std::chrono::time_point<std::chrono::system_clock> timer = std::chrono::system_clock::now();
    image_processing_engine.RetrieveBacterialData(i, detections_);
    kalman_filter.PerformEstimation(i, targets_, detections_);
//    std::chrono::duration<Real> elapsed_seconds = std::chrono::system_clock::now() - timer;
//    std::cout << elapsed_seconds.count() << "s" << std::endl;
  }

  TrajectoryLinker trajectory_linker(parameter_handler, image_processing_engine);
  trajectory_linker.CreateTrackLinkingOutputFiles();
  trajectory_linker.InitializeTrajectories(trajectories_, timestamps_);
  trajectory_linker.PerformTrackLinking(trajectories_, timestamps_);
}

void MultitargetTracker::PerformOnlyImageProcessingForOneExperiment(const std::string &configuration_file_name)
{
  ParameterHandlerExperimental parameter_handler(configuration_file_name);
  ImageProcessingEngine image_processing_engine(parameter_handler);
  image_processing_engine.CreateImageProcessingOutputFile();

  for (int i = parameter_handler.GetFirstImage(); i <= parameter_handler.GetLastImage(); ++i)
  {
    image_processing_engine.RetrieveBacterialData(i, detections_);
  }
}

void MultitargetTracker::PerformOnlyFilteringForOneExperiment(const std::string &configuration_file_name)
{
  ParameterHandlerExperimental parameter_handler(configuration_file_name);
  std::ostringstream image_processing_data_input_file_name;
  image_processing_data_input_file_name << parameter_handler.GetInputFolder()
                                        << parameter_handler.GetDataAnalysisSubfolder()
                                        << parameter_handler.GetImageProcessingOutputFileName();
  std::ifstream image_processing_input_file(image_processing_data_input_file_name.str(), std::ios::in);
  assert(image_processing_input_file.is_open());

  ImageProcessingEngine image_processing_engine(parameter_handler);
  KalmanFilterExperimental kalman_filter(parameter_handler, image_processing_engine);
  kalman_filter.CreateKalmanFilterOutputFiles();
  kalman_filter.InitializeTargets(targets_, image_processing_input_file);

  for (int i = parameter_handler.GetFirstImage() + 1; i <= parameter_handler.GetLastImage(); ++i)
  {
    kalman_filter.ObtainNewDetections(detections_, image_processing_input_file);
    kalman_filter.PerformEstimation(i, targets_, detections_);
  }
}

void MultitargetTracker::PerformOnlyTrackLinkingForOneExperiment(const std::string &configuration_file_name)
{
  ParameterHandlerExperimental parameter_handler(configuration_file_name);
  ImageProcessingEngine image_processing_engine(parameter_handler);
  TrajectoryLinker trajectory_linker(parameter_handler, image_processing_engine);
  trajectory_linker.CreateTrackLinkingOutputFiles();
  trajectory_linker.InitializeTrajectories(trajectories_, timestamps_);
  trajectory_linker.PerformTrackLinking(trajectories_, timestamps_);
}

void MultitargetTracker::PerformActionForMultipleExperiments(int action, const std::string &experiments_directory)
{
  boost::filesystem::path experiments_path(experiments_directory);
  if (boost::filesystem::exists(experiments_path))
  {
    for (boost::filesystem::directory_entry &date : boost::filesystem::directory_iterator(experiments_path))
    {
      // consider only folders of a specific date-of-experiment type
      if (boost::filesystem::is_directory(date.path())
          && std::regex_match(date.path().filename().string(), std::regex("[[:d:]]+")))
      {
        for (boost::filesystem::directory_entry &experiment : boost::filesystem::directory_iterator(date.path()))
        {
          // ignore various hidden files
          if (boost::filesystem::is_directory(experiment.path()))
          {
            std::cout << experiment.path().string() << std::endl;
            std::string configuration_file_name(experiment.path().string() + std::string("/ConfigExperimental.cfg"));
            PerformTrackingForOneExperiment(configuration_file_name);
          }
        }
      }
    }
  } else
  {
    std::cout << "error: directory does not exist" << std::endl;
  }
}

void MultitargetTracker::StartOnSyntheticData(Real phi, Real a, Real U0, Real kappa, Real percentage_of_misdetections)
{
  ParameterHandlerSynthetic parameter_handler;
  parameter_handler.SetNewParametersFromSimulation(phi, a, U0, kappa);
  parameter_handler.SetPercentageOfMisdetections(percentage_of_misdetections);
  std::ostringstream modified_solution_file_name_buffer;
  modified_solution_file_name_buffer << parameter_handler.GetTrackingFolder()
                                     << parameter_handler.GetNumberOfOriginalTargets()
                                     << "/modified_solution_phi_" << parameter_handler.GetSimulationParameter("phi")
                                     << "_a_" << parameter_handler.GetSimulationParameter("a") << "_U0_"
                                     << parameter_handler.GetSimulationParameter("U_0") << "_k_"
                                     << parameter_handler.GetSimulationParameter("kappa") << "_pom_"
                                     << percentage_of_misdetections << ".txt";
  std::ifstream modified_solution_file(modified_solution_file_name_buffer.str(), std::ios::in);
  assert(modified_solution_file.is_open());

  PeriodicBoundaryConditionsConfiguration
      pbc_config(parameter_handler.GetSubimageXSize(), parameter_handler.GetSubimageYSize());
  KalmanFilterSynthetic kalman_filter(parameter_handler, pbc_config);
  kalman_filter.InitializeTargets(targets_, modified_solution_file);
  for (int i = parameter_handler.GetFirstImage() + 1; i <= parameter_handler.GetLastImage(); ++i)
  {
    kalman_filter.ObtainNewDetections(detections_, modified_solution_file);
    kalman_filter.PerformEstimation(i, targets_, detections_);
  }
}

void MultitargetTracker::StartOnSyntheticDataForDifferentParameters()
{
  const Real U0 = 250.0;
  const Real kappa = 0.0;
  const Real a = 4.0;
  const Real phi = 0.1;
  for (int percentage = 0; percentage <= 50; ++percentage)
  {
    StartOnSyntheticData(phi, a, U0, kappa, percentage * 0.01f);
  }
}