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

void MultitargetTracker::StartOnExperimentalData()
{
  ParameterHandlerExperimental parameter_handler
      (std::string("/Users/nikita/CLionProjects/sprMultiTargetTracking/Parameters/ConfigExperimental.cfg"));
  ImageProcessingEngine image_processing_engine(parameter_handler);
  image_processing_engine.CreateNewImageProcessingOutputFile(parameter_handler);
  KalmanFilterExperimental kalman_filter(parameter_handler, image_processing_engine);
  kalman_filter.CreateNewKalmanFilterOutputFiles(parameter_handler);

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
}

void MultitargetTracker::PerformImageProcessingForOneExperiment(const std::string &configuration_file_name)
{
  ParameterHandlerExperimental parameter_handler(configuration_file_name);
  ImageProcessingEngine image_processing_engine(parameter_handler);
  image_processing_engine.CreateNewImageProcessingOutputFile(parameter_handler);

  for (int i = parameter_handler.GetFirstImage(); i <= parameter_handler.GetLastImage(); ++i)
  {
    image_processing_engine.RetrieveBacterialData(i, detections_);
  }
}

void MultitargetTracker::StartFilteringWithoutImageProcessingForOneExperiment(const std::string &configuration_file_name)
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
  kalman_filter.CreateNewKalmanFilterOutputFiles(parameter_handler);
  kalman_filter.InitializeTargets(targets_, image_processing_input_file);

  for (int i = parameter_handler.GetFirstImage() + 1; i <= parameter_handler.GetLastImage(); ++i)
  {
    kalman_filter.ObtainNewDetections(detections_, image_processing_input_file);
    kalman_filter.PerformEstimation(i, targets_, detections_);
  }
}

void MultitargetTracker::StartTrackLinkingViaTemporalAssignment(const std::string &configuration_file_name)
{
  ParameterHandlerExperimental parameter_handler(configuration_file_name);
  std::ostringstream filtered_trajectories_file_name_buffer;
  filtered_trajectories_file_name_buffer << parameter_handler.GetInputFolder()
                                         << parameter_handler.GetDataAnalysisSubfolder()
                                         << parameter_handler.GetKalmanFilterOutputFileName();
  std::ifstream filtered_trajectories_file(filtered_trajectories_file_name_buffer.str(), std::ios::in);
  assert(filtered_trajectories_file.is_open());

  ImageProcessingEngine image_processing_engine(parameter_handler);
  KalmanFilterExperimental kalman_filter(parameter_handler, image_processing_engine);
  TrajectoryLinker trajectory_linker(parameter_handler, image_processing_engine);
  trajectory_linker.CreateNewTrackLinkingOutputFiles(parameter_handler);
  trajectory_linker.InitializeTrajectories(trajectories_, timestamps_, filtered_trajectories_file);
  trajectory_linker.PerformTrackLinking(trajectories_, timestamps_);
}

void MultitargetTracker::StartImageProcessingOrFilteringForMultipleExperiments(const char &dependence)
{
  boost::filesystem::path current_dir("//tsclient/D/Documents/Internship/02.08-_MultiTargetTracking/");
  boost::filesystem::path origin_dir = current_dir;
  boost::regex pattern("20.*");
  for (boost::filesystem::directory_iterator iter(current_dir), end; iter != end; ++iter)
  {
    boost::smatch match;
    std::string fn = iter->path().filename().string();
    if (boost::regex_match(fn, match, pattern))
    {
      std::cout << match[0] << "\n";
      current_dir += (fn + "/");
      boost::regex pattern("100.*");
      for (boost::filesystem::directory_iterator iter(current_dir), end; iter != end; ++iter)
      {
        boost::smatch match;
        std::string fn = iter->path().filename().string();
        if (boost::regex_match(fn, match, pattern))
        {
          boost::filesystem::path current_dir_copy = current_dir;
          current_dir += (fn + "/");
          boost::regex pattern("Config.*");
          for (boost::filesystem::directory_iterator iter(current_dir), end; iter != end; ++iter)
          {
            boost::smatch match;
            std::string fn = iter->path().filename().string();
            if ((boost::regex_match(fn, match, pattern)))
            {
              current_dir += fn;

              switch (dependence)
              {
                case '1':PerformImageProcessingForOneExperiment(current_dir.string());
                  break;
                case '2':StartFilteringWithoutImageProcessingForOneExperiment(current_dir.string());
                  break;
                case '3':StartTrackLinkingViaTemporalAssignment(current_dir.string());
                  break;
              }
            }
          } // iter
          current_dir = current_dir_copy;
        }
      } // iter
      current_dir = origin_dir;
    }
  } // iter
  // TODO: iter three times?
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