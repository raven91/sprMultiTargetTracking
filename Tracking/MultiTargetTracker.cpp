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

MultitargetTracker::MultitargetTracker() :
    targets_(),
    detections_()
{
  std::cout << "multi-target tracking started" << std::endl;
}

MultitargetTracker::~MultitargetTracker()
{
  std::cout << "multi-target tracking ended" << std::endl;
}

//void MultitargetTracker::StartOnExperimentalData()
//
//{
//  ParameterHandlerExperimental parameter_handler;
//  ImageProcessingEngine image_processing_engine(parameter_handler);
//  KalmanFilterExperimental kalman_filter(parameter_handler, image_processing_engine);
//
//  image_processing_engine.RetrieveBacterialData(parameter_handler.GetFirstImage(), detections_);
//  kalman_filter.InitializeTargets(targets_, detections_);
//
//  for (int i = parameter_handler.GetFirstImage() + 1; i <= parameter_handler.GetLastImage(); ++i)
//  {
//    image_processing_engine.RetrieveBacterialData(i, detections_);
//    kalman_filter.PerformEstimation(i, targets_, detections_);
//  }
//}


//
// Created by Stanislav Stepaniuk on 10.08.18 MODIFIED on  13.08.18
//
// Image processing without Kalman filtering
void MultitargetTracker::PerformImageProcessingForOneExperiment( const std::string& file_name )

{
	ParameterHandlerExperimental parameter_handler( file_name ) ;
	ImageProcessingEngine image_processing_engine(parameter_handler);

	image_processing_engine.RetrieveBacterialData(parameter_handler.GetFirstImage(), detections_);

	for (int i = parameter_handler.GetFirstImage() + 1; i <= parameter_handler.GetLastImage(); ++i)
	{
		image_processing_engine.RetrieveBacterialData(i, detections_);
	}
}//END of Image processing without Kalman filtering


 //
 // Created by Stanislav Stepaniuk on 13.08.18
 //
 // Searching for a Configuration file name and call of PerformImageProcessingForOneExperiment
void MultitargetTracker::PerformImageProcessingForMultipleExperiments()

{
	//path here to directory with folders with experiments (example: 20170705)
	boost::filesystem::path current_dir("//tsclient/D/Documents/Internship/02.08-_MultiTargetTracking/");
	boost::filesystem::path origin_dir = current_dir;
	// list all files starting with 20
	boost::regex pattern("20.*");
	for (boost::filesystem::directory_iterator iter(current_dir), end;
		iter != end;
		++iter)
	{
		boost::smatch match;
		std::string fn = iter->path().filename().string();
		if (boost::regex_match(fn, match, pattern))
		{
			std::cout << match[0] << "\n";
			current_dir += (fn + "/");
			boost::regex pattern("100.*");
			for (boost::filesystem::directory_iterator iter(current_dir), end;
				iter != end;
				++iter)
			{
				boost::smatch match;
				std::string fn = iter->path().filename().string();
				if (boost::regex_match(fn, match, pattern))
				{
					//std::cout << match[0] << "\n";
					boost::filesystem::path current_dir_copy = current_dir;
					current_dir += (fn + "/");
					//boost::regex pattern("DataAnalysis.*");
					boost::regex pattern("Config.*");
					for (boost::filesystem::directory_iterator iter(current_dir), end;
						iter != end;
						++iter)

					{
						boost::smatch match;
						std::string fn = iter->path().filename().string();
						//if ((boost::regex_match(fn, match, pattern_1)) or (boost::regex_match(fn, match, pattern_2)))
						if ((boost::regex_match(fn, match, pattern)))
						{
							current_dir += fn;
							//std::cout << current_dir << "\n";
							PerformImageProcessingForOneExperiment(current_dir.string());						
						}

					}


					//std::cout << "\n";
					current_dir = current_dir_copy;
				}
			}
			current_dir = origin_dir;
		}
	}
}//END of Searching for a Configuration file name and call of PerformImageProcessingForOneExperiment



 //
 // Created by Stanislav Stepaniuk on 10.08.18
 //
 // Kalman filtering without Image processing 
void MultitargetTracker::StartOnReceivedDataWithoutImageProcessing()
{
	ParameterHandlerSynthetic parameter_handler; 
	std::ostringstream solution_file_name;
	solution_file_name	<< parameter_handler.GetTrackingFolder()
						<< "image_processing_output.txt";
	std::ifstream modified_solution_file(solution_file_name.str(), std::ios::in);
	assert(modified_solution_file.is_open());
	
	//KalmanFilterExperimental kalman_filter(parameter_handler, image_processing_engine);
	//kalman_filter.InitializeTargets(targets_, detections_);

	//for (int i = parameter_handler.GetFirstImage() + 1; i <= parameter_handler.GetLastImage(); ++i)
	//{
	//	kalman_filter.PerformEstimation(i, targets_, detections_);
	//}
}//END of Kalman filtering without Image processing 



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