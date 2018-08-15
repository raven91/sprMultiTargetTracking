//
// Created by Nikita Kruk on 27.11.17.
//

#include "KalmanFilterExperimental.hpp"
#include "HungarianAlgorithm.hpp"

#include <iostream>
#include <sstream>
#include <algorithm> // std::copy, std::max, std::set_difference, std::for_each, std::sort
#include <iterator>  // std::back_inserter, std::prev
#include <cmath>
#include <set>
#include <numeric>   // std::iota
#include <random>
#include <iomanip>   // std::setfill, std::setw

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

KalmanFilterExperimental::KalmanFilterExperimental(ParameterHandlerExperimental &parameter_handler,
	ImageProcessingEngine &image_processing_engine) :
	parameter_handler_(parameter_handler),
	image_processing_engine_(image_processing_engine),
	costs_order_of_magnitude_(1000.0),
	unmatched_(),
	max_prediction_time_(5),
	max_target_index_(0)
{
	std::string kalman_filter_output_file_name =
		parameter_handler_.GetInputFolder() + parameter_handler.GetDataAnalysisSubfolder()
		+ parameter_handler.GetKalmanFilterOutputFileName();
	kalman_filter_output_file_.open(kalman_filter_output_file_name, std::ios::out | std::ios::trunc);
	assert(kalman_filter_output_file_.is_open());

	std::string kalman_filter_matlab_output_file_name =
		parameter_handler_.GetInputFolder() + parameter_handler_.GetDataAnalysisSubfolder()
		+ parameter_handler_.GetKalmanFilterMatlabOutputFileName();
	kalman_filter_matlab_output_file_.open(kalman_filter_matlab_output_file_name, std::ios::out | std::ios::trunc);
	assert(kalman_filter_matlab_output_file_.is_open());
}

KalmanFilterExperimental::~KalmanFilterExperimental()
{
	kalman_filter_output_file_.close();
	kalman_filter_matlab_output_file_.close();
}

void KalmanFilterExperimental::InitializeTargets(std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections)
{
	targets.clear();

	int last_index = 0;
	Eigen::VectorXf new_target(kNumOfExtractedFeatures);
	for (int b = 0; b < detections.size(); ++b)
	{
		if (targets.empty())
		{
			last_index = -1;
		}
		else
		{
			last_index = std::prev(targets.end())->first;
			//			last_index = targets.rbegin()->first;
		}
		new_target = detections[b];
		targets[++last_index] = new_target;
	}
	max_target_index_ = last_index;

	SaveTargets(kalman_filter_output_file_, parameter_handler_.GetFirstImage(), targets);
	SaveTargetsMatlab(kalman_filter_matlab_output_file_, parameter_handler_.GetFirstImage(), targets);
	SaveImages(parameter_handler_.GetFirstImage(), targets);
}

//
// Created by Stanislav Stepaniuk on 14.08.18 MODOFIED on 15.08.18
//
// New function for initializing targets
void KalmanFilterExperimental::InitializeTargets(std::map<int, Eigen::VectorXf> &targets, std::ifstream &file)
{
	int last_index = 0;
	Eigen::VectorXf new_target = Eigen::MatrixXf::Zero(kNumOfExtractedFeatures, 1);
	int time_idx = 0;
	int target_idx = 0;
	int number_of_detections = 0;

	do
	{
		targets.clear();
		file >> time_idx >> number_of_detections;
		for (int b = 0; b < number_of_detections; ++b)
		{
			if (targets.empty())
			{
				last_index = -1;
			}
			else
			{
				last_index = std::prev(targets.end())->first;
			}
			file >> target_idx >> new_target(0) >> new_target(1) >> new_target(2)
				>> new_target(3) >> new_target(4) >> new_target(5) >> new_target(6) >> new_target(7);
			targets[++last_index] = new_target;
		}
		max_target_index_ = last_index;
	} while (time_idx < parameter_handler_.GetFirstImage());

	SaveTargets(kalman_filter_output_file_, parameter_handler_.GetFirstImage(), targets);
	SaveTargetsMatlab(kalman_filter_matlab_output_file_, parameter_handler_.GetFirstImage(), targets);
	SaveImages(parameter_handler_.GetFirstImage(), targets);
}// END of New function for initializing targets

 //
 // Created by Stanislav Stepaniuk on 14.08.18
 //
 // Obtaining new detections for filtering 
void KalmanFilterExperimental::ObtainNewDetections(std::vector<Eigen::VectorXf> &detections, std::ifstream &file)
{
	detections.clear();

	Eigen::VectorXf new_detection = Eigen::MatrixXf::Zero(kNumOfExtractedFeatures, 1);
	int time_idx = 0;
	int detection_idx = 0;
	int number_of_detections = 0;

	file >> time_idx >> number_of_detections;
	for (int b = 0; b < number_of_detections; ++b)
	{
		file >> detection_idx >> new_detection(0) >> new_detection(1) >> new_detection(2)
			>> new_detection(3) >> new_detection(4) >> new_detection(5) >> new_detection(6) >> new_detection(7);
		detections.push_back(new_detection);
	}
}//END of Obtaining new detections for filtering 

void KalmanFilterExperimental::PerformEstimation(int image_idx,
	std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections)
{
	std::cout << "kalman filter: image#" << image_idx << std::endl;

	int n_max_dim = 0; // max size between targets and detections
	CostInt max_cost = 0;
	Real dt = 1;// in ms

	Eigen::MatrixXf I = Eigen::MatrixXf::Identity(kNumOfStateVars, kNumOfStateVars);
	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
	Eigen::MatrixXf W = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
	Eigen::MatrixXf H = Eigen::MatrixXf::Zero(kNumOfDetectionVars, kNumOfStateVars);
	Eigen::MatrixXf Q = Eigen::MatrixXf::Zero(kNumOfDetectionVars, kNumOfDetectionVars);
	Eigen::MatrixXf P_estimate = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);
	Eigen::MatrixXf K = Eigen::MatrixXf::Zero(kNumOfStateVars, kNumOfStateVars);

	A(0, 0) = A(1, 1) = A(2, 2) = A(3, 3) = 1.0;
	A(0, 2) = A(1, 3) = dt;
	H(0, 0) = H(1, 1) = 1.0;

	W(0, 0) = W(1, 1) = dt * dt * dt * dt / 4.0f;
	W(2, 2) = W(3, 3) = dt * dt;
	W(0, 2) = W(1, 3) = W(2, 0) = W(3, 1) = dt * dt * dt / 2.0f;
	Q(0, 0) = Q(1, 1) = dt;

	P_estimate = W;
	ComputePriorEstimate(targets, P_estimate, A, W, H);
	ComputeKalmanGainMatrix(K, P_estimate, H, Q);

	if (detections.size() > 0)
	{
		n_max_dim = (int)std::max(targets.size(), detections.size());
		std::vector<int> target_indexes;
		std::vector<std::vector<CostInt>> cost_matrix(n_max_dim, std::vector<CostInt>(n_max_dim, 0));
		std::vector<int> assignments(n_max_dim, -1);
		std::vector<CostInt> costs(n_max_dim);

		PerformDataAssociation(targets, detections, n_max_dim, target_indexes, cost_matrix, assignments, costs);
		UnassignUnrealisticTargets(targets, detections, n_max_dim, assignments, costs);
		ComputePosteriorEstimate(targets, detections, P_estimate, K, H, assignments, target_indexes);
		RemoveRecapturedTargetsFromStrikes(targets, assignments, target_indexes);
		AddNewTargets(targets, detections, assignments);
		MarkLostTargetsAsUnmatched(targets, assignments, target_indexes);
	}
	else // detections.size() == 0
	{
		MarkAllTargetsAsUnmatched(targets);
	}
	// if the target has been lost for too long -> remove it
	DeleteLongLostTargets(targets);
	CorrectForOrientationUniqueness(targets);

	SaveTargets(kalman_filter_output_file_, image_idx, targets);
	SaveTargetsMatlab(kalman_filter_matlab_output_file_, image_idx, targets);
	SaveImages(image_idx, targets);

	std::cout << "number of overall targets taken part: " << max_target_index_ + 1 << "; number of current targets: "
		<< targets.size() << std::endl;
}

void KalmanFilterExperimental::ComputePriorEstimate(std::map<int, Eigen::VectorXf> &targets,
	Eigen::MatrixXf &P_estimate,
	const Eigen::MatrixXf &A,
	const Eigen::MatrixXf &W,
	const Eigen::MatrixXf &H)
{
	Eigen::VectorXf x_i_estimate(kNumOfStateVars);
	for (std::map<int, Eigen::VectorXf>::iterator it = targets.begin(); it != targets.end(); ++it)
	{
		x_i_estimate = (it->second).head(kNumOfStateVars);
		x_i_estimate = A * x_i_estimate;
		(it->second).head(kNumOfStateVars) = x_i_estimate;
	}
	P_estimate = A * P_estimate * A.transpose() + W;
}

void KalmanFilterExperimental::ComputeKalmanGainMatrix(Eigen::MatrixXf &K,
	const Eigen::MatrixXf &P_estimate,
	const Eigen::MatrixXf &H,
	const Eigen::MatrixXf &Q)
{
	K = P_estimate * H.transpose() * (H * P_estimate * H.transpose() + Q).inverse();
}

void KalmanFilterExperimental::PerformDataAssociation(const std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections,
	int n_max_dim,
	std::vector<int> &target_indexes,
	std::vector<std::vector<CostInt>> &cost_matrix,
	std::vector<int> &assignments,
	std::vector<CostInt> &costs)
{
	CostInt max_cost = InitializeCostMatrix(targets, detections, cost_matrix, target_indexes);
	HungarianAlgorithm hungarian_algorithm(n_max_dim, cost_matrix);
	hungarian_algorithm.Start(assignments, costs);
	std::for_each(costs.begin(),
		costs.end(),
		[&](CostInt &c)
	{
		c = CostInt((max_cost - c) / costs_order_of_magnitude_);
	});
}

void KalmanFilterExperimental::UnassignUnrealisticTargets(const std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections,
	int n_max_dim,
	std::vector<int> &assignments,
	std::vector<CostInt> &costs)
{
	// if a cost is too high or if the assignment is into an imaginary detection
	for (int i = 0; i < targets.size(); ++i)
	{
		if (costs[i] > parameter_handler_.GetDataAssociationCost() || assignments[i] >= detections.size()) // in pixels
		{
			assignments[i] = -1;
		}
	}
	// if the assignment is from an imaginary target
	for (int i = (int)targets.size(); i < n_max_dim; ++i)
	{
		assignments[i] = -1;
	}
}

void KalmanFilterExperimental::ComputePosteriorEstimate(std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections,
	Eigen::MatrixXf &P_estimate,
	const Eigen::MatrixXf &K,
	const Eigen::MatrixXf &H,
	const std::vector<int> &assignments,
	const std::vector<int> &target_indexes)
{
	Eigen::VectorXf z_i(kNumOfDetectionVars);
	Eigen::VectorXf x_i_estimate(kNumOfStateVars);
	for (int i = 0; i < targets.size(); ++i)
	{
		if (assignments[i] != -1)
		{
			x_i_estimate = targets[target_indexes[i]].head(kNumOfStateVars);
			z_i = detections[assignments[i]].head(2);
			x_i_estimate = x_i_estimate + K * (z_i - H * x_i_estimate);
			targets[target_indexes[i]].head(kNumOfStateVars) = x_i_estimate;

			targets[target_indexes[i]][4] = detections[assignments[i]][4];
			targets[target_indexes[i]][5] = detections[assignments[i]][5];
			targets[target_indexes[i]][6] = detections[assignments[i]][6];
			targets[target_indexes[i]][7] = detections[assignments[i]][7];
		}
	}
	Eigen::MatrixXf I = Eigen::MatrixXf::Identity(kNumOfStateVars, kNumOfStateVars);
	P_estimate = (I - K * H) * P_estimate;
}

void KalmanFilterExperimental::MarkLostTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets,
	const std::vector<int> &assignments,
	const std::vector<int> &target_indexes)
{
	// idx.size() == number of targets at the beginning of the iteration
	for (int i = 0; i < target_indexes.size(); ++i)
	{
		if (assignments[i] == -1)
		{
			if (unmatched_.find(target_indexes[i]) != unmatched_.end())
			{
				++unmatched_[target_indexes[i]];
			}
			else
			{
				unmatched_[target_indexes[i]] = 1;
			}
		}
	}
	//    for (int i = 0; i < additional_target_indexes.size(); ++i)
	//    {
	//      if (additional_assignments[i] == -1)
	//      {
	//        if (unmatched_.find(additional_target_indexes[i]) != unmatched_.end())
	//        {
	//          ++unmatched_[additional_target_indexes[i]];
	//        } else
	//        {
	//          unmatched_[additional_target_indexes[i]] = 1;
	//        }
	//      }
	//    }
	//  }
}

void KalmanFilterExperimental::RemoveRecapturedTargetsFromStrikes(std::map<int, Eigen::VectorXf> &targets,
	const std::vector<int> &assignments,
	const std::vector<int> &target_indexes)
{
	for (int i = 0; i < targets.size(); ++i)
	{
		if (assignments[i] != -1)
		{
			if (unmatched_.find(target_indexes[i]) != unmatched_.end())
			{
				unmatched_.erase(target_indexes[i]); // stop suspecting a target if it has been recovered
			}
		}
	}
}

void KalmanFilterExperimental::MarkAllTargetsAsUnmatched(std::map<int, Eigen::VectorXf> &targets)
{
	// all the targets have been lost
	for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
	{
		if (unmatched_.find(it->first) != unmatched_.end())
		{
			++unmatched_[it->first];
		}
		else
		{
			unmatched_[it->first] = 1;
		}
	}
}

void KalmanFilterExperimental::AddNewTargets(std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections,
	const std::vector<int> &assignments)
{
	std::vector<int> all_detection_indexes(detections.size());
	std::iota(all_detection_indexes.begin(),
		all_detection_indexes.end(),
		0); // construct detection indexes from 0 through d.size()-1
	std::vector<int> sorted_assignments(assignments.begin(), assignments.end());
	std::sort(sorted_assignments.begin(), sorted_assignments.end());
	std::vector<int> indexes_to_unassigned_detections;
	std::set_difference(all_detection_indexes.begin(),
		all_detection_indexes.end(),
		sorted_assignments.begin(),
		sorted_assignments.end(),
		std::back_inserter(indexes_to_unassigned_detections)); // set_difference requires pre-sorted containers
															   //  // for each new detection, divide its contour if there exists significant convexity defect
															   //
															   //  std::vector<Eigen::VectorXf> additional_detections;
															   //  image_processing_engine_.ProcessAdditionalDetections(indexes_to_unassigned_detections,
															   //                                                       additional_detections,
															   //                                                       detections);
															   //  std::map<int, Eigen::VectorXf> terminated_targets;
															   //  for (int i = 0; i < target_indexes.size(); ++i)
															   //  {
															   //    if (assignments[i] == -1)
															   //    {
															   //      terminated_targets[target_indexes[i]] = targets.find(target_indexes[i])->second;
															   //    }
															   //  }
															   //
															   ////	if (/* DISABLES CODE */ (false))
															   ////	if ((terminated_targets.size() != 0) && (additional_detections.size() != 0))
															   //  {
															   //    n_max_dim = (int) std::max(terminated_targets.size(), additional_detections.size());
															   //    cost_matrix = std::vector<std::vector<CostInt>>(n_max_dim, std::vector<CostInt>(n_max_dim, 0));
															   //    std::vector<int> additional_target_indexes;
															   //    std::vector<int> additional_assignments(n_max_dim, -1);
															   //    std::vector<CostInt> additional_costs(n_max_dim);
															   //    max_cost = InitializeSecondaryCostMatrix(terminated_targets,
															   //                                             additional_detections,
															   //                                             cost_matrix,
															   //                                             additional_target_indexes);
															   //    HungarianAlgorithm additional_hungarian_algorithm(n_max_dim, cost_matrix);
															   //    additional_hungarian_algorithm.Start(additional_assignments, additional_costs);
															   //    std::for_each(additional_costs.begin(),
															   //                  additional_costs.end(),
															   //                  [&](CostInt &c) { c = CostInt(Real(max_cost - c) * costs_order_of_magnitude_); });
															   //    for (int i = 0; i < terminated_targets.size(); ++i)
															   //    {
															   //      if (additional_costs[i] > parameter_handler_.GetSecondaryDataAssociationCost()
															   //          || additional_assignments[i] >= additional_detections.size())
															   //      {
															   //        additional_assignments[i] = -1;
															   //      }
															   //    }
															   //    for (int i = (int) terminated_targets.size(); i < n_max_dim; ++i)
															   //    {
															   //      additional_assignments[i] = -1;
															   //    }
															   //    // POSTERIOR ESTIMATE OF DOUBLE-SHAPED BACTERIA
															   //    for (int i = 0; i < terminated_targets.size(); ++i)
															   //    {
															   //      if (additional_assignments[i] != -1)
															   //      {
															   //        x_i_estimate = terminated_targets[additional_target_indexes[i]].head(kNumOfStateVars);
															   //        z_i = additional_detections[additional_assignments[i]].head(2);
															   //        x_i_estimate = x_i_estimate + K * (z_i - H * x_i_estimate);
															   //        targets[additional_target_indexes[i]].head(kNumOfStateVars) = x_i_estimate;
															   //
															   //        targets[additional_target_indexes[i]][4] = additional_detections[additional_assignments[i]][4];
															   //        targets[additional_target_indexes[i]][5] = additional_detections[additional_assignments[i]][5];
															   //        targets[additional_target_indexes[i]][6] = additional_detections[additional_assignments[i]][6];
															   //        targets[additional_target_indexes[i]][7] = additional_detections[additional_assignments[i]][7];
															   //
															   //        if (unmatched_.find(additional_target_indexes[i]) != unmatched_.end())
															   //        {
															   //          unmatched_.erase(additional_target_indexes[i]); // stop suspecting a target if it has been recovered
															   //        }
															   //      }
															   //    }

															   // add the newly found trackings
															   //	for (int i = 0; i < new_trackings.size(); ++i)
															   //	{
															   //		x[std::prev(x.end())->first + 1] = d[new_trackings[i]];
															   //	}
															   //    indexes_to_unassigned_detections.clear();
															   //    all_detection_indexes = std::vector<int>(additional_detections.size());
															   //    std::iota(all_detection_indexes.begin(), all_detection_indexes.end(), 0);
															   //    sorted_assignments = std::vector<int>(additional_assignments.begin(), additional_assignments.end());
															   //    std::sort(sorted_assignments.begin(), sorted_assignments.end());
															   //    std::set_difference(all_detection_indexes.begin(),
															   //                        all_detection_indexes.end(),
															   //                        sorted_assignments.begin(),
															   //                        sorted_assignments.end(),
															   //                        std::back_inserter(indexes_to_unassigned_detections));
															   // consider detections, left after second segmentation, as new targets
	for (int i = 0; i < indexes_to_unassigned_detections.size(); ++i)
	{
		//      targets[std::prev(targets.end())->first + 1] = additional_detections[indexes_to_unassigned_detections[i]]; // TODO: UNCOMMENT FOR SECONDARY DATA ASSOCIATION
		targets[max_target_index_ + 1] = detections[indexes_to_unassigned_detections[i]];
		++max_target_index_;
	}
}

void KalmanFilterExperimental::DeleteLongLostTargets(std::map<int, Eigen::VectorXf> &targets)
{
	for (std::map<int, int>::iterator it = unmatched_.begin(); it != unmatched_.end();)
	{
		if (it->second > max_prediction_time_)
		{
			targets.erase(it->first);
			it = unmatched_.erase(it);
		}
		else
		{
			++it;
		}
	}
}

void KalmanFilterExperimental::CorrectForOrientationUniqueness(std::map<int, Eigen::VectorXf> &targets)
{
	Eigen::VectorXf x_i(kNumOfExtractedFeatures);
	Eigen::Vector2f velocity_i;
	Eigen::Vector2f orientation_i;
	for (std::map<int, Eigen::VectorXf>::iterator it = targets.begin(); it != targets.end(); ++it)
	{
		x_i = it->second;
		velocity_i = x_i.segment(2, 2);
		orientation_i << std::cosf(x_i[5]), std::sinf(x_i[5]);

		// in order to determine the orientation vector uniquely, we assume the angle difference between the orientation and velocity is < \pi/2
		if (velocity_i.dot(orientation_i) < 0.0f)
		{
			(it->second)[5] = WrappingModulo(x_i[5] + M_PI, 2 * M_PI);
		}
	}
}

void KalmanFilterExperimental::SaveTargets(std::ofstream &file,
	int image_idx,
	const std::map<int, Eigen::VectorXf> &targets)
{
	Eigen::VectorXf x_i;
	file << image_idx << " " << targets.size() << " ";
	for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
	{
		x_i = it->second;
		file << it->first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3) << " " << x_i(4) << " "
			<< x_i(5) << " " << x_i(6) << " " << x_i(7) << " ";
	}
	file << std::endl;
}

void KalmanFilterExperimental::SaveTargetsMatlab(std::ofstream &file,
	int image_idx,
	const std::map<int, Eigen::VectorXf> &targets)
{
	Eigen::VectorXf x_i;
	for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
	{
		x_i = it->second;
		file << image_idx << " " << it->first << " " << x_i(0) << " " << x_i(1) << " " << x_i(2) << " " << x_i(3) << " "
			<< x_i(4) << " " << x_i(5) << " " << x_i(6) << " " << x_i(7) << std::endl;
	}
}

void KalmanFilterExperimental::SaveImages(int image_idx, const std::map<int, Eigen::VectorXf> &targets)
{
	cv::Mat image;
	image = image_processing_engine_.GetSourceImage(image_idx);

	Eigen::VectorXf x_i;
	cv::Point2f center;
	cv::Scalar color(255, 127, 0);
	Real length = 0.0f;

	for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it)
	{
		x_i = it->second;
		center = cv::Point2f(x_i(0), x_i(1));
		cv::circle(image, center, 3, color, -1, 8);
		cv::putText(image, std::to_string(it->first), center, cv::FONT_HERSHEY_DUPLEX, 0.4, color);
		//		cv::Point2f pt = cv::Point2f(std::cosf(x_i(5)), std::sinf(x_i(5)));
		length = std::max(x_i(6), x_i(7));
		cv::line(image,
			center,
			center + cv::Point2f(std::cosf(x_i(5)), std::sinf(x_i(5))) * length / 2.0f,
			cv::Scalar(255, 0, 0));
		//		std::cout << "(" << center.x << "," << center.y << ") -> (" << center.x + std::cosf(x_i(5)) * x_i(4) / 10.0f << "," << center.y + std::sinf(x_i(5)) * x_i(4) / 10.0f << ")" << std::endl;
	}

	std::ostringstream output_image_name_buf;
	output_image_name_buf << parameter_handler_.GetInputFolder() << parameter_handler_.GetKalmanFilterSubfolder()
		<< parameter_handler_.GetFileName0() << std::setfill('0') << std::setw(9) << image_idx
		<< parameter_handler_.GetFileName1();
	std::string output_image_name = output_image_name_buf.str();
	cv::imwrite(output_image_name, image);
}

CostInt KalmanFilterExperimental::InitializeCostMatrix(const std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections,
	std::vector<std::vector<CostInt>> &cost_matrix,
	std::vector<int> &target_indexes)
{
	target_indexes.clear();

	Eigen::VectorXf target(kNumOfExtractedFeatures);
	Eigen::VectorXf detection(kNumOfExtractedFeatures);
	Real cost = 0.0;
	Real max_cost = 0;
	int i = 0;
	Real d_x = 0.0, d_y = 0.0;
	Real dist = 0.0;
	Real max_dist = Real(std::sqrt(parameter_handler_.GetSubimageXSize() * parameter_handler_.GetSubimageXSize()
		+ parameter_handler_.GetSubimageYSize() * parameter_handler_.GetSubimageYSize()));
	for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it, ++i)
	{
		target_indexes.push_back(it->first);
		target = it->second;

		for (int j = 0; j < detections.size(); ++j)
		{
			detection = detections[j];
			d_x = (target(0) - detection(0));
			d_y = (target(1) - detection(1));

			// put only close assignment costs in the cost matrix
			dist = std::sqrt(d_x * d_x + d_y * d_y);
			if (dist <= parameter_handler_.GetDataAssociationCost())
			{
				cost = dist; // Euclidean norm from a target to a detection
			}
			else
			{
				cost = max_dist;
			}

			cost_matrix[i][j] = CostInt(cost * costs_order_of_magnitude_);
			if (max_cost < cost)
			{
				max_cost = cost;
			}
		}
	}

	// turn min cost problem into max cost problem
	for (int i = 0; i < targets.size(); ++i)
	{
		for (int j = 0; j < detections.size(); ++j)
		{
			cost_matrix[i][j] = CostInt(max_cost * costs_order_of_magnitude_) - cost_matrix[i][j];
		}
	}

	return CostInt(max_cost * costs_order_of_magnitude_);
}

CostInt KalmanFilterExperimental::InitializeSecondaryCostMatrix(const std::map<int, Eigen::VectorXf> &targets,
	const std::vector<Eigen::VectorXf> &detections,
	std::vector<std::vector<CostInt>> &cost_matrix,
	std::vector<int> &target_indexes)
{
	target_indexes.clear();

	Eigen::VectorXf target(kNumOfExtractedFeatures);
	Eigen::VectorXf detection(kNumOfExtractedFeatures);
	Real cost = 0.0;
	Real max_cost = 0;
	int i = 0;
	Real d_x = 0.0, d_y = 0.0;
	Real dist = 0.0;
	Real max_dist = Real(std::sqrt(parameter_handler_.GetSubimageXSize() * parameter_handler_.GetSubimageXSize()
		+ parameter_handler_.GetSubimageYSize() * parameter_handler_.GetSubimageYSize()));
	for (std::map<int, Eigen::VectorXf>::const_iterator it = targets.begin(); it != targets.end(); ++it, ++i)
	{
		target_indexes.push_back(it->first);
		target = it->second;

		for (int j = 0; j < detections.size(); ++j)
		{
			detection = detections[j];

			d_x = (target(0) - detection(0));
			d_y = (target(1) - detection(1));

			// put only close assignment costs in the cost matrix
			dist = std::sqrt(d_x * d_x + d_y * d_y);
			if (dist <= parameter_handler_.GetDataAssociationCost())
			{
				cost = dist; // Euclidean norm from a target to a detection
			}
			else
			{
				cost = max_dist;
			}

			cost_matrix[i][j] = CostInt(cost * costs_order_of_magnitude_);
			if (max_cost < cost)
			{
				max_cost = cost;
			}
		}
	}

	// turn min cost problem into max cost problem
	for (int i = 0; i < targets.size(); ++i)
	{
		for (int j = 0; j < detections.size(); ++j)
		{
			cost_matrix[i][j] = CostInt(max_cost * costs_order_of_magnitude_) - cost_matrix[i][j];
		}
	}

	return int(max_cost);
}
