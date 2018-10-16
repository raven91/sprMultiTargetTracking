//
// Created by Nikita Kruk on 23.07.18.
//

#include "TrackingStatistics.hpp"

#include <sstream>

TrackingStatistics::TrackingStatistics() :
    number_of_suspended_targets_(0),
    number_of_missed_targets_(0),
    number_of_deleted_targets_(0),
    number_of_new_targets_(0),
    number_of_time_points_(0)
{

}

TrackingStatistics::~TrackingStatistics() = default;

int TrackingStatistics::GetNumberOfSuspendedTargets()
{
  return number_of_suspended_targets_;
}

int TrackingStatistics::GetNumberOfMissedTargets()
{
  return number_of_missed_targets_;
}

int TrackingStatistics::GetNumberOfDeletedTargets()
{
  return number_of_deleted_targets_;
}

int TrackingStatistics::GetNumberOfNewTargets()
{
  return number_of_new_targets_;
}

int TrackingStatistics::GetNumberOfRecapturedTargets()
{
  return number_of_recaptured_targets_;
}

int TrackingStatistics::GetNumberOfTimePoints()
{
  return number_of_time_points_;
}

void TrackingStatistics::IncrementNumberOfSuspendedTargets(int increment)
{
  number_of_suspended_targets_ += increment;
}

void TrackingStatistics::IncrementNumberOfMissedTargets(int increment)
{
  number_of_missed_targets_ += increment;
}

void TrackingStatistics::IncrementNumberOfDeletedTargets(int increment)
{
  number_of_deleted_targets_ += increment;
}

void TrackingStatistics::IncrementNumberOfNewTargets(int increment)
{
  number_of_new_targets_ += increment;
}

void TrackingStatistics::IncrementNumberOfRecapturedTargets(int increment)
{
  number_of_recaptured_targets_ += increment;
}

void TrackingStatistics::IncrementNumberOfTimePoints(int increment)
{
  number_of_time_points_ += increment;
}