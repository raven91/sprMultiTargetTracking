//
// Created by Nikita Kruk on 27.11.17.
//

#ifndef SPRMULTITARGETTRACKING_HUNGARIANALGORITHM_HPP
#define SPRMULTITARGETTRACKING_HUNGARIANALGORITHM_HPP

#include "../Definitions.hpp"

#include <vector>

class HungarianAlgorithm
{
public:

    HungarianAlgorithm(int n, std::vector<std::vector<CostInt>> &cost_matrix);
    ~HungarianAlgorithm();

    void Start(std::vector<int> &assignments, std::vector<CostInt> &costs);

private:

    int n_;
    std::vector<std::vector<CostInt>> &cost_matrix_;
    std::vector<CostInt> labels_x_;
    std::vector<CostInt> labels_y_;
    std::vector<int> xy_;
    std::vector<int> yx_;
    std::vector<bool> S_;
    std::vector<bool> T_;
    std::vector<CostInt> slack_;
    std::vector<int> slack_x_;
    std::vector<int> prev_;
    int max_match_;

    void InitializeLabels();
    void Augment();
    void UpdateLabels();
    void AddToTree(int x, int prev_x);
};

#endif //SPRMULTITARGETTRACKING_HUNGARIANALGORITHM_HPP
