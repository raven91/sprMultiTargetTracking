//
// Created by Nikita Kruk on 27.11.17.
//

#include "HungarianAlgorithm.hpp"

#include <algorithm>//std::max, std::min
#include <limits>//std::numeric_limits

HungarianAlgorithm::HungarianAlgorithm(int n, std::vector<std::vector<CostInt>> &cost_matrix) :
	n_(n),
	cost_matrix_(cost_matrix),
	labels_x_(n, 0),
	labels_y_(n, 0),
	xy_(n, -1),
	yx_(n, -1),
	S_(n, false),
	T_(n, false),
	slack_(n, 0),
	slack_x_(n, -1),
	prev_(n, -1)
{
	max_match_ = 0;
}

HungarianAlgorithm::~HungarianAlgorithm()
{

}

void HungarianAlgorithm::Start(std::vector<int> &assignment, std::vector<CostInt> &cost)
{
	max_match_ = 0;
	InitializeLabels();
	Augment();

	for (int x = 0; x < n_; ++x)
	{
		assignment[x] = xy_[x];
		cost[x] = cost_matrix_[x][xy_[x]];
	}
}

void HungarianAlgorithm::InitializeLabels()
{
	//	std::fill(labels_x_.begin(), labels_x_.end(), 0);
	//	std::fill(labels_y_.begin(), labels_y_.end(), 0);

	for (int x = 0; x < n_; ++x)
	{
		for (int y = 0; y < n_; ++y)
		{
			labels_x_[x] = std::max(labels_x_[x], cost_matrix_[x][y]);
		}
	}
}

void HungarianAlgorithm::Augment()
{
	if (max_match_ == n_)
	{
		return;
	}

	int x = 0, y = 0, root = 0;
	std::vector<int> q(n_, 0); // BFS queue for the alternating tree
	int wr = 0, rd = 0; // counters for the queue
	std::fill(S_.begin(), S_.end(), false);
	std::fill(T_.begin(), T_.end(), false);
	std::fill(prev_.begin(), prev_.end(), -1);
	for (x = 0; x < n_; ++x)
	{
		if (xy_[x] == -1)
		{
			q[wr++] = root = x;
			prev_[x] = -2;
			S_[x] = true;
			break;
		}
	}

	for (y = 0; y < n_; ++y)
	{
		slack_[y] = labels_x_[root] + labels_y_[y] - cost_matrix_[root][y];
		slack_x_[y] = root;
	}

	while (true)
	{
		while (rd < wr)
		{
			x = q[rd++];
			for (y = 0; y < n_; ++y)
			{
				if (cost_matrix_[x][y] == labels_x_[x] + labels_y_[y] && !T_[y])
				{
					if (yx_[y] == -1) // the augmenting path has been found
					{
						break;
					}
					T_[y] = true;
					q[wr++] = yx_[y];
					AddToTree(yx_[y], x);
				}
			}

			if (y < n_) // the augmenting path has been found
			{
				break;
			}
		}

		if (y < n_) // the augmenting path has been found
		{
			break;
		}

		UpdateLabels();

		wr = rd = 0;
		for (y = 0; y < n_; ++y)
		{
			if (!T_[y] && slack_[y] == 0)
			{
				if (yx_[y] == -1) // the augmenting path has been found
				{
					x = slack_x_[y];
					break;
				}
				else
				{
					T_[y] = true;
					if (!S_[yx_[y]])
					{
						q[wr++] = yx_[y];
						AddToTree(yx_[y], slack_x_[y]);
					}
				}
			}
		}

		if (y < n_) // the augmenting path has been found
		{
			break;
		}
	}

	if (y < n_)
	{
		++max_match_;
		for (int cx = x, cy = y, ty; cx != -2; cx = prev_[cx], cy = ty)
		{
			ty = xy_[cx];
			yx_[cy] = cx;
			xy_[cx] = cy;
		}
		Augment();
	}
}

void HungarianAlgorithm::UpdateLabels()
{
	CostInt alpha_l = std::numeric_limits<CostInt>::max();

	for (int y = 0; y < n_; ++y)
	{
		if (!T_[y])
		{
			alpha_l = std::min(alpha_l, slack_[y]);
		}
	}

	for (int x = 0; x < n_; ++x)
	{
		if (S_[x])
		{
			labels_x_[x] -= alpha_l;
		}
	}

	for (int y = 0; y < n_; ++y)
	{
		if (T_[y])
		{
			labels_y_[y] += alpha_l;
		}
	}

	for (int y = 0; y < n_; ++y)
	{
		if (!T_[y])
		{
			slack_[y] -= alpha_l;
		}
	}
}

void HungarianAlgorithm::AddToTree(int x, int prev_x)
{
	S_[x] = true;
	prev_[x] = prev_x;
	for (int y = 0; y < n_; ++y)
	{
		if (labels_x_[x] + labels_y_[y] - cost_matrix_[x][y] < slack_[y])
		{
			slack_[y] = labels_x_[x] + labels_y_[y] - cost_matrix_[x][y];
			slack_x_[y] = x;
		}
	}
}