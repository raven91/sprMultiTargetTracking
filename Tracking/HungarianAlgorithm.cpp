//
// Created by Nikita Kruk on 27.11.17.
//

#include "HungarianAlgorithm.hpp"

#include <algorithm>    // std::max, std::min
#include <limits>   // std::numeric_limits

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
  AugmentSequentially();

  for (int x = 0; x < n_; ++x)
  {
    assignment[x] = xy_[x];
    cost[x] = cost_matrix_[x][xy_[x]];
  }
}

void HungarianAlgorithm::InitializeLabels()
{
  std::fill(labels_x_.begin(), labels_x_.end(), 0);
  std::fill(labels_y_.begin(), labels_y_.end(), 0);

  for (int x = 0; x < n_; ++x)
  {
    for (int y = 0; y < n_; ++y)
    {
      labels_x_[x] = std::max(labels_x_[x], cost_matrix_[x][y]);
    }
  }
}

void HungarianAlgorithm::AugmentRecursively()
{
  if (max_match_ == n_) //check wether matching is already perfect
  {
    return;
  }

  int x = 0, y = 0, root = 0;
  std::vector<int> q(n_, 0); // breadth-first search (BFS) queue for the alternating tree
  int wr = 0, rd = 0; // counters for the queue
  std::fill(S_.begin(), S_.end(), false);
  std::fill(T_.begin(), T_.end(), false);
  std::fill(prev_.begin(), prev_.end(), -1);
  for (x = 0; x < n_; ++x) //finding root of the tree
  {
    if (xy_[x] == -1)
    {
      q[wr++] = root = x;
      prev_[x] = -2;
      S_[x] = true;
      break;
    }
  }

  for (y = 0; y < n_; ++y) //initializing slack array
  {
    slack_[y] = labels_x_[root] + labels_y_[y] - cost_matrix_[root][y];
    slack_x_[y] = root;
  }

  while (true)
  {
    while (rd < wr) //building tree with bfs cycle
    {
      x = q[rd++]; // current vertex from X part
      for (y = 0; y < n_; ++y) // iterate through all edges in equality graph
      {
        if (cost_matrix_[x][y] == labels_x_[x] + labels_y_[y] && !T_[y])
        {
          if (yx_[y] == -1) // the augmenting path has been found
          {
            break;
          }
          T_[y] = true;
          q[wr++] = yx_[y]; // add vertex yx[y], which is matched with y, to the queue
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

    UpdateLabels(); // augmenting path not found, so improve labeling

    wr = rd = 0;
    for (y = 0; y < n_; ++y)
    {
      // in this cycle we add edges that were added to the equality graph as a
      // result of improving the labeling, we add edge (slackx[y], y) to the tree if
      // and only if !T[y] && slack[y] == 0, also with this edge we add another one
      // (y, yx[y]) or augment the matching, if y was exposed
      if (!T_[y] && slack_[y] == 0)
      {
        if (yx_[y] == -1) // the augmenting path has been found
        {
          x = slack_x_[y];
          break;
        } else
        {
          T_[y] = true;
          if (!S_[yx_[y]])
          {
            q[wr++] = yx_[y]; // add vertex yx[y], which is matched with y, to the queue
            AddToTree(yx_[y], slack_x_[y]); // and add edges (x,y) and (y,yx[y]) to the tree
          }
        }
      }
    } // y

    if (y < n_) // the augmenting path has been found
    {
      break;
    }
  }

  if (y < n_)
  {
    ++max_match_;
    // in this cycle we inverse edges along augmenting path
    for (int cx = x, cy = y, ty; cx != -2; cx = prev_[cx], cy = ty)
    {
      ty = xy_[cx];
      yx_[cy] = cx;
      xy_[cx] = cy;
    }
    AugmentRecursively();
  }
}

void HungarianAlgorithm::AugmentSequentially()
{
  while (max_match_ < n_)
  {
    int x = 0, y = 0, root = 0;
    std::vector<int> q(n_, 0); // BFS queue for the alternating tree
    int wr = 0, rd = 0; // counters for the queue
    std::fill(S_.begin(), S_.end(), false);
    std::fill(T_.begin(), T_.end(), false);
    std::fill(prev_.begin(), prev_.end(), -1);
    for (x = 0; x < n_; ++x) //finding root of the tree
    {
      if (xy_[x] == -1)
      {
        q[wr++] = root = x;
        prev_[x] = -2;
        S_[x] = true;
        break;
      }
    }

    for (y = 0; y < n_; ++y) //initializing slack array
    {
      slack_[y] = labels_x_[root] + labels_y_[y] - cost_matrix_[root][y];
      slack_x_[y] = root;
    }

    while (true)
    {
      while (rd < wr) //building tree with bfs cycle
      {
        x = q[rd++]; // current vertex from X part
        for (y = 0; y < n_; ++y) // iterate through all edges in equality graph
        {
          if (cost_matrix_[x][y] == labels_x_[x] + labels_y_[y] && !T_[y])
          {
            if (yx_[y] == -1) // the augmenting path has been found
            {
              break;
            }
            T_[y] = true;
            q[wr++] = yx_[y]; // add vertex yx[y], which is matched with y, to the queue
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

      UpdateLabels(); // augmenting path not found, so improve labeling

      wr = rd = 0;
      for (y = 0; y < n_; ++y)
      {
        // in this cycle we add edges that were added to the equality graph as a
        // result of improving the labeling, we add edge (slackx[y], y) to the tree if
        // and only if !T[y] && slack[y] == 0, also with this edge we add another one
        // (y, yx[y]) or augment the matching, if y was exposed
        if (!T_[y] && slack_[y] == 0)
        {
          if (yx_[y] == -1) // the augmenting path has been found
          {
            x = slack_x_[y];
            break;
          } else
          {
            T_[y] = true;
            if (!S_[yx_[y]])
            {
              q[wr++] = yx_[y]; // add vertex yx[y], which is matched with y, to the queue
              AddToTree(yx_[y], slack_x_[y]); // and add edges (x,y) and (y,yx[y]) to the tree
            }
          }
        }
      }

      if (y < n_) // the augmenting path has been found
      {
        break;
      }
    } // y

    if (y < n_)
    {
      ++max_match_;
      // in this cycle we inverse edges along augmenting path
      for (int cx = x, cy = y, ty; cx != -2; cx = prev_[cx], cy = ty)
      {
        ty = xy_[cx];
        yx_[cy] = cx;
        xy_[cx] = cy;
      }
    }
  }
}

void HungarianAlgorithm::UpdateLabels()
{
  CostInt alpha_l = std::numeric_limits<CostInt>::max();

  for (int y = 0; y < n_; ++y) // calculate delta using slack
  {
    if (!T_[y])
    {
      alpha_l = std::min(alpha_l, slack_[y]);
    }
  }

  for (int x = 0; x < n_; ++x) // update X labels
  {
    if (S_[x])
    {
      labels_x_[x] -= alpha_l;
    }
  }

  for (int y = 0; y < n_; ++y) // update Y labels
  {
    if (T_[y])
    {
      labels_y_[y] += alpha_l;
    }
  }

  for (int y = 0; y < n_; ++y) // update slack array
  {
    if (!T_[y])
    {
      slack_[y] -= alpha_l;
    }
  }
}

// x - current vertex, prevx - vertex from X before x in the alternating path
// so we add edges (prevx, xy[x]), (xy[x], x)
void HungarianAlgorithm::AddToTree(int x, int prev_x)
{
  S_[x] = true; // add x to S
  prev_[x] = prev_x;  // we need this when augmenting
  for (int y = 0; y < n_; ++y) // update slacks, because we add new vertex to S
  {
    if (labels_x_[x] + labels_y_[y] - cost_matrix_[x][y] < slack_[y])
    {
      slack_[y] = labels_x_[x] + labels_y_[y] - cost_matrix_[x][y];
      slack_x_[y] = x;
    }
  }
}