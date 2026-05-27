#pragma once

#include "az/types.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace az {

struct MCTSNode {
  int N = 0;
  float W = 0.0f;
  float Q = 0.0f;
  float P = 0.0f;
  bool expanded = false;
  bool terminal = false;
  float terminal_value = 0.0f;  // from side-to-move at this node
  std::unordered_map<int, std::shared_ptr<MCTSNode>> children;

  float ucb_score(float parent_sum_n, float c_puct) const;
};

}  // namespace az
