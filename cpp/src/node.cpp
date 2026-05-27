#include "az/node.h"

#include <cmath>
#include <limits>

namespace az {

float MCTSNode::ucb_score(float parent_sum_n, float c_puct) const {
  if (N == 0) return std::numeric_limits<float>::infinity();
  return Q + c_puct * P * std::sqrt(parent_sum_n) / (1.0f + static_cast<float>(N));
}

}  // namespace az
