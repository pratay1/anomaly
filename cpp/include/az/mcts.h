#pragma once

#include "az/board.h"
#include "az/inference_queue.h"
#include "az/node.h"
#include "az/types.h"

#include <memory>
#include <random>
#include <vector>

namespace az {

struct MCTSConfig {
  int num_simulations = 200;
  float c_puct_base = 19652.0f;
  float c_puct_init = 1.25f;
  float dirichlet_alpha = 0.3f;
  float dirichlet_eps = 0.25f;
  bool add_root_noise = true;
  int virtual_loss = 1;
};

struct RootVisit {
  int move_index = -1;
  Move move;
  int N = 0;
  float Q = 0.0f;
  float P = 0.0f;
};

class MCTS {
 public:
  MCTS(InferenceQueue* queue, const MCTSConfig& cfg);

  /// think_time_ms > 0: search until deadline; else run cfg_.num_simulations.
  std::vector<float> run(Board& board, float temperature = 1.0f, int think_time_ms = 0);
  /// Reuse search tree after playing move_index (skips rebuilding root each ply).
  void advance_root(int move_index);
  void reset_tree();
  std::vector<RootVisit> root_visits(const Board& board) const;

 private:
  float c_puct(float sum_n) const;
  std::shared_ptr<MCTSNode> select_child(std::shared_ptr<MCTSNode> node, float sum_n);
  float expand_and_evaluate(Board& board, std::shared_ptr<MCTSNode> node);
  void backup(std::vector<std::pair<std::shared_ptr<MCTSNode>, int>>& path, float value);

  InferenceQueue* queue_;
  MCTSConfig cfg_;
  std::shared_ptr<MCTSNode> root_;
  std::mt19937 rng_;
};

}  // namespace az
