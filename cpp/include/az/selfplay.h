#pragma once

#include "az/inference_queue.h"
#include "az/mcts.h"
#include "az/types.h"

#include <functional>
#include <vector>

namespace az {

struct SelfPlayConfig {
  MCTSConfig mcts;
  int temperature_moves = 15;
  int max_game_length = 512;
};

using MoveCallback = std::function<void(const std::string& fen, const Move& move,
                                        const std::vector<RootVisit>& visits)>;
using GameCallback = std::function<void(const std::vector<TrainingExample>& examples,
                                         GameResult result, int plies)>;

class SelfPlayRunner {
 public:
  SelfPlayRunner(InferenceQueue* queue, const SelfPlayConfig& cfg);

  std::vector<TrainingExample> play_game(MoveCallback move_cb = nullptr);
  void run_games(int n, std::vector<std::vector<TrainingExample>>& out,
                 GameCallback game_cb = nullptr, MoveCallback move_cb = nullptr);

 private:
  InferenceQueue* queue_;
  SelfPlayConfig cfg_;
};

}  // namespace az
