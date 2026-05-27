#include "az/selfplay.h"

#include "az/encoding.h"

#include <algorithm>
#include <random>

namespace az {

SelfPlayRunner::SelfPlayRunner(InferenceQueue* queue, const SelfPlayConfig& cfg)
    : queue_(queue), cfg_(cfg) {}

std::vector<TrainingExample> SelfPlayRunner::play_game(MoveCallback move_cb) {
  Board board;
  MCTS mcts(queue_, cfg_.mcts);
  struct Snap {
    std::vector<float> state;
    std::vector<float> pi;
    Color stm;
  };
  std::vector<Snap> trajectory;
  std::mt19937 rng(std::random_device{}());

  for (int ply = 0; ply < cfg_.max_game_length; ++ply) {
    GameResult res = board.result();
    if (res != GameResult::Ongoing) break;

    Color stm = board.side_to_move();
    float temp = ply < cfg_.temperature_moves ? 1.0f : 0.1f;
    auto pi = mcts.run(board, temp);
    auto visits = mcts.root_visits(board);

    trajectory.push_back({encode(board), pi, stm});

    if (move_cb) move_cb(board.fen(), Move{}, visits);

    // Sample move from pi
    std::vector<int> legal = legal_move_indices(board);
    std::vector<float> probs;
    std::vector<int> indices;
    for (int idx : legal) {
      if (pi[static_cast<size_t>(idx)] > 0) {
        probs.push_back(pi[static_cast<size_t>(idx)]);
        indices.push_back(idx);
      }
    }
    if (indices.empty()) break;

    int chosen_idx;
    if (temp < 0.5f) {
      chosen_idx = indices[std::distance(probs.begin(),
                                         std::max_element(probs.begin(), probs.end()))];
    } else {
      std::discrete_distribution<int> dist(probs.begin(), probs.end());
      chosen_idx = indices[dist(rng)];
    }

    Move m = index_to_move(board, chosen_idx);
    if (move_cb) move_cb(board.fen(), m, visits);
    board.make_move(m);
  }

  GameResult result = board.result();
  if (result == GameResult::Ongoing) result = GameResult::Draw;

  std::vector<TrainingExample> examples;
  for (const auto& snap : trajectory) {
    TrainingExample ex;
    ex.state = snap.state;
    ex.policy = snap.pi;
    if (result == GameResult::Draw) {
      ex.value = 0.0f;
    } else if (result == GameResult::WhiteWin) {
      ex.value = snap.stm == Color::White ? 1.0f : -1.0f;
    } else {
      ex.value = snap.stm == Color::Black ? 1.0f : -1.0f;
    }
    examples.push_back(std::move(ex));
  }
  return examples;
}

void SelfPlayRunner::run_games(int n, std::vector<std::vector<TrainingExample>>& out,
                               GameCallback game_cb, MoveCallback move_cb) {
  out.clear();
  out.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    auto ex = play_game(move_cb);
    out.push_back(ex);
    if (game_cb) {
      GameResult res = GameResult::Draw;
      if (!ex.empty()) {
        if (ex.back().value > 0) res = GameResult::WhiteWin;
        else if (ex.back().value < 0) res = GameResult::BlackWin;
      }
      game_cb(ex, res, static_cast<int>(ex.size()));
    }
  }
}

}  // namespace az
