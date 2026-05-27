#include "az/mcts.h"

#include "az/encoding.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace az {

MCTS::MCTS(InferenceQueue* queue, const MCTSConfig& cfg)
    : queue_(queue), cfg_(cfg), rng_(std::random_device{}()) {}

float MCTS::c_puct(float sum_n) const {
  return std::log((1.0f + sum_n + cfg_.c_puct_base) / cfg_.c_puct_base) + cfg_.c_puct_init;
}

std::shared_ptr<MCTSNode> MCTS::select_child(std::shared_ptr<MCTSNode> node, float sum_n) {
  float cp = c_puct(sum_n);
  int best_key = -1;
  float best_score = -1e30f;
  for (auto& [key, child] : node->children) {
    float score = child->ucb_score(sum_n, cp);
    if (score > best_score) {
      best_score = score;
      best_key = key;
    }
  }
  return node->children.at(best_key);
}

float MCTS::expand_and_evaluate(Board& board, std::shared_ptr<MCTSNode> node) {
  GameResult res = board.result();
  if (res != GameResult::Ongoing) {
    node->terminal = true;
    if (res == GameResult::Draw) node->terminal_value = 0.0f;
    else if (res == GameResult::WhiteWin)
      node->terminal_value = board.side_to_move() == Color::White ? -1.0f : 1.0f;
    else
      node->terminal_value = board.side_to_move() == Color::White ? 1.0f : -1.0f;
    return node->terminal_value;
  }

  auto state = encode(board);
  PolicyValue pv = queue_->evaluate(state);
  mask_policy(pv.policy, board);

  std::vector<Move> legal;
  board.generate_legal_moves(legal);
  for (const auto& m : legal) {
    int idx = move_to_index(board, m);
    if (idx < 0) continue;
    auto child = std::make_shared<MCTSNode>();
    child->P = pv.policy[static_cast<size_t>(idx)];
    node->children[idx] = child;
  }

  node->expanded = true;
  return pv.value;
}

void MCTS::backup(std::vector<std::pair<std::shared_ptr<MCTSNode>, int>>& path, float value) {
  float v = value;
  for (auto it = path.rbegin(); it != path.rend(); ++it) {
    auto& node = it->first;
    node->N += 1;
    node->W += v;
    node->Q = node->W / node->N;
    v = -v;
  }
}

void MCTS::advance_root(int move_index) {
  if (!root_ || !root_->children.count(move_index)) {
    root_.reset();
    return;
  }
  root_ = root_->children[move_index];
}

void MCTS::reset_tree() { root_.reset(); }

std::vector<float> MCTS::run(Board& board, float temperature) {
  if (!root_) {
    root_ = std::make_shared<MCTSNode>();
  }

  for (int sim = 0; sim < cfg_.num_simulations; ++sim) {
    Board sim_board = board;
    auto node = root_;
    std::vector<std::pair<std::shared_ptr<MCTSNode>, int>> path;

    while (node->expanded && !node->terminal && !node->children.empty()) {
      float sum_n = static_cast<float>(node->N);
      auto child = select_child(node, sum_n);
      int move_idx = -1;
      for (auto& [k, c] : node->children) {
        if (c.get() == child.get()) {
          move_idx = k;
          break;
        }
      }
      path.emplace_back(node, move_idx);
      Move m = index_to_move(sim_board, move_idx);
      sim_board.make_move(m);
      node = child;
      node->N += cfg_.virtual_loss;
      node->W -= cfg_.virtual_loss;
    }

    float leaf_value;
    if (node->terminal) {
      leaf_value = node->terminal_value;
    } else if (!node->expanded) {
      leaf_value = expand_and_evaluate(sim_board, node);
      if (cfg_.add_root_noise && node == root_ && path.empty()) {
        std::gamma_distribution<float> gamma(cfg_.dirichlet_alpha, 1.0f);
        std::vector<float> noise;
        float sum = 0.0f;
        for (auto& [k, ch] : node->children) {
          float n = gamma(rng_);
          noise.push_back(n);
          sum += n;
        }
        size_t i = 0;
        for (auto& [k, ch] : node->children) {
          float noisy = (1.0f - cfg_.dirichlet_eps) * ch->P +
                        cfg_.dirichlet_eps * noise[i++] / sum;
          ch->P = noisy;
        }
      }
    } else {
      leaf_value = node->Q;
    }

    for (auto& [n, idx] : path) {
      if (idx >= 0 && n->children.count(idx)) {
        auto& ch = n->children[idx];
        ch->N -= cfg_.virtual_loss;
        ch->W += cfg_.virtual_loss;
      }
    }

    backup(path, leaf_value);
  }

  std::vector<float> pi(POLICY_SIZE, 0.0f);
  float sum_visits = 0.0f;
  for (auto& [idx, ch] : root_->children) {
    float v = static_cast<float>(ch->N);
    if (temperature < 1e-6f) {
      pi[static_cast<size_t>(idx)] = v;
    } else {
      pi[static_cast<size_t>(idx)] = std::pow(v, 1.0f / temperature);
    }
    sum_visits += pi[static_cast<size_t>(idx)];
  }
  if (sum_visits > 0) {
    for (auto& [idx, ch] : root_->children) pi[static_cast<size_t>(idx)] /= sum_visits;
  }
  return pi;
}

std::vector<RootVisit> MCTS::root_visits(const Board& board) const {
  std::vector<RootVisit> visits;
  if (!root_) return visits;
  for (auto& [idx, ch] : root_->children) {
    RootVisit rv;
    rv.move_index = idx;
    rv.move = index_to_move(board, idx);
    rv.N = ch->N;
    rv.Q = ch->Q;
    rv.P = ch->P;
    visits.push_back(rv);
  }
  std::sort(visits.begin(), visits.end(),
            [](const RootVisit& a, const RootVisit& b) { return a.N > b.N; });
  return visits;
}

}  // namespace az
