#pragma once

#include "az/board.h"
#include "az/types.h"

#include <vector>

namespace az {

// AlphaZero-style 119-plane encoding (8*8 spatial)
std::vector<float> encode(const Board& board);

// Map move to policy index (0..4671), -1 if invalid
int move_to_index(const Board& board, const Move& move);

// Decode policy index to move if legal
Move index_to_move(const Board& board, int index);

std::vector<int> legal_move_indices(const Board& board);

void mask_policy(std::vector<float>& policy, const Board& board);

}  // namespace az
