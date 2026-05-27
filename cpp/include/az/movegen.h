#pragma once

#include "az/board.h"

namespace az {

void generate_pseudo_legal_moves(const Board& board, std::vector<Move>& out);
bool gives_check(const Board& board, const Move& m);

}  // namespace az
