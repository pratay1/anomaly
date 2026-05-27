#pragma once

#include "az/bitboard.h"

#include <array>

namespace az {

struct MagicEntry {
  uint64_t mask;
  uint64_t magic;
  int shift;
  const uint64_t* attacks;
};

void init_magics();

uint64_t rook_attacks(int sq, uint64_t occupied);
uint64_t bishop_attacks(int sq, uint64_t occupied);
uint64_t knight_attacks(int sq);
uint64_t king_attacks(int sq);

extern std::array<MagicEntry, 64> rook_magics;
extern std::array<MagicEntry, 64> bishop_magics;

}  // namespace az
