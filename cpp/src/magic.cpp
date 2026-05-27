#include "az/magic.h"

namespace az {

namespace {

constexpr int rook_dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
constexpr int bishop_dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
constexpr int knight_deltas[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2},
                                     {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};

uint64_t slide_attacks(int square, uint64_t occupied, const int dirs[4][2]) {
  uint64_t attacks = 0;
  int f = file_of(square);
  int r = rank_of(square);
  for (int d = 0; d < 4; ++d) {
    int nf = f + dirs[d][0];
    int nr = r + dirs[d][1];
    while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
      int ts = sq(nf, nr);
      attacks |= bit(ts);
      if (occupied & bit(ts)) break;
      nf += dirs[d][0];
      nr += dirs[d][1];
    }
  }
  return attacks;
}

uint64_t compute_knight_attacks(int square) {
  uint64_t att = 0;
  int f = file_of(square);
  int r = rank_of(square);
  for (auto& d : knight_deltas) {
    int nf = f + d[0];
    int nr = r + d[1];
    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) att |= bit(sq(nf, nr));
  }
  return att;
}

uint64_t compute_king_attacks(int square) {
  uint64_t att = 0;
  int f = file_of(square);
  int r = rank_of(square);
  for (int df = -1; df <= 1; ++df) {
    for (int dr = -1; dr <= 1; ++dr) {
      if (df == 0 && dr == 0) continue;
      int nf = f + df;
      int nr = r + dr;
      if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) att |= bit(sq(nf, nr));
    }
  }
  return att;
}

uint64_t knight_lookup[64];
uint64_t king_lookup[64];
bool initialized = false;

}  // namespace

std::array<MagicEntry, 64> rook_magics{};
std::array<MagicEntry, 64> bishop_magics{};

void init_magics() {
  if (initialized) return;
  for (int s = 0; s < 64; ++s) {
    knight_lookup[s] = compute_knight_attacks(s);
    king_lookup[s] = compute_king_attacks(s);
  }
  initialized = true;
}

uint64_t rook_attacks(int square, uint64_t occupied) {
  init_magics();
  return slide_attacks(square, occupied, rook_dirs);
}

uint64_t bishop_attacks(int square, uint64_t occupied) {
  init_magics();
  return slide_attacks(square, occupied, bishop_dirs);
}

uint64_t knight_attacks(int square) {
  init_magics();
  return knight_lookup[square];
}

uint64_t king_attacks(int square) {
  init_magics();
  return king_lookup[square];
}

}  // namespace az
