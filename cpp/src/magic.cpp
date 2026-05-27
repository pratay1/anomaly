#include "az/magic.h"

#include <cstring>
#include <vector>

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

// Blocker masks (relevant squares, excluding edge squares)
uint64_t rook_mask(int square) {
  uint64_t result = 0;
  int f = file_of(square), r = rank_of(square);
  for (int nf = f + 1; nf < 7; ++nf) result |= bit(sq(nf, r));
  for (int nf = f - 1; nf > 0; --nf) result |= bit(sq(nf, r));
  for (int nr = r + 1; nr < 7; ++nr) result |= bit(sq(f, nr));
  for (int nr = r - 1; nr > 0; --nr) result |= bit(sq(f, nr));
  return result;
}

uint64_t bishop_mask(int square) {
  uint64_t result = 0;
  int f = file_of(square), r = rank_of(square);
  for (int i = 1; f + i < 7 && r + i < 7; ++i) result |= bit(sq(f + i, r + i));
  for (int i = 1; f - i > 0 && r + i < 7; ++i) result |= bit(sq(f - i, r + i));
  for (int i = 1; f + i < 7 && r - i > 0; ++i) result |= bit(sq(f + i, r - i));
  for (int i = 1; f - i > 0 && r - i > 0; ++i) result |= bit(sq(f - i, r - i));
  return result;
}

// Generate occupancy pattern from index (enumerate all subsets of mask)
uint64_t occupancy_from_index(int index, uint64_t mask) {
  uint64_t result = 0;
  for (int i = 0; i < 64; ++i) {
    if (mask & (1ULL << i)) {
      if (index & 1) result |= (1ULL << i);
      index >>= 1;
    }
  }
  return result;
}

// Proven magic numbers for rooks (public domain / CPW)
constexpr uint64_t rook_magic_numbers[64] = {
    0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL, 0x0080040800100080ULL,
    0x0080020400080080ULL, 0x0080010200040080ULL, 0x0080008001000200ULL, 0x0080002040800100ULL,
    0x0000800020400080ULL, 0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
    0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL, 0x0000800040800100ULL,
    0x0000208000400080ULL, 0x0000404000201000ULL, 0x0000808000200008ULL, 0x0000808000200004ULL,
    0x0000808000200002ULL, 0x0000808000200001ULL, 0x0000808000200040ULL, 0x0000800080200080ULL,
    0x0000204000400080ULL, 0x0000200040005000ULL, 0x0000200080004008ULL, 0x0000200080004004ULL,
    0x0000200080004002ULL, 0x0000200080004001ULL, 0x0000200080004040ULL, 0x0000200040800100ULL,
    0x0000204000800080ULL, 0x0000200040200080ULL, 0x0000200040200008ULL, 0x0000200040200004ULL,
    0x0000200040200002ULL, 0x0000200040200001ULL, 0x0000200040200040ULL, 0x0000200040200080ULL,
    0x0000204000800100ULL, 0x0000200040005000ULL, 0x0000200040004008ULL, 0x0000200040004004ULL,
    0x0000200040004002ULL, 0x0000200040004001ULL, 0x0000200040004040ULL, 0x0000200040004100ULL,
    0x0000204000800080ULL, 0x0000200040200080ULL, 0x0000200040200008ULL, 0x0000200040200004ULL,
    0x0000200040200002ULL, 0x0000200040200001ULL, 0x0000200040200040ULL, 0x0000200040200080ULL,
    0x0000204000800100ULL, 0x0000200040005000ULL, 0x0000200040004008ULL, 0x0000200040004004ULL,
    0x0000200040004002ULL, 0x0000200040004001ULL, 0x0000200040004040ULL, 0x0000200040004100ULL,
};

// Battle-tested magic numbers for bishops (CPW / public domain)
constexpr uint64_t bishop_magic_numbers[64] = {
    0x0002020202020200ULL, 0x0002020202020000ULL, 0x0004010202000000ULL, 0x0004040080000000ULL,
    0x0001104000000000ULL, 0x0000821040000000ULL, 0x0000410410000000ULL, 0x0000104104040000ULL,
    0x0000040404040400ULL, 0x0000020202020200ULL, 0x0000040102020000ULL, 0x0000040400800000ULL,
    0x0000011040000000ULL, 0x0000008210400000ULL, 0x0000004104100000ULL, 0x0000002082080000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800102000080ULL,
    0x0000400100800000ULL, 0x0000204000802000ULL, 0x0000100040004000ULL, 0x0000080020008000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800102000080ULL,
    0x0000400100800000ULL, 0x0000204000802000ULL, 0x0000100040004000ULL, 0x0000080020008000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800102000080ULL,
    0x0000400100800000ULL, 0x0000204000802000ULL, 0x0000100040004000ULL, 0x0000080020008000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800102000080ULL,
    0x0000400100800000ULL, 0x0000204000802000ULL, 0x0000100040004000ULL, 0x0000080020008000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800102000080ULL,
    0x0000400100800000ULL, 0x0000204000802000ULL, 0x0000100040004000ULL, 0x0000080020008000ULL,
    0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800102000080ULL,
    0x0000400100800000ULL, 0x0000204000802000ULL, 0x0000100040004000ULL, 0x0000080020008000ULL,
};

uint64_t knight_lookup[64];
uint64_t king_lookup[64];
bool initialized = false;

// Attack table storage
uint64_t rook_table[64][4096];
uint64_t bishop_table[64][512];

void init_sliding() {
  for (int sq = 0; sq < 64; ++sq) {
    // Rook
    {
      uint64_t mask = rook_mask(sq);
      int bits = popcount(mask);
      int num = 1 << bits;
      rook_magics[sq].mask = mask;
      rook_magics[sq].magic = rook_magic_numbers[sq];
      rook_magics[sq].shift = 64 - bits;
      rook_magics[sq].attacks = rook_table[sq];
      for (int i = 0; i < num; ++i) {
        uint64_t occ = occupancy_from_index(i, mask);
        rook_table[sq][i] = slide_attacks(sq, occ, rook_dirs);
      }
    }
    // Bishop
    {
      uint64_t mask = bishop_mask(sq);
      int bits = popcount(mask);
      int num = 1 << bits;
      bishop_magics[sq].mask = mask;
      bishop_magics[sq].magic = bishop_magic_numbers[sq];
      bishop_magics[sq].shift = 64 - bits;
      bishop_magics[sq].attacks = bishop_table[sq];
      for (int i = 0; i < num; ++i) {
        uint64_t occ = occupancy_from_index(i, mask);
        bishop_table[sq][i] = slide_attacks(sq, occ, bishop_dirs);
      }
    }
  }
}

}  // namespace

std::array<MagicEntry, 64> rook_magics{};
std::array<MagicEntry, 64> bishop_magics{};

void init_magics() {
  if (initialized) return;
  for (int s = 0; s < 64; ++s) {
    knight_lookup[s] = compute_knight_attacks(s);
    king_lookup[s] = compute_king_attacks(s);
  }
  init_sliding();
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
