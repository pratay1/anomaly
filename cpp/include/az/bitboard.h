#pragma once

#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace az {

constexpr int FILE_A = 0;
constexpr int RANK_1 = 0;

inline int sq(int file, int rank) { return rank * 8 + file; }
inline int file_of(int s) { return s & 7; }
inline int rank_of(int s) { return s >> 3; }

inline uint64_t bit(int square) { return 1ULL << square; }

constexpr uint64_t FILE_A_BB = 0x0101010101010101ULL;
constexpr uint64_t FILE_H_BB = 0x8080808080808080ULL;
constexpr uint64_t RANK_1_BB = 0x00000000000000FFULL;
constexpr uint64_t RANK_8_BB = 0xFF00000000000000ULL;

constexpr uint64_t NOT_A_FILE = ~FILE_A_BB;
constexpr uint64_t NOT_H_FILE = ~FILE_H_BB;

inline int ctz(uint64_t bb) {
#ifdef _MSC_VER
  unsigned long idx;
  _BitScanForward64(&idx, bb);
  return static_cast<int>(idx);
#else
  return __builtin_ctzll(bb);
#endif
}

inline int pop_lsb(uint64_t& bb) {
  const int s = ctz(bb);
  bb &= bb - 1;
  return s;
}

inline int popcount(uint64_t bb) {
#ifdef _MSC_VER
  return static_cast<int>(__popcnt64(bb));
#else
  return __builtin_popcountll(bb);
#endif
}

}  // namespace az
