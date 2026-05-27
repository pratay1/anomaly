#include "az/board.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static bool test_perft(const std::string& fen, int depth, uint64_t expected) {
  az::Board b = az::Board::from_fen(fen);
  uint64_t nodes = az::perft(b, depth);
  if (nodes != expected) {
    std::printf("FAIL fen=%s depth=%d got=%llu expected=%llu\n", fen.c_str(), depth,
                static_cast<unsigned long long>(nodes),
                static_cast<unsigned long long>(expected));
    return false;
  }
  std::printf("OK   fen=%s depth=%d nodes=%llu\n", fen.c_str(), depth,
              static_cast<unsigned long long>(nodes));
  return true;
}

int main() {
  bool ok = true;
  ok &= test_perft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20);
  ok &= test_perft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400);
  ok &= test_perft("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902);
  ok &= test_perft(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48);
  ok &= test_perft(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039);
  return ok ? 0 : 1;
}
