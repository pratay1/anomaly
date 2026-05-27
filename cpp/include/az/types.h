#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace az {

enum class Color : uint8_t { White = 0, Black = 1 };
enum class PieceType : uint8_t {
  None = 0,
  Pawn,
  Knight,
  Bishop,
  Rook,
  Queen,
  King
};

enum class Piece : uint8_t {
  None = 0,
  WP, WN, WB, WR, WQ, WK,
  BP, BN, BB, BR, BQ, BK
};

constexpr int NUM_SQUARES = 64;
constexpr int NUM_MOVE_PLANES = 73;
constexpr int POLICY_SIZE = NUM_SQUARES * NUM_MOVE_PLANES;  // 4672
constexpr int ENCODING_CHANNELS = 119;

using Bitboard = uint64_t;

inline Color color_of(Piece p) {
  return (static_cast<uint8_t>(p) >= static_cast<uint8_t>(Piece::BP))
             ? Color::Black
             : Color::White;
}

inline PieceType type_of(Piece p) {
  switch (p) {
    case Piece::WP: case Piece::BP: return PieceType::Pawn;
    case Piece::WN: case Piece::BN: return PieceType::Knight;
    case Piece::WB: case Piece::BB: return PieceType::Bishop;
    case Piece::WR: case Piece::BR: return PieceType::Rook;
    case Piece::WQ: case Piece::BQ: return PieceType::Queen;
    case Piece::WK: case Piece::BK: return PieceType::King;
    default: return PieceType::None;
  }
}

inline Piece make_piece(Color c, PieceType t) {
  static constexpr Piece table[2][7] = {
      {Piece::None, Piece::WP, Piece::WN, Piece::WB, Piece::WR, Piece::WQ, Piece::WK},
      {Piece::None, Piece::BP, Piece::BN, Piece::BB, Piece::BR, Piece::BQ, Piece::BK}};
  return table[static_cast<int>(c)][static_cast<int>(t)];
}

enum MoveFlag : uint8_t {
  Quiet = 0,
  Capture = 1,
  DoublePawn = 2,
  EnPassant = 3,
  Castle = 4,
  Promotion = 8
};

struct Move {
  int from = 0;
  int to = 0;
  Piece piece = Piece::None;
  Piece captured = Piece::None;
  PieceType promotion = PieceType::None;
  MoveFlag flag = MoveFlag::Quiet;

  bool operator==(const Move& o) const {
    return from == o.from && to == o.to && promotion == o.promotion && flag == o.flag;
  }
};

struct PolicyValue {
  std::vector<float> policy;  // size POLICY_SIZE
  float value = 0.0f;
};

struct TrainingExample {
  std::vector<float> state;  // ENCODING_CHANNELS * 64
  std::vector<float> policy; // POLICY_SIZE
  float value = 0.0f;
};

enum GameResult : int8_t { Ongoing = 0, WhiteWin = 1, BlackWin = -1, Draw = 2 };

}  // namespace az
