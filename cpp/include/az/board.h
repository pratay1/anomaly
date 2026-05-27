#pragma once

#include "az/types.h"
#include "az/bitboard.h"

#include <array>
#include <string>
#include <vector>

namespace az {

struct Undo {
  Move move;
  Piece captured;
  uint8_t castling_rights;
  int en_passant_sq;
  int halfmove_clock;
  uint64_t hash;
  int repetition_count;
};

class Board {
 public:
  Board();

  static Board from_fen(const std::string& fen);

  std::string fen() const;
  std::string pretty() const;

  Color side_to_move() const { return stm_; }
  int fullmove_number() const { return fullmove_; }
  int halfmove_clock() const { return halfmove_clock_; }
  uint8_t castling_rights() const { return castling_; }
  int en_passant_square() const { return en_passant_sq_; }
  uint64_t hash() const { return hash_; }

  Piece at(int sq) const;
  void set_at(int sq, Piece p);

  Bitboard pieces(Color c, PieceType t) const;
  Bitboard all_pieces(Color c) const;
  Bitboard occupancy() const { return white_ | black_; }

  void generate_legal_moves(std::vector<Move>& out) const;
  bool is_legal(const Move& m) const;
  void make_move(const Move& m);
  void unmake_move(const Move& m);

  bool in_check(Color c) const;
  bool square_attacked(int square, Color by) const;
  bool is_checkmate() const;
  bool is_stalemate() const;
  bool is_draw() const;
  GameResult result() const;

  bool is_repetition() const;
  bool has_insufficient_material() const;

  // History for encoding (last 8 positions piece placement)
  void push_history();
  const std::vector<std::array<Piece, 64>>& history() const { return history_; }

  int ply() const { return static_cast<int>(undo_stack_.size()); }

 private:
  void load_fen(const std::string& fen);
  void clear();
  void place(Piece p, int sq);
  void remove(int sq);
  void update_hash_remove(Piece p, int sq);
  void update_hash_place(Piece p, int sq);
  void refresh_occupancy();

  std::array<Bitboard, 12> bb_{};
  Bitboard white_ = 0;
  Bitboard black_ = 0;

  std::array<Piece, 64> mailbox_{};

  Color stm_ = Color::White;
  uint8_t castling_ = 0x0F;
  int en_passant_sq_ = -1;
  int halfmove_clock_ = 0;
  int fullmove_ = 1;

  uint64_t hash_ = 0;
  std::vector<Undo> undo_stack_;
  std::vector<std::array<Piece, 64>> history_;

  static void init_zobrist();
  static uint64_t zobrist_piece[64][12];
  static uint64_t zobrist_stm;
  static uint64_t zobrist_castling[16];
  static uint64_t zobrist_ep[64];
  static bool zobrist_ready_;
};

uint64_t perft(Board& board, int depth);

}  // namespace az
