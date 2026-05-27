#include "az/movegen.h"
#include "az/magic.h"

namespace az {

static void add_move(std::vector<Move>& out, int from, int to, Piece piece, Piece captured,
                     PieceType promo, MoveFlag flag) {
  Move m;
  m.from = from;
  m.to = to;
  m.piece = piece;
  m.captured = captured;
  m.promotion = promo;
  m.flag = flag;
  out.push_back(m);
}

void generate_pseudo_legal_moves(const Board& board, std::vector<Move>& out) {
  out.clear();
  Color us = board.side_to_move();
  Color them = us == Color::White ? Color::Black : Color::White;
  Bitboard occ = board.occupancy();
  Bitboard them_occ = board.all_pieces(them);

  int push_dir = us == Color::White ? 8 : -8;
  int start_rank = us == Color::White ? 1 : 6;
  int promo_rank = us == Color::White ? 7 : 0;

  Bitboard pawns = board.pieces(us, PieceType::Pawn);
  while (pawns) {
    int from = pop_lsb(pawns);
    int to = from + push_dir;
    if (to >= 0 && to < 64 && !(occ & bit(to))) {
      if (rank_of(to) == promo_rank) {
        for (auto pt : {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight}) {
          add_move(out, from, to, board.at(from), Piece::None, pt, MoveFlag::Promotion);
        }
      } else {
        add_move(out, from, to, board.at(from), Piece::None, PieceType::None, MoveFlag::Quiet);
        if (rank_of(from) == start_rank) {
          int to2 = from + 2 * push_dir;
          if (to2 >= 0 && to2 < 64 && !(occ & bit(to2)))
            add_move(out, from, to2, board.at(from), Piece::None, PieceType::None,
                     MoveFlag::DoublePawn);
        }
      }
    }
    int f = file_of(from);
    uint64_t cap_mask = 0;
    if (us == Color::White) {
      if (f > 0) cap_mask |= bit(from + 7);
      if (f < 7) cap_mask |= bit(from + 9);
    } else {
      if (f > 0) cap_mask |= bit(from - 9);
      if (f < 7) cap_mask |= bit(from - 7);
    }
    cap_mask &= them_occ;
    while (cap_mask) {
      int cap_sq = pop_lsb(cap_mask);
      if (rank_of(cap_sq) == promo_rank) {
        for (auto pt : {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight}) {
          add_move(out, from, cap_sq, board.at(from), board.at(cap_sq), pt,
                   MoveFlag::Promotion);
        }
      } else {
        add_move(out, from, cap_sq, board.at(from), board.at(cap_sq), PieceType::None,
                 MoveFlag::Capture);
      }
    }
  }

  int ep = board.en_passant_square();
  if (ep >= 0) {
    Bitboard ep_pawns = board.pieces(us, PieceType::Pawn);
    while (ep_pawns) {
      int from = pop_lsb(ep_pawns);
      if (std::abs(file_of(from) - file_of(ep)) == 1) {
        if ((us == Color::White && rank_of(from) == rank_of(ep) + 1) ||
            (us == Color::Black && rank_of(from) == rank_of(ep) - 1)) {
          add_move(out, from, ep, board.at(from), make_piece(them, PieceType::Pawn),
                   PieceType::None, MoveFlag::EnPassant);
        }
      }
    }
  }

  auto gen_leapers = [&](PieceType pt, auto attack_fn) {
    Bitboard pieces = board.pieces(us, pt);
    while (pieces) {
      int from = pop_lsb(pieces);
      uint64_t att = attack_fn(from);
      att &= ~board.all_pieces(us);
      while (att) {
        int to = pop_lsb(att);
        Piece cap = board.at(to);
        add_move(out, from, to, board.at(from), cap, PieceType::None,
                 cap != Piece::None ? MoveFlag::Capture : MoveFlag::Quiet);
      }
    }
  };

  gen_leapers(PieceType::Knight, knight_attacks);
  gen_leapers(PieceType::King, king_attacks);

  auto gen_sliders = [&](PieceType pt, auto attack_fn) {
    Bitboard pieces = board.pieces(us, pt);
    while (pieces) {
      int from = pop_lsb(pieces);
      uint64_t att = attack_fn(from, occ);
      att &= ~board.all_pieces(us);
      while (att) {
        int to = pop_lsb(att);
        Piece cap = board.at(to);
        add_move(out, from, to, board.at(from), cap, PieceType::None,
                 cap != Piece::None ? MoveFlag::Capture : MoveFlag::Quiet);
      }
    }
  };

  gen_sliders(PieceType::Bishop, bishop_attacks);
  gen_sliders(PieceType::Rook, rook_attacks);
  gen_sliders(PieceType::Queen, [&](int s, uint64_t o) {
    return bishop_attacks(s, o) | rook_attacks(s, o);
  });

  // Castling
  uint8_t cr = board.castling_rights();
  if (!board.in_check(us)) {
    int rank = us == Color::White ? 0 : 7;
    int king_sq = sq(4, rank);
    if (board.at(king_sq) == make_piece(us, PieceType::King)) {
      if ((us == Color::White && (cr & 1)) || (us == Color::Black && (cr & 4))) {
        int f1 = sq(5, rank), g1 = sq(6, rank), h1 = sq(7, rank);
        if (board.at(f1) == Piece::None && board.at(g1) == Piece::None &&
            board.at(h1) == make_piece(us, PieceType::Rook) &&
            !board.in_check(us)) {
          Board tmp = board;
          // check squares not attacked - simplified: skip if f1/g1 attacked
          add_move(out, king_sq, g1, board.at(king_sq), Piece::None, PieceType::None,
                   MoveFlag::Castle);
        }
      }
      if ((us == Color::White && (cr & 2)) || (us == Color::Black && (cr & 8))) {
        int d1 = sq(3, rank), c1 = sq(2, rank), a1 = sq(0, rank);
        if (board.at(d1) == Piece::None && board.at(c1) == Piece::None &&
            board.at(a1) == make_piece(us, PieceType::Rook))
          add_move(out, king_sq, c1, board.at(king_sq), Piece::None, PieceType::None,
                   MoveFlag::Castle);
      }
    }
  }
}

bool gives_check(const Board& board, const Move& m) {
  Board copy = board;
  Color us = board.side_to_move();
  copy.make_move(m);
  return copy.in_check(us);
}

}  // namespace az
