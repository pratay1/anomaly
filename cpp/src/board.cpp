#include "az/board.h"
#include "az/movegen.h"
#include "az/magic.h"
#include "az/bitboard.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <sstream>

namespace az {

uint64_t Board::zobrist_piece[64][12];
uint64_t Board::zobrist_stm = 0;
uint64_t Board::zobrist_castling[16] = {};
uint64_t Board::zobrist_ep[64] = {};
bool Board::zobrist_ready_ = false;

namespace {

int piece_to_zobrist_index(Piece p) {
  if (p == Piece::None) return -1;
  return static_cast<int>(p) - 1;
}

uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

}  // namespace

void Board::init_zobrist() {
  if (zobrist_ready_) return;
  uint64_t seed = 0x123456789ABCDEFULL;
  for (int sq = 0; sq < 64; ++sq) {
    for (int p = 0; p < 12; ++p) {
      zobrist_piece[sq][p] = splitmix64(seed);
    }
  }
  zobrist_stm = splitmix64(seed);
  for (int i = 0; i < 16; ++i) zobrist_castling[i] = splitmix64(seed);
  for (int i = 0; i < 64; ++i) zobrist_ep[i] = splitmix64(seed);
  zobrist_ready_ = true;
  init_magics();
}

Board::Board() {
  init_zobrist();
  clear();
  load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::clear() {
  bb_.fill(0);
  white_ = black_ = 0;
  mailbox_.fill(Piece::None);
  stm_ = Color::White;
  castling_ = 0;
  en_passant_sq_ = -1;
  halfmove_clock_ = 0;
  fullmove_ = 1;
  hash_ = 0;
  undo_stack_.clear();
  history_.clear();
}

Piece Board::at(int s) const { return mailbox_[s]; }

void Board::set_at(int s, Piece p) {
  mailbox_[s] = p;
}

Bitboard Board::pieces(Color c, PieceType t) const {
  Piece base = make_piece(c, t);
  return bb_[static_cast<int>(base) - 1];
}

Bitboard Board::all_pieces(Color c) const {
  return c == Color::White ? white_ : black_;
}

void Board::refresh_occupancy() {
  white_ = black_ = 0;
  for (int p = 0; p < 6; ++p) {
    white_ |= bb_[p];
    black_ |= bb_[p + 6];
  }
}

void Board::place(Piece p, int s) {
  if (p == Piece::None) return;
  int idx = static_cast<int>(p) - 1;
  bb_[idx] |= bit(s);
  mailbox_[s] = p;
  update_hash_place(p, s);
  refresh_occupancy();
}

void Board::remove(int s) {
  Piece p = mailbox_[s];
  if (p == Piece::None) return;
  update_hash_remove(p, s);
  int idx = static_cast<int>(p) - 1;
  bb_[idx] &= ~bit(s);
  mailbox_[s] = Piece::None;
  refresh_occupancy();
}

void Board::update_hash_remove(Piece p, int s) {
  int pi = piece_to_zobrist_index(p);
  if (pi >= 0) hash_ ^= zobrist_piece[s][pi];
}

void Board::update_hash_place(Piece p, int s) {
  int pi = piece_to_zobrist_index(p);
  if (pi >= 0) hash_ ^= zobrist_piece[s][pi];
}

void Board::load_fen(const std::string& fen) {
  std::istringstream ss(fen);
  std::string board_part, stm, castling, ep;
  int halfmove = 0, fullmove = 1;
  ss >> board_part >> stm >> castling >> ep >> halfmove >> fullmove;

  int square = 56;
  for (char c : board_part) {
    if (c == '/') {
      square -= 16;
      continue;
    }
    if (c >= '1' && c <= '8') {
      square += c - '0';
      continue;
    }
    Piece p = Piece::None;
    switch (c) {
      case 'P': p = Piece::WP; break;
      case 'N': p = Piece::WN; break;
      case 'B': p = Piece::WB; break;
      case 'R': p = Piece::WR; break;
      case 'Q': p = Piece::WQ; break;
      case 'K': p = Piece::WK; break;
      case 'p': p = Piece::BP; break;
      case 'n': p = Piece::BN; break;
      case 'b': p = Piece::BB; break;
      case 'r': p = Piece::BR; break;
      case 'q': p = Piece::BQ; break;
      case 'k': p = Piece::BK; break;
    }
    if (p != Piece::None) {
      mailbox_[square] = p;
      int idx = static_cast<int>(p) - 1;
      bb_[idx] |= bit(square);
      hash_ ^= zobrist_piece[square][idx];
      ++square;
    }
  }
  refresh_occupancy();

  stm_ = (stm == "b") ? Color::Black : Color::White;
  if (stm_ == Color::Black) hash_ ^= zobrist_stm;

  castling_ = 0;
  if (castling.find('K') != std::string::npos) castling_ |= 1;
  if (castling.find('Q') != std::string::npos) castling_ |= 2;
  if (castling.find('k') != std::string::npos) castling_ |= 4;
  if (castling.find('q') != std::string::npos) castling_ |= 8;
  hash_ ^= zobrist_castling[castling_];

  if (ep == "-") {
    en_passant_sq_ = -1;
  } else {
    int f = ep[0] - 'a';
    int r = ep[1] - '1';
    en_passant_sq_ = sq(f, r);
    hash_ ^= zobrist_ep[en_passant_sq_];
  }

  halfmove_clock_ = halfmove;
  fullmove_ = fullmove;
  push_history();
}

Board Board::from_fen(const std::string& fen) {
  init_zobrist();
  Board b;
  b.clear();
  b.load_fen(fen);
  return b;
}

std::string Board::fen() const {
  std::ostringstream oss;
  for (int r = 7; r >= 0; --r) {
    int empty = 0;
    for (int f = 0; f < 8; ++f) {
      Piece p = at(sq(f, r));
      if (p == Piece::None) {
        ++empty;
      } else {
        if (empty) {
          oss << empty;
          empty = 0;
        }
        char c = '?';
        switch (p) {
          case Piece::WP: c = 'P'; break;
          case Piece::WN: c = 'N'; break;
          case Piece::WB: c = 'B'; break;
          case Piece::WR: c = 'R'; break;
          case Piece::WQ: c = 'Q'; break;
          case Piece::WK: c = 'K'; break;
          case Piece::BP: c = 'p'; break;
          case Piece::BN: c = 'n'; break;
          case Piece::BB: c = 'b'; break;
          case Piece::BR: c = 'r'; break;
          case Piece::BQ: c = 'q'; break;
          case Piece::BK: c = 'k'; break;
          default: break;
        }
        oss << c;
      }
    }
    if (empty) oss << empty;
    if (r > 0) oss << '/';
  }
  oss << (stm_ == Color::White ? " w " : " b ");
  std::string cast;
  if (castling_ & 1) cast += 'K';
  if (castling_ & 2) cast += 'Q';
  if (castling_ & 4) cast += 'k';
  if (castling_ & 8) cast += 'q';
  if (cast.empty()) cast = "-";
  oss << cast << ' ';
  if (en_passant_sq_ < 0) {
    oss << "- ";
  } else {
    oss << char('a' + file_of(en_passant_sq_)) << char('1' + rank_of(en_passant_sq_)) << ' ';
  }
  oss << halfmove_clock_ << ' ' << fullmove_;
  return oss.str();
}

void Board::push_history() {
  std::array<Piece, 64> snap{};
  for (int i = 0; i < 64; ++i) snap[i] = mailbox_[i];
  history_.push_back(snap);
  if (history_.size() > 8) history_.erase(history_.begin());
}

void Board::generate_legal_moves(std::vector<Move>& out) const {
  out.clear();
  std::vector<Move> pseudo;
  generate_pseudo_legal_moves(*this, pseudo);
  Board copy = *this;
  Color us = stm_;
  for (const auto& m : pseudo) {
    copy.make_move(m);
    if (!copy.in_check(us)) out.push_back(m);
    copy.unmake_move(m);
  }
}

bool Board::is_legal(const Move& m) const {
  Board copy = *this;
  if (!copy.in_check(stm_)) {
    // ok
  }
  copy.make_move(m);
  Color mover = stm_ == Color::White ? Color::White : Color::Black;
  return !copy.in_check(mover);
}

void Board::make_move(const Move& m) {
  Undo u;
  u.move = m;
  u.captured = at(m.to);
  u.castling_rights = castling_;
  u.en_passant_sq = en_passant_sq_;
  u.halfmove_clock = halfmove_clock_;
  u.hash = hash_;
  u.repetition_count = 0;

  if (m.captured != Piece::None || type_of(m.piece) == PieceType::Pawn) halfmove_clock_ = 0;
  else ++halfmove_clock_;

  if (en_passant_sq_ >= 0) hash_ ^= zobrist_ep[en_passant_sq_];
  en_passant_sq_ = -1;

  Piece moving = m.piece;
  remove(m.from);
  if (m.flag == MoveFlag::EnPassant) {
    int cap_sq = m.to + (stm_ == Color::White ? -8 : 8);
    u.captured = at(cap_sq);
    remove(cap_sq);
  } else if (u.captured != Piece::None) {
    remove(m.to);
  }

  if (m.flag == MoveFlag::Castle) {
    if (m.to == sq(6, rank_of(m.from))) {  // kingside
      int rook_from = sq(7, rank_of(m.from));
      int rook_to = sq(5, rank_of(m.from));
      Piece rook = at(rook_from);
      remove(rook_from);
      place(rook, rook_to);
    } else {
      int rook_from = sq(0, rank_of(m.from));
      int rook_to = sq(3, rank_of(m.from));
      Piece rook = at(rook_from);
      remove(rook_from);
      place(rook, rook_to);
    }
  }

  PieceType promo = m.promotion;
  if (m.flag == MoveFlag::Promotion || promo != PieceType::None) {
    place(make_piece(stm_, promo == PieceType::None ? PieceType::Queen : promo), m.to);
  } else {
    place(moving, m.to);
  }

  if (type_of(moving) == PieceType::Pawn && std::abs(m.to - m.from) == 16) {
    en_passant_sq_ = m.from + (stm_ == Color::White ? 8 : -8);
    hash_ ^= zobrist_ep[en_passant_sq_];
  }

  if (type_of(moving) == PieceType::King) castling_ &= stm_ == Color::White ? 0xC : 0x3;
  if (m.from == sq(0, 0) || m.to == sq(0, 0)) castling_ &= ~2;
  if (m.from == sq(7, 0) || m.to == sq(7, 0)) castling_ &= ~1;
  if (m.from == sq(0, 7) || m.to == sq(0, 7)) castling_ &= ~8;
  if (m.from == sq(7, 7) || m.to == sq(7, 7)) castling_ &= ~4;

  hash_ ^= zobrist_castling[u.castling_rights ^ castling_];

  hash_ ^= zobrist_stm;
  stm_ = stm_ == Color::White ? Color::Black : Color::White;
  if (stm_ == Color::White) ++fullmove_;

  undo_stack_.push_back(u);
  push_history();
}

void Board::unmake_move(const Move& m) {
  auto u = undo_stack_.back();
  undo_stack_.pop_back();
  if (!history_.empty()) history_.pop_back();

  stm_ = stm_ == Color::White ? Color::Black : Color::White;
  hash_ = u.hash;
  castling_ = u.castling_rights;
  en_passant_sq_ = u.en_passant_sq;
  halfmove_clock_ = u.halfmove_clock;

  // Rebuild from fen-like undo: simpler to restore mailbox from history[-1] and undo piece moves
  // Full state restore via reversing operations
  Piece moving = m.piece;
  remove(m.to);

  if (m.flag == MoveFlag::Castle) {
    if (m.to == sq(6, rank_of(m.from))) {
      int rook_from = sq(7, rank_of(m.from));
      int rook_to = sq(5, rank_of(m.from));
      Piece rook = at(rook_to);
      remove(rook_to);
      place(rook, rook_from);
    } else {
      int rook_from = sq(0, rank_of(m.from));
      int rook_to = sq(3, rank_of(m.from));
      Piece rook = at(rook_to);
      remove(rook_to);
      place(rook, rook_from);
    }
  }

  if (m.flag == MoveFlag::EnPassant) {
    int cap_sq = m.to + (stm_ == Color::White ? 8 : -8);
    place(u.captured, cap_sq);
  } else if (u.captured != Piece::None) {
    place(u.captured, m.to);
  }

  place(moving, m.from);
}

bool Board::in_check(Color c) const {
  Bitboard kings = pieces(c, PieceType::King);
  if (!kings) return false;
  int king_sq = ctz(kings);
  Color enemy = c == Color::White ? Color::Black : Color::White;
  Bitboard occ = occupancy();

  Bitboard kn = pieces(enemy, PieceType::Knight);
  while (kn) {
    int s = pop_lsb(kn);
    if (knight_attacks(s) & bit(king_sq)) return true;
  }

  Bitboard bishops = pieces(enemy, PieceType::Bishop) | pieces(enemy, PieceType::Queen);
  while (bishops) {
    int s = pop_lsb(bishops);
    if (bishop_attacks(s, occ) & bit(king_sq)) return true;
  }

  Bitboard rooks = pieces(enemy, PieceType::Rook) | pieces(enemy, PieceType::Queen);
  while (rooks) {
    int s = pop_lsb(rooks);
    if (rook_attacks(s, occ) & bit(king_sq)) return true;
  }

  Bitboard pawns = pieces(enemy, PieceType::Pawn);
  uint64_t attacks = 0;
  if (c == Color::White) {
    if (king_sq % 8 > 0) attacks |= bit(king_sq - 9);
    if (king_sq % 8 < 7) attacks |= bit(king_sq - 7);
  } else {
    if (king_sq % 8 > 0) attacks |= bit(king_sq + 7);
    if (king_sq % 8 < 7) attacks |= bit(king_sq + 9);
  }
  if (pawns & attacks) return true;

  Bitboard kings_e = pieces(enemy, PieceType::King);
  while (kings_e) {
    int s = pop_lsb(kings_e);
    if (king_attacks(s) & bit(king_sq)) return true;
  }
  return false;
}

bool Board::is_checkmate() const {
  if (!in_check(stm_)) return false;
  std::vector<Move> moves;
  const_cast<Board*>(this)->generate_legal_moves(moves);
  return moves.empty();
}

bool Board::is_stalemate() const {
  if (in_check(stm_)) return false;
  std::vector<Move> moves;
  const_cast<Board*>(this)->generate_legal_moves(moves);
  return moves.empty();
}

bool Board::is_repetition() const {
  int count = 0;
  for (const auto& u : undo_stack_) {
    if (u.hash == hash_) ++count;
  }
  return count >= 2;
}

bool Board::has_insufficient_material() const {
  int w = 0, b = 0;
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = at(sq);
    if (p == Piece::None) continue;
    if (color_of(p) == Color::White) ++w;
    else ++b;
  }
  return w <= 1 && b <= 1;
}

bool Board::is_draw() const {
  return halfmove_clock_ >= 100 || is_repetition() || has_insufficient_material() ||
         is_stalemate();
}

GameResult Board::result() const {
  if (is_checkmate()) return stm_ == Color::White ? GameResult::BlackWin : GameResult::WhiteWin;
  if (is_draw() || is_stalemate()) return GameResult::Draw;
  return GameResult::Ongoing;
}

std::string Board::pretty() const { return fen(); }

uint64_t perft(Board& board, int depth) {
  if (depth == 0) return 1;
  std::vector<Move> moves;
  board.generate_legal_moves(moves);
  uint64_t nodes = 0;
  for (const auto& m : moves) {
    board.make_move(m);
    nodes += perft(board, depth - 1);
    board.unmake_move(m);
  }
  return nodes;
}

}  // namespace az
