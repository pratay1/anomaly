#include "az/encoding.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace az {

namespace {

constexpr int DIRS[8][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
constexpr int KNIGHT_D[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};

int direction_index(int df, int dr) {
  for (int d = 0; d < 8; ++d) {
    if (DIRS[d][0] == df && DIRS[d][1] == dr) return d;
  }
  return -1;
}

int knight_direction_index(int df, int dr) {
  for (int d = 0; d < 8; ++d) {
    if (KNIGHT_D[d][0] == df && KNIGHT_D[d][1] == dr) return d;
  }
  return -1;
}

void set_plane(std::vector<float>& out, int channel, int square, float v = 1.0f) {
  out[channel * 64 + square] = v;
}

// Piece-Square Tables (white's perspective, sq = rank*8 + file, rank 0 = a1)
// Values in centipawns / 100, range roughly [-0.5, 0.5]
constexpr float PST_PAWN[64] = {
  0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
  0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f, 0.50f,
  0.10f, 0.10f, 0.20f, 0.30f, 0.30f, 0.20f, 0.10f, 0.10f,
  0.05f, 0.05f, 0.10f, 0.25f, 0.25f, 0.10f, 0.05f, 0.05f,
  0.00f, 0.00f, 0.00f, 0.20f, 0.20f, 0.00f, 0.00f, 0.00f,
  0.05f,-0.05f,-0.10f, 0.00f, 0.00f,-0.10f,-0.05f, 0.05f,
  0.05f, 0.10f, 0.10f,-0.20f,-0.20f, 0.10f, 0.10f, 0.05f,
  0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
};

constexpr float PST_KNIGHT[64] = {
 -0.50f,-0.40f,-0.30f,-0.30f,-0.30f,-0.30f,-0.40f,-0.50f,
 -0.40f,-0.20f, 0.00f, 0.05f, 0.05f, 0.00f,-0.20f,-0.40f,
 -0.30f, 0.05f, 0.10f, 0.15f, 0.15f, 0.10f, 0.05f,-0.30f,
 -0.30f, 0.00f, 0.15f, 0.20f, 0.20f, 0.15f, 0.00f,-0.30f,
 -0.30f, 0.05f, 0.15f, 0.20f, 0.20f, 0.15f, 0.05f,-0.30f,
 -0.30f, 0.00f, 0.10f, 0.15f, 0.15f, 0.10f, 0.00f,-0.30f,
 -0.40f,-0.20f, 0.00f, 0.00f, 0.00f, 0.00f,-0.20f,-0.40f,
 -0.50f,-0.40f,-0.30f,-0.30f,-0.30f,-0.30f,-0.40f,-0.50f,
};

constexpr float PST_BISHOP[64] {
 -0.20f,-0.10f,-0.10f,-0.10f,-0.10f,-0.10f,-0.10f,-0.20f,
 -0.10f, 0.05f, 0.00f, 0.00f, 0.00f, 0.00f, 0.05f,-0.10f,
 -0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f,-0.10f,
 -0.10f, 0.00f, 0.10f, 0.10f, 0.10f, 0.10f, 0.00f,-0.10f,
 -0.10f, 0.05f, 0.05f, 0.10f, 0.10f, 0.05f, 0.05f,-0.10f,
 -0.10f, 0.00f, 0.05f, 0.10f, 0.10f, 0.05f, 0.00f,-0.10f,
 -0.10f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,-0.10f,
 -0.20f,-0.10f,-0.10f,-0.10f,-0.10f,-0.10f,-0.10f,-0.20f,
};

constexpr float PST_ROOK[64] {
  0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
  0.05f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.05f,
 -0.05f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,-0.05f,
 -0.05f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,-0.05f,
 -0.05f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,-0.05f,
 -0.05f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,-0.05f,
  0.05f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.05f,
  0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f,
};

constexpr float PST_QUEEN[64] {
 -0.20f,-0.10f,-0.10f,-0.05f,-0.05f,-0.10f,-0.10f,-0.20f,
 -0.10f, 0.00f, 0.05f, 0.00f, 0.00f, 0.00f, 0.00f,-0.10f,
 -0.10f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.00f,-0.10f,
 -0.05f, 0.00f, 0.05f, 0.05f, 0.05f, 0.05f, 0.00f,-0.05f,
  0.00f, 0.00f, 0.05f, 0.05f, 0.05f, 0.05f, 0.00f,-0.05f,
 -0.10f, 0.05f, 0.05f, 0.05f, 0.05f, 0.05f, 0.00f,-0.10f,
 -0.10f, 0.00f, 0.05f, 0.00f, 0.00f, 0.00f, 0.00f,-0.10f,
 -0.20f,-0.10f,-0.10f,-0.05f,-0.05f,-0.10f,-0.10f,-0.20f,
};

constexpr float PST_KING[64] {
  0.20f, 0.30f, 0.10f, 0.00f, 0.00f, 0.10f, 0.30f, 0.20f,
  0.20f, 0.20f, 0.00f, 0.00f, 0.00f, 0.00f, 0.20f, 0.20f,
 -0.10f,-0.20f,-0.20f,-0.20f,-0.20f,-0.20f,-0.20f,-0.10f,
 -0.20f,-0.30f,-0.30f,-0.40f,-0.40f,-0.30f,-0.30f,-0.20f,
 -0.30f,-0.40f,-0.40f,-0.50f,-0.50f,-0.40f,-0.40f,-0.30f,
 -0.30f,-0.40f,-0.40f,-0.50f,-0.50f,-0.40f,-0.40f,-0.30f,
 -0.30f,-0.40f,-0.40f,-0.50f,-0.50f,-0.40f,-0.40f,-0.30f,
 -0.30f,-0.40f,-0.40f,-0.50f,-0.50f,-0.40f,-0.40f,-0.30f,
};

inline int flip_rank(int sq) { return sq ^ 56; }

constexpr const float* PST_TABLES[6] = {
  PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING
};

}  // namespace

std::vector<float> encode(const Board& board) {
  std::vector<float> out(ENCODING_CHANNELS * 64, 0.0f);

  const auto& hist = board.history();
  int frames = static_cast<int>(hist.size());
  for (int t = 0; t < 8; ++t) {
    int idx = frames - 1 - t;
    const std::array<Piece, 64>* snap = nullptr;
    if (idx >= 0 && idx < frames) {
      snap = &hist[static_cast<size_t>(idx)];
    } else if (!hist.empty()) {
      snap = &hist.back();
    }
    if (!snap) continue;
    int base_ch = t * 12;
    for (int sq = 0; sq < 64; ++sq) {
      Piece p = (*snap)[sq];
      if (p == Piece::None) continue;
      int plane = base_ch + static_cast<int>(p) - 1;
      if (plane < 96) set_plane(out, plane, sq);
    }
  }

  int ch = 96;
  if (board.side_to_move() == Color::Black) {
    for (int sq = 0; sq < 64; ++sq) set_plane(out, ch, sq);
  }
  ++ch;

  int rep = 0;
  uint64_t h = board.hash();
  // repetition approximated via history size
  (void)h;
  if (board.is_repetition()) rep = 1;
  for (int sq = 0; sq < 64; ++sq) {
    if (rep >= 1) set_plane(out, ch, sq);
    if (rep >= 2) set_plane(out, ch + 1, sq);
  }
  ch += 2;

  uint8_t cr = board.castling_rights();
  if (cr & 1) for (int sq = 0; sq < 64; ++sq) set_plane(out, ch, sq);
  if (cr & 2) for (int sq = 0; sq < 64; ++sq) set_plane(out, ch + 1, sq);
  if (cr & 4) for (int sq = 0; sq < 64; ++sq) set_plane(out, ch + 2, sq);
  if (cr & 8) for (int sq = 0; sq < 64; ++sq) set_plane(out, ch + 3, sq);
  ch += 4;

  float fifty = static_cast<float>(board.halfmove_clock()) / 100.0f;
  for (int sq = 0; sq < 64; ++sq) set_plane(out, ch, sq, fifty);
  ++ch;

  int move_num = std::min(board.fullmove_number(), 255);
  for (int bit = 0; bit < 8; ++bit) {
    if ((move_num >> bit) & 1) {
      for (int sq = 0; sq < 64; ++sq) set_plane(out, ch + bit, sq);
    }
  }
  ch += 8;

  // PST value planes (channels 112-117) — one per piece type
  // Fills the former padding slots with positional priors for rapid learning.
  for (int sq = 0; sq < 64; ++sq) {
    Piece p = board.at(sq);
    if (p == Piece::None) continue;
    int pst_idx = static_cast<int>(type_of(p)) - 1;
    if (pst_idx < 0 || pst_idx > 5) continue;
    int lookup_sq = (color_of(p) == Color::White) ? sq : flip_rank(sq);
    set_plane(out, ch + pst_idx, sq, PST_TABLES[pst_idx][lookup_sq]);
  }
  ch += 6;

  // Pad remaining channels to 119 with zeros (already zero)
  (void)ch;
  return out;
}

int move_to_index(const Board& board, const Move& move) {
  int from = move.from;
  int to = move.to;
  int df = file_of(to) - file_of(from);
  int dr = rank_of(to) - rank_of(from);

  if (type_of(move.piece) == PieceType::Pawn && move.promotion != PieceType::None &&
      move.promotion != PieceType::Queen) {
    int promo_dir = board.side_to_move() == Color::White ? 1 : -1;
    int plane_base = 64;
    if (move.promotion == PieceType::Knight) plane_base += 0;
    else if (move.promotion == PieceType::Bishop) plane_base += 3;
    else if (move.promotion == PieceType::Rook) plane_base += 6;
    int offset = 0;
    if (df == -1 && dr == promo_dir) offset = 1;
    else if (df == 0 && dr == promo_dir) offset = 0;
    else if (df == 1 && dr == promo_dir) offset = 2;
    return from * NUM_MOVE_PLANES + plane_base + offset;
  }

  int kd = knight_direction_index(df, dr);
  if (kd >= 0) return from * NUM_MOVE_PLANES + 56 + kd;

  int dir = direction_index(df == 0 ? 0 : df / std::abs(df), dr == 0 ? 0 : dr / std::abs(dr));
  if (dir < 0) return -1;
  int dist = std::max(std::abs(df), std::abs(dr));
  if (dist < 1 || dist > 7) return -1;
  return from * NUM_MOVE_PLANES + dir * 7 + (dist - 1);
}

Move index_to_move(const Board& board, int index) {
  int from = index / NUM_MOVE_PLANES;
  int move_type = index % NUM_MOVE_PLANES;
  int f0 = file_of(from);
  int r0 = rank_of(from);

  if (move_type >= 64) {
    int promo_base = move_type - 64;
    PieceType promo = PieceType::Knight;
    int df = 0, dr = board.side_to_move() == Color::White ? 1 : -1;
    if (promo_base < 3) {
      promo = PieceType::Knight;
      df = (promo_base == 1) ? -1 : (promo_base == 2 ? 1 : 0);
    } else if (promo_base < 6) {
      promo = PieceType::Bishop;
      df = (promo_base == 4) ? -1 : (promo_base == 5 ? 1 : 0);
    } else {
      promo = PieceType::Rook;
      df = (promo_base == 7) ? -1 : (promo_base == 8 ? 1 : 0);
    }
    Move m;
    m.from = from;
    m.to = sq(f0 + df, r0 + dr);
    m.piece = board.at(from);
    m.promotion = promo;
    m.flag = MoveFlag::Promotion;
    return m;
  }

  if (move_type >= 56) {
    int kd = move_type - 56;
    Move m;
    m.from = from;
    m.to = sq(f0 + KNIGHT_D[kd][0], r0 + KNIGHT_D[kd][1]);
    m.piece = board.at(from);
    return m;
  }

  int dir = move_type / 7;
  int dist = (move_type % 7) + 1;
  Move m;
  m.from = from;
  m.to = sq(f0 + DIRS[dir][0] * dist, r0 + DIRS[dir][1] * dist);
  m.piece = board.at(from);
  if (type_of(m.piece) == PieceType::Pawn) {
    int promo_rank = board.side_to_move() == Color::White ? 7 : 0;
    if (rank_of(m.to) == promo_rank) {
      m.promotion = PieceType::Queen;
      m.flag = MoveFlag::Promotion;
      Piece cap = board.at(m.to);
      if (cap != Piece::None) {
        m.captured = cap;
        m.flag = MoveFlag::Promotion;
      }
    }
  }
  return m;
}

std::vector<int> legal_move_indices(const Board& board) {
  std::vector<Move> moves;
  board.generate_legal_moves(moves);
  std::vector<int> idx;
  for (const auto& m : moves) {
    int i = move_to_index(board, m);
    if (i >= 0) idx.push_back(i);
  }
  return idx;
}

void mask_policy(std::vector<float>& policy, const Board& board) {
  if (static_cast<int>(policy.size()) < POLICY_SIZE) {
    policy.resize(POLICY_SIZE, 0.0f);
  }
  std::vector<float> masked(POLICY_SIZE, 0.0f);
  auto legal = legal_move_indices(board);
  float sum = 0.0f;
  for (int i : legal) {
    if (i >= 0 && i < POLICY_SIZE) {
      masked[i] = policy[i];
      sum += masked[i];
    }
  }
  if (sum > 0) {
    for (int i : legal) masked[i] /= sum;
  } else if (!legal.empty()) {
    float u = 1.0f / legal.size();
    for (int i : legal) masked[i] = u;
  }
  policy = std::move(masked);
}

}  // namespace az
