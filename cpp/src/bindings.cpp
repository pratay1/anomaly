#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "az/board.h"
#include "az/encoding.h"
#include "az/inference_queue.h"
#include "az/mcts.h"
#include "az/selfplay.h"
#include "az/types.h"

namespace py = pybind11;

PYBIND11_MODULE(_az_core, m) {
  m.doc() = "AlphaZero chess C++ core";

  py::enum_<az::Color>(m, "Color")
      .value("White", az::Color::White)
      .value("Black", az::Color::Black);

  py::enum_<az::PieceType>(m, "PieceType")
      .value("None", az::PieceType::None)
      .value("Pawn", az::PieceType::Pawn)
      .value("Knight", az::PieceType::Knight)
      .value("Bishop", az::PieceType::Bishop)
      .value("Rook", az::PieceType::Rook)
      .value("Queen", az::PieceType::Queen)
      .value("King", az::PieceType::King);

  py::enum_<az::Piece>(m, "Piece")
      .value("None", az::Piece::None)
      .value("WP", az::Piece::WP)
      .value("WN", az::Piece::WN)
      .value("WB", az::Piece::WB)
      .value("WR", az::Piece::WR)
      .value("WQ", az::Piece::WQ)
      .value("WK", az::Piece::WK)
      .value("BP", az::Piece::BP)
      .value("BN", az::Piece::BN)
      .value("BB", az::Piece::BB)
      .value("BR", az::Piece::BR)
      .value("BQ", az::Piece::BQ)
      .value("BK", az::Piece::BK);

  py::enum_<az::GameResult>(m, "GameResult")
      .value("Ongoing", az::GameResult::Ongoing)
      .value("WhiteWin", az::GameResult::WhiteWin)
      .value("BlackWin", az::GameResult::BlackWin)
      .value("Draw", az::GameResult::Draw);

  py::class_<az::Move>(m, "Move")
      .def_readwrite("from_sq", &az::Move::from)
      .def_readwrite("to_sq", &az::Move::to)
      .def_readwrite("piece", &az::Move::piece)
      .def_readwrite("captured", &az::Move::captured)
      .def_readwrite("promotion", &az::Move::promotion)
      .def_readwrite("flag", &az::Move::flag);

  py::class_<az::Board>(m, "Board")
      .def(py::init<>())
      .def_static("from_fen", &az::Board::from_fen)
      .def("fen", &az::Board::fen)
      .def("at", [](const az::Board& b, int sq) { return b.at(sq); })
      .def("side_to_move", &az::Board::side_to_move)
      .def("make_move", &az::Board::make_move, py::call_guard<py::gil_scoped_release>())
      .def("unmake_move", &az::Board::unmake_move, py::call_guard<py::gil_scoped_release>())
      .def("generate_legal_moves", [](const az::Board& b) {
        std::vector<az::Move> moves;
        b.generate_legal_moves(moves);
        return moves;
      })
      .def("result", &az::Board::result)
      .def("in_check", &az::Board::in_check)
      .def("is_legal", &az::Board::is_legal);

  m.def("encode", &az::encode, py::call_guard<py::gil_scoped_release>());
  m.def("move_to_index", &az::move_to_index);
  m.def("index_to_move", &az::index_to_move);
  m.def("legal_move_indices", &az::legal_move_indices);
  m.def("mask_policy", &az::mask_policy);

  m.def("perft", [](const std::string& fen, int depth) {
    az::Board b = az::Board::from_fen(fen);
    return az::perft(b, depth);
  });

  py::class_<az::DrainedRequest>(m, "DrainedRequest")
      .def_readonly("id", &az::DrainedRequest::id)
      .def_readonly("state", &az::DrainedRequest::state);

  py::class_<az::InferenceRequest>(m, "InferenceRequest")
      .def_readonly("id", &az::InferenceRequest::id)
      .def_readonly("state", &az::InferenceRequest::state);

  py::class_<az::InferenceQueue>(m, "InferenceQueue")
      .def(py::init<>())
      .def("drain", &az::InferenceQueue::drain, py::arg("max_batch"), py::arg("max_wait_us"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "fulfill",
          [](az::InferenceQueue& q, const std::vector<int>& ids,
             const std::vector<std::vector<float>>& policies, const std::vector<float>& values) {
            q.fulfill(ids, policies, values);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("pending", &az::InferenceQueue::pending)
      .def("shutdown", &az::InferenceQueue::shutdown,
           py::call_guard<py::gil_scoped_release>());

  py::class_<az::MCTSConfig>(m, "MCTSConfig")
      .def(py::init<>())
      .def_readwrite("num_simulations", &az::MCTSConfig::num_simulations)
      .def_readwrite("c_puct_base", &az::MCTSConfig::c_puct_base)
      .def_readwrite("c_puct_init", &az::MCTSConfig::c_puct_init)
      .def_readwrite("dirichlet_alpha", &az::MCTSConfig::dirichlet_alpha)
      .def_readwrite("dirichlet_eps", &az::MCTSConfig::dirichlet_eps)
      .def_readwrite("add_root_noise", &az::MCTSConfig::add_root_noise);

  py::class_<az::RootVisit>(m, "RootVisit")
      .def_readonly("move_index", &az::RootVisit::move_index)
      .def_readonly("N", &az::RootVisit::N)
      .def_readonly("Q", &az::RootVisit::Q)
      .def_readonly("P", &az::RootVisit::P);

  py::class_<az::MCTS>(m, "MCTS")
      .def(py::init<az::InferenceQueue*, const az::MCTSConfig&>())
      .def("run", &az::MCTS::run, py::arg("board"), py::arg("temperature") = 1.0f,
           py::call_guard<py::gil_scoped_release>())
      .def("advance_root", &az::MCTS::advance_root, py::arg("move_index"),
           py::call_guard<py::gil_scoped_release>())
      .def("reset_tree", &az::MCTS::reset_tree, py::call_guard<py::gil_scoped_release>())
      .def("root_visits", &az::MCTS::root_visits);

  py::class_<az::TrainingExample>(m, "TrainingExample")
      .def_readonly("state", &az::TrainingExample::state)
      .def_readonly("policy", &az::TrainingExample::policy)
      .def_readonly("value", &az::TrainingExample::value);

  py::class_<az::SelfPlayConfig>(m, "SelfPlayConfig")
      .def(py::init<>())
      .def_readwrite("mcts", &az::SelfPlayConfig::mcts)
      .def_readwrite("temperature_moves", &az::SelfPlayConfig::temperature_moves)
      .def_readwrite("max_game_length", &az::SelfPlayConfig::max_game_length);

  py::class_<az::SelfPlayRunner>(m, "SelfPlayRunner")
      .def(py::init<az::InferenceQueue*, const az::SelfPlayConfig&>())
      .def("play_game", &az::SelfPlayRunner::play_game, py::call_guard<py::gil_scoped_release>())
      .def("run_games", [](az::SelfPlayRunner& r, int n) {
        std::vector<std::vector<az::TrainingExample>> out;
        r.run_games(n, out);
        return out;
      }, py::arg("n"), py::call_guard<py::gil_scoped_release>());

  m.attr("POLICY_SIZE") = az::POLICY_SIZE;
  m.attr("ENCODING_CHANNELS") = az::ENCODING_CHANNELS;
}
