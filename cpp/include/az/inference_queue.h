#pragma once

#include "az/types.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace az {

struct InferenceRequest {
  int id = 0;
  std::vector<float> state;
  std::vector<float> policy_out;
  float value_out = 0.0f;
  bool fulfilled = false;
};

struct DrainedRequest {
  int id = 0;
  std::vector<float> state;
};

class InferenceQueue {
 public:
  InferenceQueue() = default;

  // Called from C++ MCTS (blocks until fulfilled)
  PolicyValue evaluate(const std::vector<float>& state);

  // Called from Python inference server — copies state under lock (safe for parallel MCTS)
  std::vector<DrainedRequest> drain(int max_batch, int max_wait_us);
  void fulfill(const std::vector<int>& ids,
               const std::vector<std::vector<float>>& policies,
               const std::vector<float>& values);

  int pending() const;

  /// Fulfill all pending requests with zero policy/value so blocked MCTS threads
  /// can unblock during shutdown. Safe to call from any thread.
  void shutdown();

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_producer_;
  std::condition_variable cv_consumer_;
  std::deque<InferenceRequest> queue_;
  uint64_t next_id_ = 0;
};

}  // namespace az
