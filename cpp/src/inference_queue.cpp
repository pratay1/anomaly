#include "az/inference_queue.h"

#include <chrono>

namespace az {

PolicyValue InferenceQueue::evaluate(const std::vector<float>& state) {
  int my_id;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    my_id = next_id_++;
    InferenceRequest req;
    req.id = my_id;
    req.state = state;
    req.policy_out.resize(POLICY_SIZE, 0.0f);
    queue_.push_back(std::move(req));
    cv_consumer_.notify_one();
  }

  std::unique_lock<std::mutex> lock(mutex_);
  cv_producer_.wait(lock, [&] {
    for (const auto& r : queue_) {
      if (r.id == my_id && r.fulfilled) return true;
    }
    return false;
  });

  PolicyValue pv;
  pv.policy.resize(POLICY_SIZE, 0.0f);
  for (auto it = queue_.begin(); it != queue_.end(); ++it) {
    if (it->id == my_id) {
      if (static_cast<int>(it->policy_out.size()) == POLICY_SIZE) {
        pv.policy = it->policy_out;
      }
      pv.value = it->value_out;
      queue_.erase(it);
      break;
    }
  }
  return pv;
}

std::vector<DrainedRequest> InferenceQueue::drain(int max_batch, int max_wait_us) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::microseconds(max_wait_us);

  cv_consumer_.wait_until(lock, deadline, [&] {
    for (const auto& r : queue_) {
      if (!r.fulfilled) return true;
    }
    return false;
  });

  std::vector<DrainedRequest> out;
  for (const auto& r : queue_) {
    if (!r.fulfilled) {
      DrainedRequest d;
      d.id = r.id;
      d.state = r.state;
      out.push_back(std::move(d));
      if (static_cast<int>(out.size()) >= max_batch) break;
    }
  }
  return out;
}

void InferenceQueue::fulfill(const std::vector<int>& ids,
                             const std::vector<std::vector<float>>& policies,
                             const std::vector<float>& values) {
  std::unique_lock<std::mutex> lock(mutex_);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i >= policies.size() || i >= values.size()) break;
    for (auto& r : queue_) {
      if (r.id == ids[i]) {
        r.policy_out = policies[i];
        if (static_cast<int>(r.policy_out.size()) != POLICY_SIZE) {
          r.policy_out.assign(POLICY_SIZE, 0.0f);
        }
        r.value_out = values[i];
        r.fulfilled = true;
        break;
      }
    }
  }
  cv_producer_.notify_all();
}

int InferenceQueue::pending() const {
  std::lock_guard<std::mutex> lock(mutex_);
  int n = 0;
  for (const auto& r : queue_) {
    if (!r.fulfilled) ++n;
  }
  return n;
}

}  // namespace az
