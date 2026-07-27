#pragma once

#include "policy/policy.hpp"

namespace policy_runner {

// Hard-coded demo: left + right knee pitch follow a sine wave.
// Sparse output — only knee pitches are written; other joints stay weight=0.
class SineKneePolicy : public Policy {
 public:
  explicit SineKneePolicy(float control_dt = 0.02f);

  std::string name() const override;
  std::size_t input_dim() const override;
  std::vector<int> controlled_joints() const override;

  Observation BuildObservation(
      const RobotState& state,
      const std::vector<float>& command) const override;

  Action Infer(const Observation& obs) override;
  void Reset() override;

 private:
  float control_dt_;
  float time_s_ = 0.f;
};

}  // namespace policy_runner
