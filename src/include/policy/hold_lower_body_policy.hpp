#pragma once

#include "policy/policy.hpp"

#include <vector>

namespace policy_runner {

// Holds all lower-body joints at the pose latched when control starts.
// Sparse output — only legs (indices 10-21); upper body stays weight=0.
class HoldLowerBodyPolicy : public Policy {
 public:
  explicit HoldLowerBodyPolicy(float control_dt = 0.02f);

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
  bool hold_captured_ = false;
  std::vector<float> hold_q_;
};

}  // namespace policy_runner
