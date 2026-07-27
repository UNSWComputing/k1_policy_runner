#pragma once

#include "policy/policy.hpp"

namespace policy_runner {

// Hard-coded demo: left + right elbow (indices 5, 9) step between two angles.
// Sparse output — only elbows are written; other joints stay weight=0.
class StepArmPolicy : public Policy {
 public:
  explicit StepArmPolicy(float control_dt = 0.02f, float period_s = 2.f);

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
  float period_s_;
  float time_s_ = 0.f;
};

}  // namespace policy_runner
