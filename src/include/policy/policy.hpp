#pragma once

#include "types.hpp"

#include <string>
#include <vector>

namespace policy_runner {

// Abstract policy interface.
//
// Input shape differs per model (joints + optional extras).
// Output is always sparse joint commands — not necessarily all joints.
class Policy {
 public:
  virtual ~Policy() = default;

  virtual std::string name() const = 0;

  // Expected observation length after BuildObservation().
  virtual std::size_t input_dim() const = 0;

  // Joint indices this policy will write. Empty means "undeclared".
  virtual std::vector<int> controlled_joints() const = 0;

  // Pack RobotState (+ optional command extras) into this policy's observation.
  // `command` is model-specific (e.g. 3-D walk cmd). Arm demos ignore it.
  virtual Observation BuildObservation(
      const RobotState& state,
      const std::vector<float>& command) const = 0;

  virtual Action Infer(const Observation& obs) = 0;

  virtual void Reset() {}
};

}  // namespace policy_runner
