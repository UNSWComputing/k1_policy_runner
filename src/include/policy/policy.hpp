#pragma once

#include "types.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace policy_runner {

// Abstract policy interface.
//
// Dimensionality:
//   observation_dim() — size of one frame from BuildObservation()
//   input_dim()       — size consumed by Infer / the model (may include history)
//   history_len()     — frames stacked inside the policy (default 1)
//
// History buffers live inside the policy. The runner only passes one
// Observation per step.
class Policy {
 public:
  virtual ~Policy() = default;

  virtual std::string name() const = 0;

  // Model / Infer input size (may be observation_dim * history_len).
  virtual std::size_t input_dim() const = 0;

  // Single-frame observation size. Default: same as input_dim (no history).
  virtual std::size_t observation_dim() const { return input_dim(); }

  // Number of frames stacked inside the policy. Default: 1.
  virtual std::size_t history_len() const { return 1; }

  // Joint indices this policy will write. Empty means "undeclared".
  virtual std::vector<int> controlled_joints() const = 0;

  // Pack RobotState (+ optional command extras) into this policy's observation.
  virtual Observation BuildObservation(
      const RobotState& state,
      const std::vector<float>& command) const = 0;

  virtual Action Infer(const Observation& obs) = 0;

  virtual void Reset() {}

  // Sanity-check one frame from BuildObservation (not the stacked model input).
  void AssertFrameObservation(const Observation& obs) const {
    const std::size_t got = obs.data.size();
    const std::size_t expected = observation_dim();
    if (got != expected) {
      throw std::runtime_error(
          name() + ": frame observation dim " + std::to_string(got) +
          " != observation_dim " + std::to_string(expected) +
          " (input_dim=" + std::to_string(input_dim()) +
          ", history_len=" + std::to_string(history_len()) + ")");
    }
  }
};

}  // namespace policy_runner
