#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace policy_runner {

// 22-DoF joint count (no waist), matching K1 / joint_index layout.
inline constexpr std::size_t kB1JointCount = 22;
inline constexpr std::size_t kImuDim = 9;  // rpy(3) + gyro(3) + acc(3)
inline constexpr std::size_t kProjectedGravityDim = 3;
inline constexpr std::size_t kBaseObsDim =
    kB1JointCount + kImuDim + kProjectedGravityDim;

struct ImuState {
  std::array<float, 3> rpy{{0.f, 0.f, 0.f}};
  std::array<float, 3> gyro{{0.f, 0.f, 0.f}};
  std::array<float, 3> acc{{0.f, 0.f, 0.f}};
};

// Gravity [0, 0, -1] expressed in the base frame from roll/pitch (yaw-invariant).
inline std::array<float, 3> ComputeProjectedGravity(
    const std::array<float, 3>& rpy) {
  const float roll = rpy[0];
  const float pitch = rpy[1];
  return {
      std::sin(pitch),
      -std::sin(roll) * std::cos(pitch),
      -std::cos(roll) * std::cos(pitch),
  };
}

// Full robot state: joints + IMU + projected gravity.
struct RobotState {
  std::vector<float> q;   // joint positions, size == kB1JointCount
  std::vector<float> dq;  // joint velocities, size == kB1JointCount
  ImuState imu;
  std::array<float, 3> projected_gravity{{0.f, 0.f, -1.f}};
};

// Flat observation vector consumed by a policy.
// Layout is policy-defined. Typical pattern:
//   [all_joint_q..., imu..., projected_gravity..., optional_extras...]
struct Observation {
  std::vector<float> data;
};

// One controlled joint. Policies that only actuate a subset of the body
// (arms-only, half-body, etc.) emit only the joints they own.
struct JointCommand {
  int index = -1;  // matches JointIndex / motor_cmd slot
  float q = 0.f;
  float dq = 0.f;
  float tau = 0.f;
  float kp = 0.f;
  float kd = 0.f;
  float weight = 1.f;
};

// Sparse action: only joints this policy controls. Never assume full-body.
struct Action {
  std::vector<JointCommand> joint_cmds;
};

// Default layout: [joint_q..., imu..., projected_gravity..., optional command...].
inline Observation PackObservation(const RobotState& state,
                                   const std::vector<float>& command = {}) {
  if (state.q.size() != kB1JointCount) {
    throw std::runtime_error("PackObservation: RobotState.q size mismatch");
  }
  Observation obs;
  obs.data.reserve(kBaseObsDim + command.size());
  obs.data.insert(obs.data.end(), state.q.begin(), state.q.end());
  obs.data.insert(obs.data.end(), state.imu.rpy.begin(), state.imu.rpy.end());
  obs.data.insert(obs.data.end(), state.imu.gyro.begin(), state.imu.gyro.end());
  obs.data.insert(obs.data.end(), state.imu.acc.begin(), state.imu.acc.end());
  obs.data.insert(obs.data.end(), state.projected_gravity.begin(),
                  state.projected_gravity.end());
  obs.data.insert(obs.data.end(), command.begin(), command.end());
  return obs;
}

}  // namespace policy_runner
