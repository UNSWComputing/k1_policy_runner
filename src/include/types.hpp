#pragma once

#include <cstddef>
#include <vector>

namespace policy_runner {

// 22-DoF joint count (no waist), matching K1 / joint_index layout.
inline constexpr std::size_t kB1JointCount = 22;

// Full-body robot state as read from LowState.
// Always sized to the robot (all joints), independent of any policy.
struct RobotState {
  std::vector<float> q;   // joint positions, size == kB1JointCount
  std::vector<float> dq;  // joint velocities, size == kB1JointCount
};

// Flat observation vector consumed by a policy.
// Layout is policy-defined. Typical pattern:
//   [all_joint_q..., optional_extras...]
// e.g. a walk policy may append 3 command dims after the joints.
struct Observation {
  std::vector<float> data;
};

// One controlled joint. Policies that only actuate a subset of the body
// (arms-only, half-body, etc.) emit only the joints they own.
struct JointCommand {
  int index = -1;  // matches booster::robot::b1::JointIndex / motor_cmd slot
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

}  // namespace policy_runner
