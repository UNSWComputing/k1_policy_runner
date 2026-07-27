#pragma once

#include "joint_gains.hpp"
#include "joint_index.hpp"

namespace policy_runner {
namespace arm_demo {

// Elbow joints per robot joint table (README): indices 5 and 9.
// (SDK names these ElbowYaw; README calls them Elbow.)
inline constexpr int kLeftElbow = joint::kLeftElbowYaw;    // 5
inline constexpr int kRightElbow = joint::kRightElbowYaw;  // 9

inline JointCommand MakeElbowCmd(int index, float q) {
  return MakeJointCmd(index, q);
}

}  // namespace arm_demo
}  // namespace policy_runner
