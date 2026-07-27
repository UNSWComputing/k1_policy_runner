#pragma once

#include "types.hpp"

namespace policy_runner {

// Default PD gains for hard-coded policies.
// From README K1_CFG joint_stiffness / joint_damping (22-DoF, no waist).
inline constexpr float kDefaultKp[kB1JointCount] = {
    4.f, 4.f,                    // head
    4.f, 4.f, 4.f, 4.f,          // left arm
    4.f, 4.f, 4.f, 4.f,          // right arm
    80.f, 80.f, 80.f, 80.f, 30.f, 30.f,  // left leg
    80.f, 80.f, 80.f, 80.f, 30.f, 30.f,  // right leg
};

inline constexpr float kDefaultKd[kB1JointCount] = {
    1.f, 1.f,
    1.f, 1.f, 1.f, 1.f,
    1.f, 1.f, 1.f, 1.f,
    2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
    2.f, 2.f, 2.f, 2.f, 2.f, 2.f,
};

inline JointCommand MakeJointCmd(int index, float q) {
  JointCommand cmd;
  cmd.index = index;
  cmd.q = q;
  cmd.dq = 0.f;
  cmd.tau = 0.f;
  cmd.kp = kDefaultKp[index];
  cmd.kd = kDefaultKd[index];
  cmd.weight = 1.f;
  return cmd;
}

}  // namespace policy_runner
