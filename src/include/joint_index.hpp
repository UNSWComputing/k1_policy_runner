#pragma once

// 22-DoF joint indices (no waist), matching python joint_index.py / K1 layout.
namespace policy_runner {
namespace joint {

enum Index : int {
  kHeadYaw = 0,
  kHeadPitch = 1,
  kLeftShoulderPitch = 2,
  kLeftShoulderRoll = 3,
  kLeftElbowPitch = 4,
  kLeftElbowYaw = 5,
  kRightShoulderPitch = 6,
  kRightShoulderRoll = 7,
  kRightElbowPitch = 8,
  kRightElbowYaw = 9,
  kLeftHipPitch = 10,
  kLeftHipRoll = 11,
  kLeftHipYaw = 12,
  kLeftKneePitch = 13,
  kLeftAnklePitch = 14,
  kLeftAnkleRoll = 15,
  kRightHipPitch = 16,
  kRightHipRoll = 17,
  kRightHipYaw = 18,
  kRightKneePitch = 19,
  kRightAnklePitch = 20,
  kRightAnkleRoll = 21,
};

}  // namespace joint
}  // namespace policy_runner
