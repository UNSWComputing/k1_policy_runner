#include "policy/hold_lower_body_policy.hpp"

#include "joint_gains.hpp"
#include "joint_index.hpp"

#include <stdexcept>

namespace policy_runner {
namespace {

// Lower body: left leg 10-15, right leg 16-21.
constexpr int kLowerBodyBegin = joint::kLeftHipPitch;
constexpr int kLowerBodyEnd = joint::kRightAnkleRoll;  // inclusive

}  // namespace

HoldLowerBodyPolicy::HoldLowerBodyPolicy(float control_dt)
    : control_dt_(control_dt) {}

std::string HoldLowerBodyPolicy::name() const { return "hold_lower"; }

std::size_t HoldLowerBodyPolicy::input_dim() const {
  return kBaseObsDim;
}

std::vector<int> HoldLowerBodyPolicy::controlled_joints() const {
  std::vector<int> joints;
  joints.reserve(static_cast<std::size_t>(kLowerBodyEnd - kLowerBodyBegin + 1));
  for (int i = kLowerBodyBegin; i <= kLowerBodyEnd; ++i) {
    joints.push_back(i);
  }
  return joints;
}

Observation HoldLowerBodyPolicy::BuildObservation(
    const RobotState& state,
    const std::vector<float>& command) const {
  return PackObservation(state, command);
}

Action HoldLowerBodyPolicy::Infer(const Observation& obs) {
  AssertFrameObservation(obs);

  // Latch joint positions only (not IMU / command extras).
  if (!hold_captured_) {
    hold_q_.assign(obs.data.begin(), obs.data.begin() + kB1JointCount);
    hold_captured_ = true;
  }

  Action action;
  action.joint_cmds.reserve(
      static_cast<std::size_t>(kLowerBodyEnd - kLowerBodyBegin + 1));
  for (int i = kLowerBodyBegin; i <= kLowerBodyEnd; ++i) {
    action.joint_cmds.push_back(MakeJointCmd(i, hold_q_[i]));
  }
  return action;
}

void HoldLowerBodyPolicy::Reset() {
  hold_captured_ = false;
  hold_q_.clear();
}

}  // namespace policy_runner
