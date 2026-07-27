#include "policy/sine_arm_policy.hpp"

#include "policy/arm_demo_common.hpp"

#include <cmath>
#include <stdexcept>

namespace policy_runner {
namespace {

constexpr float kElbowBase = 0.0f;
constexpr float kElbowAmp = 0.4f;
constexpr float kFreqHz = 0.5f;

}  // namespace

SineArmPolicy::SineArmPolicy(float control_dt) : control_dt_(control_dt) {}

std::string SineArmPolicy::name() const { return "sine_arm"; }

std::size_t SineArmPolicy::input_dim() const {
  return kBaseObsDim;
}

std::vector<int> SineArmPolicy::controlled_joints() const {
  return {arm_demo::kLeftElbow, arm_demo::kRightElbow};
}

Observation SineArmPolicy::BuildObservation(
    const RobotState& state,
    const std::vector<float>& command) const {
  return PackObservation(state, command);
}

Action SineArmPolicy::Infer(const Observation& obs) {
  AssertFrameObservation(obs);

  const float phase = 2.f * 3.14159265f * kFreqHz * time_s_;
  const float elbow_q = kElbowBase + kElbowAmp * std::sin(phase);

  Action action;
  action.joint_cmds.push_back(
      arm_demo::MakeElbowCmd(arm_demo::kLeftElbow, elbow_q));
  action.joint_cmds.push_back(
      arm_demo::MakeElbowCmd(arm_demo::kRightElbow, elbow_q));

  time_s_ += control_dt_;
  return action;
}

void SineArmPolicy::Reset() { time_s_ = 0.f; }

}  // namespace policy_runner
