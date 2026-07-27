#include "policy/sine_knee_policy.hpp"

#include "joint_gains.hpp"
#include "joint_index.hpp"

#include <cmath>
#include <stdexcept>

namespace policy_runner {
namespace {

// LeftKneePitch=13, RightKneePitch=19 (22-DoF layout, no waist).
constexpr int kLeftKneePitch = joint::kLeftKneePitch;
constexpr int kRightKneePitch = joint::kRightKneePitch;

constexpr float kKneeBase = 0.4f;
constexpr float kKneeAmp = 0.2f;
constexpr float kFreqHz = 0.5f;

}  // namespace

SineKneePolicy::SineKneePolicy(float control_dt) : control_dt_(control_dt) {}

std::string SineKneePolicy::name() const { return "sine_knee"; }

std::size_t SineKneePolicy::input_dim() const {
  return kBaseObsDim;
}

std::vector<int> SineKneePolicy::controlled_joints() const {
  return {kLeftKneePitch, kRightKneePitch};
}

Observation SineKneePolicy::BuildObservation(
    const RobotState& state,
    const std::vector<float>& command) const {
  return PackObservation(state, command);
}

Action SineKneePolicy::Infer(const Observation& obs) {
  AssertFrameObservation(obs);

  const float phase = 2.f * 3.14159265f * kFreqHz * time_s_;
  const float knee_q = kKneeBase + kKneeAmp * std::sin(phase);

  Action action;
  action.joint_cmds.push_back(MakeJointCmd(kLeftKneePitch, knee_q));
  action.joint_cmds.push_back(MakeJointCmd(kRightKneePitch, knee_q));

  time_s_ += control_dt_;
  return action;
}

void SineKneePolicy::Reset() { time_s_ = 0.f; }

}  // namespace policy_runner
