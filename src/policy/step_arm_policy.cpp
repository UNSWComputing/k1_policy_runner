#include "policy/step_arm_policy.hpp"

#include "policy/arm_demo_common.hpp"

#include <stdexcept>

namespace policy_runner {
namespace {

constexpr float kElbowA = 0.0f;
constexpr float kElbowB = -0.6f;

}  // namespace

StepArmPolicy::StepArmPolicy(float control_dt, float period_s)
    : control_dt_(control_dt), period_s_(period_s) {}

std::string StepArmPolicy::name() const { return "step_arm"; }

std::size_t StepArmPolicy::input_dim() const {
  return kBaseObsDim;
}

std::vector<int> StepArmPolicy::controlled_joints() const {
  return {arm_demo::kLeftElbow, arm_demo::kRightElbow};
}

Observation StepArmPolicy::BuildObservation(
    const RobotState& state,
    const std::vector<float>& command) const {
  return PackObservation(state, command);
}

Action StepArmPolicy::Infer(const Observation& obs) {
  AssertFrameObservation(obs);

  const bool use_b =
      (static_cast<int>(time_s_ / period_s_) % 2) == 1;
  const float elbow_q = use_b ? kElbowB : kElbowA;

  Action action;
  action.joint_cmds.push_back(
      arm_demo::MakeElbowCmd(arm_demo::kLeftElbow, elbow_q));
  action.joint_cmds.push_back(
      arm_demo::MakeElbowCmd(arm_demo::kRightElbow, elbow_q));

  time_s_ += control_dt_;
  return action;
}

void StepArmPolicy::Reset() { time_s_ = 0.f; }

}  // namespace policy_runner
