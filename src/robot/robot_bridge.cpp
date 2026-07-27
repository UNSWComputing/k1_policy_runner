#include "robot/robot_bridge.hpp"

#include <stdexcept>

namespace policy_runner {

namespace {
constexpr const char* kLowStateTopic = "rt/low_state";
}  // namespace

RobotBridge* RobotBridge::instance_ = nullptr;

RobotBridge::RobotBridge(const std::string& network_interface) {
  if (instance_ != nullptr) {
    throw std::runtime_error("RobotBridge: only one instance is supported");
  }
  instance_ = this;

  if (network_interface.empty()) {
    booster::robot::ChannelFactory::Instance()->Init(0);
  } else {
    booster::robot::ChannelFactory::Instance()->Init(0, network_interface);
  }

  publisher_.reset(
      new booster::robot::ChannelPublisher<booster_interface::msg::LowCmd>(
          booster::robot::b1::kTopicJointCtrl));

  subscriber_.reset(
      new booster::robot::ChannelSubscriber<booster_interface::msg::LowState>(
          kLowStateTopic, &RobotBridge::StateHandler));

  cmd_msg_.cmd_type(booster_interface::msg::CmdType::PARALLEL);
  for (std::size_t i = 0; i < kB1JointCount; ++i) {
    booster_interface::msg::MotorCmd motor_cmd;
    cmd_msg_.motor_cmd().push_back(motor_cmd);
  }
}

RobotBridge::~RobotBridge() {
  if (instance_ == this) {
    instance_ = nullptr;
  }
}

void RobotBridge::Start() {
  publisher_->InitChannel();
  subscriber_->InitChannel();
}

bool RobotBridge::HasState() const { return has_state_.load(); }

RobotState RobotBridge::LatestState() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return latest_state_;
}

void RobotBridge::StateHandler(const void* msg) {
  if (instance_ == nullptr) {
    return;
  }
  const auto* low_state =
      static_cast<const booster_interface::msg::LowState*>(msg);
  instance_->OnLowState(low_state);
}

void RobotBridge::OnLowState(
    const booster_interface::msg::LowState* low_state) {
  // Prefer parallel motor state to match PARALLEL LowCmd usage in examples.
  const auto& motors = low_state->motor_state_parallel().empty()
                           ? low_state->motor_state_serial()
                           : low_state->motor_state_parallel();

  RobotState state;
  state.q.resize(kB1JointCount, 0.f);
  state.dq.resize(kB1JointCount, 0.f);

  const std::size_t n =
      motors.size() < kB1JointCount ? motors.size() : kB1JointCount;
  for (std::size_t i = 0; i < n; ++i) {
    state.q[i] = motors[i].q();
    state.dq[i] = motors[i].dq();
  }

  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_state_ = std::move(state);
  }
  has_state_.store(true);
}

void RobotBridge::PublishAction(const Action& action) {
  // Clear all joints first so partial-body policies cannot leak prior commands.
  for (std::size_t i = 0; i < cmd_msg_.motor_cmd().size(); ++i) {
    auto& m = cmd_msg_.motor_cmd()[i];
    m.q(0.f);
    m.dq(0.f);
    m.tau(0.f);
    m.kp(0.f);
    m.kd(0.f);
    m.weight(0.f);
  }

  for (const auto& jc : action.joint_cmds) {
    if (jc.index < 0 ||
        static_cast<std::size_t>(jc.index) >= cmd_msg_.motor_cmd().size()) {
      throw std::runtime_error("PublishAction: joint index out of range");
    }
    auto& m = cmd_msg_.motor_cmd()[jc.index];
    m.q(jc.q);
    m.dq(jc.dq);
    m.tau(jc.tau);
    m.kp(jc.kp);
    m.kd(jc.kd);
    m.weight(jc.weight);
  }

  publisher_->Write(&cmd_msg_);
}

}  // namespace policy_runner
