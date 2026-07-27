#pragma once

#include "types.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <booster/idl/b1/LowCmd.h>
#include <booster/idl/b1/LowState.h>
#include <booster/idl/b1/MotorCmd.h>
#include <booster/robot/b1/b1_api_const.hpp>
#include <booster/robot/channel/channel_publisher.hpp>
#include <booster/robot/channel/channel_subscriber.hpp>

namespace policy_runner {

// Thin bridge around the Booster low-level DDS channels shown in the examples.
class RobotBridge {
 public:
  // network_interface may be empty; then Init(0) is used like the simple examples.
  explicit RobotBridge(const std::string& network_interface = "");
  ~RobotBridge();

  RobotBridge(const RobotBridge&) = delete;
  RobotBridge& operator=(const RobotBridge&) = delete;

  void Start();
  bool HasState() const;
  RobotState LatestState() const;

  // Writes a sparse Action onto a full LowCmd and publishes it.
  // Joints not present in `action` get weight=0 (uncontrolled).
  void PublishAction(const Action& action);

 private:
  static void StateHandler(const void* msg);
  void OnLowState(const booster_interface::msg::LowState* low_state);

  booster::robot::ChannelPublisherPtr<booster_interface::msg::LowCmd> publisher_;
  // Stack-style ownership matching the subscriber example; held via unique_ptr
  // so construction can pass the static handler.
  std::unique_ptr<
      booster::robot::ChannelSubscriber<booster_interface::msg::LowState>>
      subscriber_;

  mutable std::mutex state_mutex_;
  RobotState latest_state_;
  std::atomic<bool> has_state_{false};

  booster_interface::msg::LowCmd cmd_msg_;

  static RobotBridge* instance_;
};

}  // namespace policy_runner
