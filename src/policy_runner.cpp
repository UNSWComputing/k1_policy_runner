#include "policy/hold_lower_body_policy.hpp"
#include "policy/merge.hpp"
#include "policy/policy.hpp"
#include "policy/sine_arm_policy.hpp"
#include "policy/sine_knee_policy.hpp"
#include "policy/step_arm_policy.hpp"
#include "robot/robot_bridge.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr float kControlDt = 0.02f;  // 50 Hz, same as b1_low_sdk_example

void PrintUsage(const char* argv0) {
  std::cout
      << "Usage: " << argv0
      << " <policies> [networkInterface]\n"
      << "  policies: comma-separated list, e.g. sine_arm,hold_lower\n"
      << "  available: sine_arm | step_arm | sine_knee | hold_lower\n"
      << "\n"
      << "Before running: put the robot in Prepare mode, start this process,\n"
      << "press ENTER, then switch the robot to Custom mode.\n";
}

std::unique_ptr<policy_runner::Policy> MakePolicy(const std::string& name) {
  if (name == "sine_arm") {
    return std::make_unique<policy_runner::SineArmPolicy>(kControlDt);
  }
  if (name == "step_arm") {
    return std::make_unique<policy_runner::StepArmPolicy>(kControlDt);
  }
  if (name == "sine_knee") {
    return std::make_unique<policy_runner::SineKneePolicy>(kControlDt);
  }
  if (name == "hold_lower") {
    return std::make_unique<policy_runner::HoldLowerBodyPolicy>(kControlDt);
  }
  return nullptr;
}

std::vector<std::string> SplitPolicies(const std::string& csv) {
  std::vector<std::string> names;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      names.push_back(item);
    }
  }
  return names;
}

}  // namespace

int main(int argc, char const* argv[]) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return -1;
  }

  const std::vector<std::string> policy_names = SplitPolicies(argv[1]);
  const std::string network_iface = (argc >= 3) ? argv[2] : "";

  if (policy_names.empty()) {
    PrintUsage(argv[0]);
    return -1;
  }

  std::vector<std::unique_ptr<policy_runner::Policy>> policies;
  policies.reserve(policy_names.size());
  for (const auto& name : policy_names) {
    auto policy = MakePolicy(name);
    if (!policy) {
      std::cerr << "Unknown policy: " << name << std::endl;
      PrintUsage(argv[0]);
      return -1;
    }
    policies.push_back(std::move(policy));
  }

  policy_runner::RobotBridge robot(network_iface);
  robot.Start();

  for (const auto& policy : policies) {
    std::cout << "Policy: " << policy->name()
              << "  obs_dim=" << policy->observation_dim()
              << "  input_dim=" << policy->input_dim()
              << "  history_len=" << policy->history_len()
              << "  controlled_joints=" << policy->controlled_joints().size()
              << std::endl;
  }
  std::cout << "Waiting for LowState..." << std::endl;

  while (!robot.HasState()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  std::cout << "Press ENTER to start policy control..." << std::endl;
  std::cin.get();

  for (auto& policy : policies) {
    policy->Reset();
  }
  std::cout << "Running " << policies.size()
            << " polic(y/ies) in parallel. Ctrl+C to stop." << std::endl;

  const std::vector<float> command;

  const auto sleep_time =
      std::chrono::milliseconds(static_cast<int>(kControlDt * 1000.f));

  while (true) {
    const policy_runner::RobotState state = robot.LatestState();
    std::vector<policy_runner::Action> actions;
    actions.reserve(policies.size());
    for (auto& policy : policies) {
      const policy_runner::Observation obs =
          policy->BuildObservation(state, command);
      actions.push_back(policy->Infer(obs));
    }
    robot.PublishAction(policy_runner::MergeActions(actions));
    std::this_thread::sleep_for(sleep_time);
  }

  return 0;
}
