#pragma once

#include "types.hpp"

#include <unordered_map>
#include <vector>

namespace policy_runner {

// Merge sparse Actions from multiple policies.
// Later policies override earlier ones on the same joint index.
inline Action MergeActions(const std::vector<Action>& actions) {
  std::unordered_map<int, JointCommand> by_index;
  for (const auto& action : actions) {
    for (const auto& cmd : action.joint_cmds) {
      by_index[cmd.index] = cmd;
    }
  }

  Action merged;
  merged.joint_cmds.reserve(by_index.size());
  for (const auto& kv : by_index) {
    merged.joint_cmds.push_back(kv.second);
  }
  return merged;
}

}  // namespace policy_runner
