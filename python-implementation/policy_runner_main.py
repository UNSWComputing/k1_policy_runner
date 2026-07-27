#!/usr/bin/env python3
"""Main control loop — mirrors src/policy_runner.cpp, with ROS2 I/O."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Allow `python3 policy_runner_main.py` from this directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import rclpy

from booster_robotics_sdk_python import (  # type: ignore
    B1LocoClient,
    ChannelFactory,
    RobotMode,
)

from policy_runner.policy import (
    HoldLowerBodyPolicy,
    Policy,
    SineArmPolicy,
    SineKneePolicy,
    StepArmPolicy,
    merge_actions,
)
from policy_runner.robot import RobotBridge, spin_bridge_in_background

CONTROL_DT = 0.01  # 100 Hz
AVAILABLE = ("sine_arm", "step_arm", "sine_knee", "hold_lower")


def make_policy(name: str) -> Optional[Policy]:
    if name == "sine_arm":
        return SineArmPolicy(CONTROL_DT)
    if name == "step_arm":
        return StepArmPolicy(CONTROL_DT)
    if name == "sine_knee":
        return SineKneePolicy(CONTROL_DT)
    if name == "hold_lower":
        return HoldLowerBodyPolicy(CONTROL_DT)
    return None


def parse_policy_list(csv: str) -> List[str]:
    return [p.strip() for p in csv.split(",") if p.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more policies in parallel over ROS2 "
            "(/joint_states + /low_state IMU -> /joint_ctrl). "
            "Sparse actions are merged."
        )
    )
    parser.add_argument(
        "policies",
        help="Comma-separated policies, e.g. sine_arm,hold_lower. "
        f"Available: {', '.join(AVAILABLE)}",
    )
    parser.add_argument(
        "--joint-states-topic",
        default="/joint_states",
        help="Input JointState topic (default: /joint_states)",
    )
    parser.add_argument(
        "--low-state-topic",
        default="/low_state",
        help="Input LowState topic for IMU (default: /low_state)",
    )
    parser.add_argument(
        "--joint-ctrl-topic",
        default="/joint_ctrl",
        help="Output LowCmd topic (default: /joint_ctrl)",
    )
    args = parser.parse_args(argv)

    names = parse_policy_list(args.policies)
    if not names:
        print("No policies specified.", file=sys.stderr)
        return 1

    policies: List[Policy] = []
    for name in names:
        policy = make_policy(name)
        if policy is None:
            print(f"Unknown policy: {name}", file=sys.stderr)
            print(f"Available: {', '.join(AVAILABLE)}", file=sys.stderr)
            return 1
        policies.append(policy)

    rclpy.init(args=None)
    bridge = RobotBridge(
        joint_state_topic=args.joint_states_topic,
        joint_ctrl_topic=args.joint_ctrl_topic,
        low_state_topic=args.low_state_topic,
    )
    spin_bridge_in_background(bridge)

    for policy in policies:
        print(
            f"Policy: {policy.name()}  input_dim={policy.input_dim()}  "
            f"controlled_joints={len(policy.controlled_joints())}"
        )
    print("Waiting for /joint_states and /low_state (IMU)...")
    ChannelFactory.Instance().Init(0)
    client = B1LocoClient()
    client.Init()
    time.sleep(2)

    try:
        while rclpy.ok() and not bridge.has_state():
            time.sleep(0.1)

        if not rclpy.ok():
            return 1

        input("Press ENTER to start policy control...")
        client.ChangeMode(RobotMode.kCustom)
        for policy in policies:
            policy.reset()
        print(f"Running {len(policies)} polic(y/ies) in parallel. Ctrl+C to stop.")

        command: list[float] = []

        while rclpy.ok():
            state = bridge.latest_state()
            actions = [
                policy.infer(policy.build_observation(state, command))
                for policy in policies
            ]
            bridge.publish_action(merge_actions(actions))
            time.sleep(CONTROL_DT)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
