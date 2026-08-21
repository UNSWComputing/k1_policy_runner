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

from policy_runner.obs_record import ModelInputRecorder
from policy_runner.policy import (
    HoldLowerBodyPolicy,
    Policy,
    SineAnklePolicy,
    SineArmPolicy,
    SineHeadPolicy,
    SineHipPolicy,
    SineKneePolicy,
    StepAnklePolicy,
    StepArmPolicy,
    WalkPolicy,
    WalkPolicyV1,
    WalkPolicyV2,
    WalkPolicyV3,
    WalkPolicyV4,
    WalkPolicyV5,
    WalkPolicyV6,
    WalkPolicyNubotsV1,
    merge_actions,
)
from policy_runner.robot import RobotBridge, spin_bridge_in_background

CONTROL_DT = 0.02  # 50 Hz target control period
AVAILABLE = (
    "sine_arm",
    "sine_hip",
    "sine_ankle",
    "sine_head",
    "step_arm",
    "step_ankle",
    "sine_knee",
    "hold_lower",
    "walk",
    "walk_v1",
    "walk_v2",
    "walk_v3",
    "walk_v4",
    "walk_v5",
    "walk_v6",
    "walk_nubots_v1",
)


def make_policy(
    name: str,
    model_path: Optional[str] = None,
    recorder: Optional[ModelInputRecorder] = None,
) -> Optional[Policy]:
    if name == "sine_arm":
        return SineArmPolicy(CONTROL_DT)
    if name == "sine_hip":
        return SineHipPolicy(CONTROL_DT)
    if name == "sine_ankle":
        return SineAnklePolicy(CONTROL_DT)
    if name == "sine_head":
        return SineHeadPolicy(CONTROL_DT)
    if name == "step_arm":
        return StepArmPolicy(CONTROL_DT)
    if name == "step_ankle":
        return StepAnklePolicy(CONTROL_DT)
    if name == "sine_knee":
        return SineKneePolicy(CONTROL_DT)
    if name == "hold_lower":
        return HoldLowerBodyPolicy(CONTROL_DT)
    if name == "walk":
        return WalkPolicy(CONTROL_DT)
    if name == "walk_v1":
        return WalkPolicyV1(CONTROL_DT, model_path=model_path, recorder=recorder)
    if name == "walk_v2":
        return WalkPolicyV2(CONTROL_DT, model_path=model_path)
    if name == "walk_v3":
        return WalkPolicyV3(CONTROL_DT, model_path=model_path)
    if name == "walk_v4":
        return WalkPolicyV4(CONTROL_DT, model_path=model_path)
    if name == "walk_v5":
        if WalkPolicyV5 is None:
            raise RuntimeError("walk_v5 unavailable (install onnxruntime)")
        return WalkPolicyV5(CONTROL_DT, model_path=model_path)
    if name == "walk_v6":
        if WalkPolicyV6 is None:
            raise RuntimeError("walk_v6 unavailable (install onnxruntime)")
        return WalkPolicyV6(CONTROL_DT, model_path=model_path)
    if name == "walk_nubots_v1":
        if WalkPolicyNubotsV1 is None:
            raise RuntimeError("walk_nubots_v1 unavailable (install onnxruntime)")
        return WalkPolicyNubotsV1(CONTROL_DT, model_path=model_path)
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
    parser.add_argument(
        "--cmd-vel-topic",
        default="/cmd_vel",
        help="Twist topic for walk velocity [vx, vy, yaw] (default: /cmd_vel)",
    )
    parser.add_argument(
        "--enable-topic",
        default="/nubots_walk/enable",
        help=(
            "Bool topic to pause/resume /joint_ctrl publishing "
            "(default: /nubots_walk/enable; default enabled until false)"
        ),
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Skip the ENTER prompt and start Custom mode immediately",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="ONNX model for walk_v1 / walk_v2 / walk_v3 / walk_v4 / walk_v5 / walk_v6 / walk_nubots_v1",
    )
    parser.add_argument(
        "--record-obs",
        default=None,
        metavar="PATH",
        help=(
            "Record walk_v1 195-D model inputs to PATH.npz "
            "(layout meta saved alongside as PATH.json)"
        ),
    )
    args = parser.parse_args(argv)

    names = parse_policy_list(args.policies)
    if not names:
        print("No policies specified.", file=sys.stderr)
        return 1

    recorder: Optional[ModelInputRecorder] = None
    if args.record_obs:
        if "walk_v1" not in names:
            print(
                "--record-obs requires walk_v1 in the policy list",
                file=sys.stderr,
            )
            return 1
        recorder = ModelInputRecorder(args.record_obs)
        print(f"Recording walk_v1 model inputs → {Path(args.record_obs)}")

    policies: List[Policy] = []
    for name in names:
        policy = make_policy(
            name,
            model_path=args.model_path,
            recorder=recorder if name == "walk_v1" else None,
        )
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
        cmd_vel_topic=args.cmd_vel_topic,
        enable_topic=args.enable_topic,
    )
    spin_bridge_in_background(bridge)

    for policy in policies:
        print(
            f"Policy: {policy.name()}  "
            f"obs_dim={policy.observation_dim()}  "
            f"input_dim={policy.input_dim()}  "
            f"history_len={policy.history_len()}  "
            f"controlled_joints={len(policy.controlled_joints())}"
        )
    print(
        f"cmd_vel topic: {args.cmd_vel_topic} → command [vx, vy, yaw_rate]"
    )
    print(
        f"enable topic: {args.enable_topic} "
        f"(false → stop /joint_ctrl; true → Custom + reset)"
    )
    print("Waiting for /joint_states and /low_state (IMU)...")
    ChannelFactory.Instance().Init(0)
    client = B1LocoClient()
    client.Init()

    try:
        while rclpy.ok() and not bridge.has_state():
            time.sleep(0.1)

        if not rclpy.ok():
            return 1

        if not args.auto_start:
            input("Press ENTER to start policy control...")
        client.ChangeMode(RobotMode.kCustom)
        for policy in policies:
            policy.reset()
        print(f"Running {len(policies)} polic(y/ies) in parallel. Ctrl+C to stop.")

        loop_count = 0
        freq_t0 = time.perf_counter()
        next_tick = time.perf_counter()
        was_enabled = True

        while rclpy.ok():
            enabled = bridge.is_enabled()
            if enabled and not was_enabled:
                print("enable=true → ChangeMode(kCustom) + policy reset")
                try:
                    client.ChangeMode(RobotMode.kCustom)
                except Exception as e:
                    print(f"ChangeMode(kCustom) failed: {e}")
                for policy in policies:
                    policy.reset()
            elif not enabled and was_enabled:
                print("enable=false → paused (/joint_ctrl stopped)")
            was_enabled = enabled

            state = bridge.latest_state()
            command = bridge.latest_command()
            if enabled:
                actions = [
                    policy.infer(policy.build_observation(state, command))
                    for policy in policies
                ]
                bridge.publish_action(merge_actions(actions))
            loop_count += 1

            elapsed = time.perf_counter() - freq_t0
            if elapsed >= 1.0:
                print(
                    f"control loop: {loop_count / elapsed:.1f} Hz  "
                    f"enabled={enabled}  "
                    f"cmd_vel=[{command[0]:.2f}, {command[1]:.2f}, "
                    f"{command[2]:.2f}]"
                )
                loop_count = 0
                freq_t0 = time.perf_counter()

            # Sleep only the remainder of this period (work + sleep ≈ CONTROL_DT).
            next_tick += CONTROL_DT
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -CONTROL_DT:
                # Fell more than one period behind; resync instead of burst catch-up.
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if recorder is not None and len(recorder) > 0:
            recorder.save()
        bridge.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
