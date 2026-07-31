#!/usr/bin/env python3
"""MuJoCo sim-to-sim runner — same policies as policy_runner_main, no ROS/robot.

Example:
  python3 mujoco_runner_main.py parameter_walk \\
    --model-path ../parameter_walk_model_20000.pt \\
    --duration 10 --record ./recordings
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mujoco.viewer

from policy_runner.policy import (
    HoldLowerBodyPolicy,
    Policy,
    SineArmPolicy,
    SineKneePolicy,
    StepArmPolicy,
    WalkPolicy,
    WalkPolicyV1,
    WalkPolicyV2,
    merge_actions,
)
from policy_runner.sim import DEFAULT_MJCF, MujocoBridge
from policy_runner.sim.recorder import SimRecorder

try:
    from policy_runner.policy.parameter_walk_policy import ParameterWalkPolicy
except ImportError:  # pragma: no cover
    ParameterWalkPolicy = None  # type: ignore

CONTROL_DT = 0.02  # 50 Hz — matches train decimation (0.002 * 10)
AVAILABLE = (
    "sine_arm",
    "step_arm",
    "sine_knee",
    "hold_lower",
    "walk",
    "walk_v1",
    "walk_v2",
    "parameter_walk",
)


def make_policy(
    name: str,
    model_path: Optional[str] = None,
    cmd: Optional[list[float]] = None,
    gait_freq: Optional[float] = None,
) -> Optional[Policy]:
    if name == "sine_arm":
        return SineArmPolicy(CONTROL_DT)
    if name == "step_arm":
        return StepArmPolicy(CONTROL_DT)
    if name == "sine_knee":
        return SineKneePolicy(CONTROL_DT)
    if name == "hold_lower":
        return HoldLowerBodyPolicy(CONTROL_DT)
    if name == "walk":
        if WalkPolicy is None:
            raise RuntimeError("walk stub missing; use walk_v1/walk_v2 or parameter_walk")
        return WalkPolicy(CONTROL_DT)
    if name == "walk_v1":
        if WalkPolicyV1 is None:
            raise RuntimeError("walk_v1 unavailable (install onnxruntime)")
        return WalkPolicyV1(CONTROL_DT, model_path=model_path)
    if name == "walk_v2":
        if WalkPolicyV2 is None:
            raise RuntimeError("walk_v2 unavailable (install onnxruntime)")
        return WalkPolicyV2(CONTROL_DT, model_path=model_path)
    if name == "parameter_walk":
        if ParameterWalkPolicy is None:
            raise RuntimeError("parameter_walk policy not available")
        kwargs = {"model_path": model_path}
        if cmd is not None:
            kwargs["command"] = cmd
        if gait_freq is not None:
            kwargs["gait_frequency"] = gait_freq
        return ParameterWalkPolicy(CONTROL_DT, **kwargs)
    return None


def parse_policy_list(csv: str) -> List[str]:
    return [p.strip() for p in csv.split(",") if p.strip()]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run policies in MuJoCo (sim-to-sim, no ROS)."
    )
    parser.add_argument(
        "policies",
        help=f"Comma-separated policies. Available: {', '.join(AVAILABLE)}",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model for walk_v1 (.onnx) or parameter_walk (.pt/.onnx)",
    )
    parser.add_argument(
        "--cmd",
        default=None,
        help="Comma-separated ParameterWalk cmds (up to 10)",
    )
    parser.add_argument(
        "--gait-freq",
        type=float,
        default=None,
        help="ParameterWalk gait clock Hz",
    )
    parser.add_argument(
        "--mjcf",
        default=str(DEFAULT_MJCF),
        help=f"Scene MJCF (default: {DEFAULT_MJCF})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after N seconds of simulated time (0 = until viewer closed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No viewer; run for --duration (required if headless)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep to wall-clock control_dt (default: run as fast as possible)",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="DIR",
        help="Save debug recording to DIR (npz states + mp4 video)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="With --record, skip mp4 and only write .npz",
    )
    parser.add_argument(
        "--record-name",
        default=None,
        help="Recording basename (default: mujoco_YYYYMMDD-HHMMSS)",
    )
    args = parser.parse_args(argv)

    names = parse_policy_list(args.policies)
    if not names:
        print("No policies specified.", file=sys.stderr)
        return 1
    if args.headless and args.duration <= 0:
        print("--headless requires --duration > 0", file=sys.stderr)
        return 1

    cmd: Optional[list[float]] = None
    if args.cmd:
        cmd = [float(x) for x in args.cmd.split(",") if x.strip() != ""]

    policies: List[Policy] = []
    for name in names:
        try:
            policy = make_policy(
                name,
                model_path=args.model_path,
                cmd=cmd,
                gait_freq=args.gait_freq,
            )
        except Exception as exc:
            print(f"Failed to create policy {name}: {exc}", file=sys.stderr)
            return 1
        if policy is None:
            print(f"Unknown policy: {name}", file=sys.stderr)
            return 1
        policies.append(policy)

    # Match MuJoCo hold/init pose to the policy's obs default when available.
    default_q = None
    if any(p.name() == "walk_v2" for p in policies):
        from policy_runner.policy.walk_policy_v2 import DEFAULT_JOINT_POS

        default_q = DEFAULT_JOINT_POS
        print("MuJoCo default_q ← walk_v2 DEFAULT_JOINT_POS")
    elif any(p.name() == "walk_v1" for p in policies):
        from policy_runner.policy.walk_policy_v1 import DEFAULT_JOINT_POS

        default_q = DEFAULT_JOINT_POS
        print("MuJoCo default_q ← walk_v1 DEFAULT_JOINT_POS")

    bridge = MujocoBridge(
        mjcf_path=args.mjcf, control_dt=CONTROL_DT, default_q=default_q
    )
    for policy in policies:
        policy.reset()
        print(
            f"Policy: {policy.name()}  "
            f"obs_dim={policy.observation_dim()}  "
            f"input_dim={policy.input_dim()}  "
            f"history_len={policy.history_len()}  "
            f"controlled_joints={len(policy.controlled_joints())}"
        )
    print(
        f"MuJoCo: {args.mjcf}  "
        f"physics_dt={bridge.model.opt.timestep}  "
        f"decimation={bridge.decimation}  "
        f"control_dt={CONTROL_DT}"
    )

    recorder: Optional[SimRecorder] = None
    if args.record:
        run_name = args.record_name
        if run_name is None:
            run_name = f"{'_'.join(names)}"
        recorder = SimRecorder(
            bridge,
            args.record,
            record_video=not args.no_video,
            fps=1.0 / CONTROL_DT,
            run_name=run_name,
        )
        print(
            f"Recording → {args.record}/ "
            f"(video={'off' if args.no_video else 'on'})"
        )

    command: list[float] = list(cmd) if cmd is not None else []
    wall_t0 = time.perf_counter()
    sim_t = 0.0
    steps = 0
    max_steps = (
        int(round(args.duration / CONTROL_DT)) if args.duration > 0 else None
    )

    def _tick() -> None:
        nonlocal steps, sim_t
        state = bridge.latest_state()
        observations = [
            policy.build_observation(state, command) for policy in policies
        ]
        actions = [
            policy.infer(obs) for policy, obs in zip(policies, observations)
        ]
        bridge.publish_action(merge_actions(actions))
        bridge.step()
        steps += 1
        sim_t = steps * CONTROL_DT
        if recorder is not None:
            # Primary policy obs (first) for debug; full state + PD targets.
            recorder.log(sim_t, state, obs=observations[0])

    try:
        if args.headless:
            assert max_steps is not None
            while steps < max_steps:
                _tick()
                if args.realtime:
                    time.sleep(CONTROL_DT)
        else:
            with mujoco.viewer.launch_passive(bridge.model, bridge.data) as viewer:
                viewer.cam.elevation = -20
                while viewer.is_running():
                    if max_steps is not None and steps >= max_steps:
                        break
                    step_t0 = time.perf_counter()
                    _tick()
                    viewer.cam.lookat[:] = bridge.data.qpos[0:3]
                    viewer.sync()
                    if args.realtime:
                        elapsed = time.perf_counter() - step_t0
                        time.sleep(max(0.0, CONTROL_DT - elapsed))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if recorder is not None:
            info = recorder.close()
            print(f"Saved recording: {info['npz']}")
            if info.get("video"):
                print(f"Saved video:      {info['video']}")

    wall = time.perf_counter() - wall_t0
    base_z = float(bridge.data.qpos[2])
    print(
        f"Done: {steps} steps, sim_t={sim_t:.2f}s, wall={wall:.2f}s "
        f"({steps / max(wall, 1e-6):.1f} Hz loop), base_z={base_z:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
