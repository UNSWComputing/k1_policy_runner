"""Walk policy NUbots v1 — Isaac Lab k1_walk_htwk, 66-dim obs → 16-dim action.

From nubots_models/env.yaml + agent.yaml, deploy loop patterned on
htwk-gym ``k1_deploy`` (``deploy_parameter_walk_k1.py`` / ``utils/policy.py``).

Frame layout (66), Isaac Lab policy term order:
  projected_gravity (3)
  base_ang_vel (3)
  velocity_commands (12)  ParameterWalkCommand:
      vx, vy, yaw, gait_freq,
      foot_yaw_L, foot_yaw_R, body_pitch, body_roll,
      feet_off_x, feet_off_y, cos(2πφ), sin(2πφ)
  joint_pos (16)          q − DEFAULT, ACTION_JOINTS order
  joint_vel (16)          dq × 0.1
  actions (16)            last raw policy output

Joint I/O order (type-major, L/R interleaved) — obs joint blocks and
action dims share this layout:
  Shoulder_Pitch L/R, Hip_Pitch L/R,
  Shoulder_Roll L/R, Hip_Roll L/R,
  Hip_Yaw L/R, Knee_Pitch L/R,
  Ankle_Pitch L/R, Ankle_Roll L/R

Action: 16-D JointPositionAction (legs 12 + shoulder pitch/roll ×2).
  q_abs = DEFAULT + 0.8 * a_raw
  last_action feeds raw ONNX output (pre scale/offset).
  Head and elbows held at DEFAULT (not in the policy).

Gait (HTWK deploy): still (‖smoothed vx,vy,yaw‖ < 1e-5) → freq=0 and
clock zeros; else φ ← (φ + dt·f) mod 1. Velocity commands are rate-limited
by ±dt per step.

Default model: nubots_models/k1_walk_htwk_k1_walk_htwk.onnx
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from policy_runner.joint_index import JointIndex
from policy_runner.policy.base import Policy
from policy_runner.types import (
    B1_JOINT_COUNT,
    Action,
    JointCommand,
    Observation,
    RobotState,
)

# Confirmed NUbots / Isaac Lab 16-D joint order (obs + action).
# Head and elbows are not policy-controlled.
ACTION_JOINTS = np.asarray(
    [
        JointIndex.LEFT_SHOULDER_PITCH,
        JointIndex.RIGHT_SHOULDER_PITCH,
        JointIndex.LEFT_HIP_PITCH,
        JointIndex.RIGHT_HIP_PITCH,
        JointIndex.LEFT_SHOULDER_ROLL,
        JointIndex.RIGHT_SHOULDER_ROLL,
        JointIndex.LEFT_HIP_ROLL,
        JointIndex.RIGHT_HIP_ROLL,
        JointIndex.LEFT_HIP_YAW,
        JointIndex.RIGHT_HIP_YAW,
        JointIndex.LEFT_KNEE_PITCH,
        JointIndex.RIGHT_KNEE_PITCH,
        JointIndex.LEFT_ANKLE_PITCH,
        JointIndex.RIGHT_ANKLE_PITCH,
        JointIndex.LEFT_ANKLE_ROLL,
        JointIndex.RIGHT_ANKLE_ROLL,
    ],
    dtype=np.int64,
)
assert ACTION_JOINTS.shape == (16,)

ACTION_DIM = 16
ACTION_SCALE = 0.8  # env.yaml actions.joint_pos.scale
JOINT_VEL_SCALE = 0.1  # env.yaml observations.policy.joint_vel.scale

# Training actuator PD from env.yaml (BoosterDelayedPDActuator).
WALK_KP = {
    int(JointIndex.HEAD_YAW): 3.9478417602100686,
    int(JointIndex.HEAD_PITCH): 3.9478417602100686,
    int(JointIndex.LEFT_SHOULDER_PITCH): 3.9478417602100686,
    int(JointIndex.LEFT_SHOULDER_ROLL): 3.9478417602100686,
    int(JointIndex.LEFT_ELBOW_PITCH): 3.9478417602100686,
    int(JointIndex.LEFT_ELBOW_YAW): 3.9478417602100686,
    int(JointIndex.RIGHT_SHOULDER_PITCH): 3.9478417602100686,
    int(JointIndex.RIGHT_SHOULDER_ROLL): 3.9478417602100686,
    int(JointIndex.RIGHT_ELBOW_PITCH): 3.9478417602100686,
    int(JointIndex.RIGHT_ELBOW_YAW): 3.9478417602100686,
    int(JointIndex.LEFT_HIP_PITCH): 30.200989465607023,
    int(JointIndex.LEFT_HIP_ROLL): 21.447961045805584,
    int(JointIndex.LEFT_HIP_YAW): 17.846013389258083,
    int(JointIndex.LEFT_KNEE_PITCH): 60.401978931214046,
    int(JointIndex.LEFT_ANKLE_PITCH): 35.692026778516166,
    int(JointIndex.LEFT_ANKLE_ROLL): 35.692026778516166,
    int(JointIndex.RIGHT_HIP_PITCH): 30.200989465607023,
    int(JointIndex.RIGHT_HIP_ROLL): 21.447961045805584,
    int(JointIndex.RIGHT_HIP_YAW): 17.846013389258083,
    int(JointIndex.RIGHT_KNEE_PITCH): 60.401978931214046,
    int(JointIndex.RIGHT_ANKLE_PITCH): 35.692026778516166,
    int(JointIndex.RIGHT_ANKLE_ROLL): 35.692026778516166,
}
WALK_KD = {
    int(JointIndex.HEAD_YAW): 0.25132741228,
    int(JointIndex.HEAD_PITCH): 0.25132741228,
    int(JointIndex.LEFT_SHOULDER_PITCH): 0.25132741228,
    int(JointIndex.LEFT_SHOULDER_ROLL): 0.25132741228,
    int(JointIndex.LEFT_ELBOW_PITCH): 0.25132741228,
    int(JointIndex.LEFT_ELBOW_YAW): 0.25132741228,
    int(JointIndex.RIGHT_SHOULDER_PITCH): 0.25132741228,
    int(JointIndex.RIGHT_SHOULDER_ROLL): 0.25132741228,
    int(JointIndex.RIGHT_ELBOW_PITCH): 0.25132741228,
    int(JointIndex.RIGHT_ELBOW_YAW): 0.25132741228,
    int(JointIndex.LEFT_HIP_PITCH): 3.60497756989125,
    int(JointIndex.LEFT_HIP_ROLL): 2.560161764834957,
    int(JointIndex.LEFT_HIP_YAW): 2.1302109340993156,
    int(JointIndex.LEFT_KNEE_PITCH): 4.806636759855,
    int(JointIndex.LEFT_ANKLE_PITCH): 4.260421868198631,
    int(JointIndex.LEFT_ANKLE_ROLL): 4.260421868198631,
    int(JointIndex.RIGHT_HIP_PITCH): 3.60497756989125,
    int(JointIndex.RIGHT_HIP_ROLL): 2.560161764834957,
    int(JointIndex.RIGHT_HIP_YAW): 2.1302109340993156,
    int(JointIndex.RIGHT_KNEE_PITCH): 4.806636759855,
    int(JointIndex.RIGHT_ANKLE_PITCH): 4.260421868198631,
    int(JointIndex.RIGHT_ANKLE_ROLL): 4.260421868198631,
}

# env.yaml init_state.joint_pos (unlisted joints = 0).
DEFAULT_JOINT_POS = np.asarray(
    [
        0.0,  # Head_Yaw
        0.0,  # Head_Pitch
        0.0,  # Left_Shoulder_Pitch
        -1.3,  # Left_Shoulder_Roll
        0.0,  # Left_Elbow_Pitch
        0.0,  # Left_Elbow_Yaw
        0.0,  # Right_Shoulder_Pitch
        1.3,  # Right_Shoulder_Roll
        0.0,  # Right_Elbow_Pitch
        0.0,  # Right_Elbow_Yaw
        -0.2,  # Left_Hip_Pitch
        0.0,  # Left_Hip_Roll
        0.0,  # Left_Hip_Yaw
        0.4,  # Left_Knee_Pitch
        -0.25,  # Left_Ankle_Pitch
        0.0,  # Left_Ankle_Roll
        -0.2,  # Right_Hip_Pitch
        0.0,  # Right_Hip_Roll
        0.0,  # Right_Hip_Yaw
        0.4,  # Right_Knee_Pitch
        -0.25,  # Right_Ankle_Pitch
        0.0,  # Right_Ankle_Roll
    ],
    dtype=np.float64,
)

assert DEFAULT_JOINT_POS.shape == (B1_JOINT_COUNT,)

DEFAULT_ACTION_POS = DEFAULT_JOINT_POS[ACTION_JOINTS].copy()
assert DEFAULT_ACTION_POS.shape == (ACTION_DIM,)

FRAME_PROJ_GRAV = 3
FRAME_ANG_VEL = 3
FRAME_COMMAND = 12
FRAME_JOINT_POS = 16
FRAME_JOINT_VEL = 16
FRAME_ACTIONS = 16
FRAME_DIM = (
    FRAME_PROJ_GRAV
    + FRAME_ANG_VEL
    + FRAME_COMMAND
    + FRAME_JOINT_POS
    + FRAME_JOINT_VEL
    + FRAME_ACTIONS
)
assert FRAME_DIM == 66

DEFAULT_SETTLE_S = 0.4
GAIT_STILL_THRESHOLD = 1.0e-5
GAIT_FREQUENCY_HZ = 1.7  # HTWK streamlit default; train range [1.5, 3.0]
BODY_PITCH_TARGET = 0.1
BODY_ROLL_TARGET = 0.0
FOOT_YAW_L = 0.0
FOOT_YAW_R = 0.0
FEET_OFFSET_X = 0.0
FEET_OFFSET_Y = 0.0

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    _REPO_ROOT / "nubots_models" / "k1_walk_htwk_k1_walk_htwk.onnx"
)


def make_walk_joint_cmd(index: int, q: float) -> JointCommand:
    return JointCommand(
        index=index,
        q=q,
        dq=0.0,
        tau=0.0,
        kp=WALK_KP[index],
        kd=WALK_KD[index],
        weight=1.0,
    )


def full_body_action(action_q: Sequence[float]) -> Action:
    """All 22 joints: ACTION_JOINTS from `action_q`, others at DEFAULT."""
    q_act = np.asarray(action_q, dtype=np.float64)
    if q_act.shape != (ACTION_DIM,):
        raise ValueError("full_body_action: expected 16 action qs")

    q_full = DEFAULT_JOINT_POS.copy()
    q_full[ACTION_JOINTS] = q_act
    cmds = [make_walk_joint_cmd(i, float(q_full[i])) for i in range(B1_JOINT_COUNT)]
    return Action(joint_cmds=cmds)


def default_pose_action() -> Action:
    return full_body_action(DEFAULT_ACTION_POS)


def action_to_absolute(action: Sequence[float]) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"action_to_absolute: expected {ACTION_DIM} dims")
    return DEFAULT_ACTION_POS + ACTION_SCALE * a


class WalkPolicyNubotsV1(Policy):
    """observation_dim = input_dim = 66 (single frame). Full 22-DoF commands."""

    def __init__(
        self,
        control_dt: float = 0.02,
        model_path: Optional[Union[str, Path]] = None,
        load_default_model: bool = True,
        settle_s: float = DEFAULT_SETTLE_S,
        gait_frequency_hz: float = GAIT_FREQUENCY_HZ,
        body_pitch_target: float = BODY_PITCH_TARGET,
        body_roll_target: float = BODY_ROLL_TARGET,
        foot_yaw_l: float = FOOT_YAW_L,
        foot_yaw_r: float = FOOT_YAW_R,
        feet_offset_x: float = FEET_OFFSET_X,
        feet_offset_y: float = FEET_OFFSET_Y,
    ) -> None:
        self._control_dt = float(control_dt)
        self._settle_s = float(settle_s)
        self._gait_frequency_hz = float(gait_frequency_hz)
        self._body_pitch_target = float(body_pitch_target)
        self._body_roll_target = float(body_roll_target)
        self._foot_yaw_l = float(foot_yaw_l)
        self._foot_yaw_r = float(foot_yaw_r)
        self._feet_offset_x = float(feet_offset_x)
        self._feet_offset_y = float(feet_offset_y)

        self._last_action: List[float] = [0.0] * ACTION_DIM
        self._cmd = np.zeros(3, dtype=np.float64)
        self._smoothed_cmd = np.zeros(3, dtype=np.float64)
        self._gait_phase = 0.0
        self._settle_t0: Optional[float] = None
        self._rl_started = False

        self._session: Optional[ort.InferenceSession] = None
        self._input_name = "obs"
        self._output_name = "actions"

        path = model_path
        if path is None and load_default_model:
            path = DEFAULT_MODEL_PATH
        if path is not None:
            self.load_model(str(path))

    def name(self) -> str:
        return "walk_nubots_v1"

    def observation_dim(self) -> int:
        return FRAME_DIM

    def history_len(self) -> int:
        return 1

    def input_dim(self) -> int:
        return FRAME_DIM

    def controlled_joints(self) -> List[int]:
        return list(range(B1_JOINT_COUNT))

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"walk_nubots_v1: model not found: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"walk_nubots_v1: ONNX has no inputs/outputs: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        in_dims = [d for d in inputs[0].shape if isinstance(d, int) and d > 0]
        out_dims = [d for d in outputs[0].shape if isinstance(d, int) and d > 0]
        if not in_dims or int(in_dims[-1]) != FRAME_DIM:
            raise RuntimeError(
                f"walk_nubots_v1: obs dim {in_dims} != {FRAME_DIM}"
            )
        if not out_dims or int(out_dims[-1]) != ACTION_DIM:
            raise RuntimeError(
                f"walk_nubots_v1: action dim {out_dims} != {ACTION_DIM}"
            )
        print(
            f"[walk_nubots_v1] loaded {path.name}: in {self._input_name!r} "
            f"[1, {FRAME_DIM}] -> out {self._output_name!r} [1, {ACTION_DIM}]"
        )

    def _parameter_walk_command(self) -> np.ndarray:
        """12-D ParameterWalkCommand, HTWK-style still/clock handling."""
        if not self._rl_started:
            return np.zeros(FRAME_COMMAND, dtype=np.float64)

        clip = self._control_dt
        delta = np.clip(self._cmd - self._smoothed_cmd, -clip, clip)
        self._smoothed_cmd += delta

        still = float(np.linalg.norm(self._smoothed_cmd)) < GAIT_STILL_THRESHOLD
        still = False
        if still:
            self._gait_phase = 0.0
            vx = vy = yaw = 0.0
            freq = 0.0
            clock_c = 0.0
            clock_s = 0.0
        else:
            vx, vy, yaw = (float(x) for x in self._smoothed_cmd)
            freq = self._gait_frequency_hz
            self._gait_phase = (self._gait_phase + self._control_dt * freq) % 1.0
            ang = 2.0 * math.pi * self._gait_phase
            clock_c = math.cos(ang)
            clock_s = math.sin(ang)

        return np.asarray(
            [
                vx,
                vy,
                yaw,
                freq,
                self._foot_yaw_l,
                self._foot_yaw_r,
                self._body_pitch_target,
                self._body_roll_target,
                self._feet_offset_x,
                self._feet_offset_y,
                clock_c,
                clock_s,
            ],
            dtype=np.float64,
        )

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        if len(state.q) != B1_JOINT_COUNT or len(state.dq) != B1_JOINT_COUNT:
            raise ValueError("walk_nubots_v1: RobotState q/dq size mismatch")

        cmd = [float(x) for x in command[:3]]
        while len(cmd) < 3:
            cmd.append(0.0)
        self._cmd = np.asarray(cmd, dtype=np.float64)

        grav = np.zeros(FRAME_PROJ_GRAV, dtype=np.float64)
        grav[: min(FRAME_PROJ_GRAV, len(state.projected_gravity))] = (
            state.projected_gravity[:FRAME_PROJ_GRAV]
        )
        gyro = np.zeros(FRAME_ANG_VEL, dtype=np.float64)
        gyro[: min(FRAME_ANG_VEL, len(state.imu.gyro))] = state.imu.gyro[
            :FRAME_ANG_VEL
        ]

        q = np.asarray(state.q, dtype=np.float64)
        dq = np.asarray(state.dq, dtype=np.float64)
        joint_pos = (q - DEFAULT_JOINT_POS)[ACTION_JOINTS]
        joint_vel = dq[ACTION_JOINTS] * JOINT_VEL_SCALE
        last_a = np.asarray(self._last_action, dtype=np.float64)
        pw_cmd = self._parameter_walk_command()

        data = np.concatenate(
            [grav, gyro, pw_cmd, joint_pos, joint_vel, last_a]
        )
        if data.shape != (FRAME_DIM,):
            raise ValueError(
                f"walk_nubots_v1: built frame dim {data.size} != {FRAME_DIM}"
            )
        return Observation(data=data.tolist())

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        now = time.perf_counter()
        if self._settle_t0 is None:
            self._settle_t0 = now
            print(
                f"[walk_nubots_v1] settling to DEFAULT_JOINT_POS for "
                f"{self._settle_s:.1f}s..."
            )

        if (now - self._settle_t0) < self._settle_s:
            return default_pose_action()

        if not self._rl_started:
            self._last_action = [0.0] * ACTION_DIM
            self._gait_phase = 0.0
            self._smoothed_cmd[:] = 0.0
            self._rl_started = True
            print("[walk_nubots_v1] settle done — starting RL walk")

        if self._session is None:
            raise RuntimeError("walk_nubots_v1: model not loaded; call load_model()")

        x = np.asarray(obs.data, dtype=np.float32).reshape(1, FRAME_DIM)
        y = self._session.run(
            [self._output_name], {self._input_name: x}
        )[0]
        action = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(action) != ACTION_DIM:
            raise ValueError(
                f"walk_nubots_v1: action dim {len(action)} != {ACTION_DIM}"
            )

        self._last_action = [float(a) for a in action]
        q_abs = action_to_absolute(action)
        return full_body_action(q_abs)

    def reset(self) -> None:
        self._last_action = [0.0] * ACTION_DIM
        self._cmd[:] = 0.0
        self._smoothed_cmd[:] = 0.0
        self._gait_phase = 0.0
        self._settle_t0 = None
        self._rl_started = False
