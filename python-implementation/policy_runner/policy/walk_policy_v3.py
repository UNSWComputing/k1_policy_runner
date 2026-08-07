"""Walk policy v3 — 61-dim frame obs × history 3 + gait_clock(2) → 12-dim leg action.

Frame layout (61), history term-major ×3 → 183:
  base_ang_vel (3)       IMU gyro
  projected_gravity (3)
  joint_pos (20)         arms+legs, q relative to DEFAULT (no head)
  joint_vel (20)         same joints
  actions (12)           last commanded action (clamped, relative)
  command (3)            twist (vx, vy, ωz)

Appended once (no history):
  gait_clock (2)         [cos φ, sin φ]

Model input = 183 + 2 = 185.

Joint pos in obs and actions are relative to DEFAULT_JOINT_POS.
Absolute robot commands via offset layer only:
  q_abs = DEFAULT + scale * a_rel

Startup: hold all joints at DEFAULT_JOINT_POS for settle_s, then RL walk.
During RL, legs follow the policy; upper body is held at DEFAULT with
joint_gains PD.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from policy_runner.joint_gains import make_joint_cmd
from policy_runner.joint_index import JointIndex
from policy_runner.obs_record import ModelInputRecorder
from policy_runner.policy.base import Policy
from policy_runner.policy.obs_history import ObservationHistory
from policy_runner.types import (
    B1_JOINT_COUNT,
    Action,
    JointCommand,
    Observation,
    RobotState,
)

# ---------------------------------------------------------------------------
# Offset layer — default / HOME pose (absolute rad).
# ---------------------------------------------------------------------------

LEG_JOINTS = np.asarray(
    [
        JointIndex.LEFT_HIP_PITCH,
        JointIndex.LEFT_HIP_ROLL,
        JointIndex.LEFT_HIP_YAW,
        JointIndex.LEFT_KNEE_PITCH,
        JointIndex.LEFT_ANKLE_PITCH,
        JointIndex.LEFT_ANKLE_ROLL,
        JointIndex.RIGHT_HIP_PITCH,
        JointIndex.RIGHT_HIP_ROLL,
        JointIndex.RIGHT_HIP_YAW,
        JointIndex.RIGHT_KNEE_PITCH,
        JointIndex.RIGHT_ANKLE_PITCH,
        JointIndex.RIGHT_ANKLE_ROLL,
    ],
    dtype=np.int64,
)
assert LEG_JOINTS.shape == (12,)

# Arms + legs only in obs (exclude Head_Yaw=0, Head_Pitch=1).
OBS_BODY_JOINTS = np.arange(
    int(JointIndex.LEFT_SHOULDER_PITCH), B1_JOINT_COUNT, dtype=np.int64
)
assert OBS_BODY_JOINTS.shape == (20,)

# Head + arms (JointIndex 0..9). Held at DEFAULT during settle and RL.
UPPER_BODY_JOINTS = np.arange(0, int(JointIndex.LEFT_HIP_PITCH), dtype=np.int64)
assert UPPER_BODY_JOINTS.shape == (10,)

# Ankles softer + more damped than hips/knees — reduces swing-phase shake.
WALK_KP = {
    int(JointIndex.LEFT_HIP_PITCH): 80.0,
    int(JointIndex.LEFT_HIP_ROLL): 80.0,
    int(JointIndex.LEFT_HIP_YAW): 80.0,
    int(JointIndex.LEFT_KNEE_PITCH): 80.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 15.0,
    int(JointIndex.LEFT_ANKLE_ROLL): 15.0,
    int(JointIndex.RIGHT_HIP_PITCH): 80.0,
    int(JointIndex.RIGHT_HIP_ROLL): 80.0,
    int(JointIndex.RIGHT_HIP_YAW): 80.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 80.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 15.0,
    int(JointIndex.RIGHT_ANKLE_ROLL): 15.0,
}


WALK_KD = {
    int(JointIndex.LEFT_HIP_PITCH): 4.0,
    int(JointIndex.LEFT_HIP_ROLL): 4.0,
    int(JointIndex.LEFT_HIP_YAW): 4.0,
    int(JointIndex.LEFT_KNEE_PITCH): 4.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 3.0,
    int(JointIndex.LEFT_ANKLE_ROLL): 3.0,
    int(JointIndex.RIGHT_HIP_PITCH): 4.0,
    int(JointIndex.RIGHT_HIP_ROLL): 4.0,
    int(JointIndex.RIGHT_HIP_YAW): 4.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 4.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 3.0,
    int(JointIndex.RIGHT_ANKLE_ROLL): 3.0,
}

ACTION_DIM = 12

# Absolute q limits [lo, hi] per LEG_JOINTS entry — robot commands only.
Q_ABS_LIMITS = np.asarray(
    [
        [-3.000, 2.210],  # Left_Hip_Pitch
        [-0.400, 1.570],  # Left_Hip_Roll
        [-1.000, 1.000],  # Left_Hip_Yaw
        [0.000, 2.230],  # Left_Knee_Pitch
        [-0.870, 0.345],  # Left_Ankle_Pitch
        [-0.345, 0.345],  # Left_Ankle_Roll
        [-3.000, 2.210],  # Right_Hip_Pitch
        [-1.570, 0.400],  # Right_Hip_Roll
        [-1.000, 1.000],  # Right_Hip_Yaw
        [0.000, 2.230],  # Right_Knee_Pitch
        [-0.870, 0.345],  # Right_Ankle_Pitch
        [-0.345, 0.345],  # Right_Ankle_Roll
    ],
    dtype=np.float64,
)
assert Q_ABS_LIMITS.shape == (ACTION_DIM, 2)

DEFAULT_SETTLE_S = 0.5

# Open-loop gait clock (training parity). Override via WalkPolicyV3(gait=...).
# duty_* is used for contact schedule in train only; clock obs ignores it.
DEFAULT_GAIT: Dict[str, float | bool] = {
    "frequency_hz": 2.0,  # unused while adaptive_frequency=True
    "command_threshold": 0.1,
    "adaptive_frequency": True,
    "frequency_hz_min": 1.4,
    "frequency_hz_max": 2.6,
    "vel_ref": 1.5,
    "yaw_weight": 0.5,
    "zero_clock_when_idle": True,
    "stop_ramp_s": 1.0,
}


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



DEFAULT_JOINT_POS = np.asarray(
    [
        0.0,  # Head_Yaw
        0.25,  # Head_Pitch
        0.0, # -0.25,  # Left_Shoulder_Pitch
        -1.45,  # Left_Shoulder_Roll
        0.0,  # Left_Elbow_Pitch
        0.0, # -0.60,  # Left_Elbow_Yaw
        0.0, #-0.25,  # Right_Shoulder_Pitch
        1.45,  # Right_Shoulder_Roll
        0.0,  # Right_Elbow_Pitch
        0.0, # 0.60,  # Right_Elbow_Yaw
        -0.35,  # Left_Hip_Pitch
        -0.04,  # Left_Hip_Roll
        0.0,  # Left_Hip_Yaw
        0.70,  # Left_Knee_Pitch
        -0.35,  # Left_Ankle_Pitch
        0.04,  # Left_Ankle_Roll
        -0.35,  # Right_Hip_Pitch
        0.04,  # Right_Hip_Roll
        0.0,  # Right_Hip_Yaw
        0.70,  # Right_Knee_Pitch
        -0.35,  # Right_Ankle_Pitch
        -0.04,  # Right_Ankle_Roll
    ],
    dtype=np.float64,
)

assert DEFAULT_JOINT_POS.shape == (B1_JOINT_COUNT,)

DEFAULT_LEG_POS = DEFAULT_JOINT_POS[LEG_JOINTS].copy()
assert DEFAULT_LEG_POS.shape == (12,)


def default_pose_action() -> Action:
    """Command every joint to DEFAULT_JOINT_POS (walk gains on legs)."""
    return full_body_action(DEFAULT_LEG_POS)


def full_body_action(leg_q: Sequence[float]) -> Action:
    """
    Full-body command: upper body held at DEFAULT (joint_gains PD),
    legs at `leg_q` (walk PD). `leg_q` is length-12 in LEG_JOINTS order.
    """
    q_leg = np.asarray(leg_q, dtype=np.float64)
    if q_leg.shape != (12,):
        raise ValueError("full_body_action: expected 12 leg qs")

    cmds: List[JointCommand] = [
        make_joint_cmd(int(i), float(DEFAULT_JOINT_POS[i]))
        for i in UPPER_BODY_JOINTS
    ]
    cmds.extend(
        make_walk_joint_cmd(int(joint_index), float(q_leg[i]))
        for i, joint_index in enumerate(LEG_JOINTS)
    )
    return Action(joint_cmds=cmds)


def joint_pos_to_relative(q_abs: Sequence[float]) -> np.ndarray:
    """Absolute joint positions → relative to DEFAULT_JOINT_POS."""
    q = np.asarray(q_abs, dtype=np.float64)
    if q.shape != (B1_JOINT_COUNT,):
        raise ValueError("joint_pos_to_relative: expected 22 joints")
    return q - DEFAULT_JOINT_POS


def action_to_absolute(
    action: Sequence[float], action_scale: float = 1.0
) -> np.ndarray:
    """12-D relative policy action → absolute leg q targets."""
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"action_to_absolute: expected {ACTION_DIM} dims")
    return DEFAULT_LEG_POS + float(action_scale) * a


FRAME_ANG_VEL = 3
FRAME_PROJ_GRAV = 3
FRAME_JOINT_POS = 20
FRAME_JOINT_VEL = 20
FRAME_ACTIONS = 12
FRAME_COMMAND = 3
FRAME_GAIT_CLOCK = 2
FRAME_TERM_SIZES = (
    FRAME_ANG_VEL,
    FRAME_PROJ_GRAV,
    FRAME_JOINT_POS,
    FRAME_JOINT_VEL,
    FRAME_ACTIONS,
    FRAME_COMMAND,
)
FRAME_DIM = sum(FRAME_TERM_SIZES)  # 61 (no gait_clock — appended after history)
assert FRAME_DIM == 61

HISTORY_LEN = 3
HISTORY_INPUT_DIM = FRAME_DIM * HISTORY_LEN  # 183
MODEL_INPUT_DIM = HISTORY_INPUT_DIM + FRAME_GAIT_CLOCK  # 185
assert MODEL_INPUT_DIM == 185

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = _REPO_ROOT / "v3_models" / "2026-08-03_10-06-26_robocup_s2r_v40w.onnx"
DEFAULT_MODEL_PATH = _REPO_ROOT / "v3_models" / "2026-08-03_10-06-26_robocup_s2r_v40w (2).onnx"
DEFAULT_MODEL_PATH = _REPO_ROOT / "v3_models" / "2026-08-04_09-23-09_robocup_s2r_v40z2 (1).onnx"


class WalkPolicyV3(Policy):
    """
    observation_dim = 61 (one history frame, no head, no gait_clock)
    input_dim       = 185 = term-major history (183) + gait_clock (2)
    """

    def __init__(
        self,
        control_dt: float = 0.02,
        model_path: Optional[Union[str, Path]] = None,
        action_scale: float = 1.0,
        history_len: int = HISTORY_LEN,
        load_default_model: bool = True,
        recorder: Optional[ModelInputRecorder] = None,
        settle_s: float = DEFAULT_SETTLE_S,
        gait: Optional[Dict[str, float | bool]] = None,
    ) -> None:
        self._control_dt = float(control_dt)
        self._action_scale = float(action_scale)
        self._settle_s = float(settle_s)
        self._gait: Dict[str, float | bool] = dict(DEFAULT_GAIT)
        if gait:
            self._gait.update(gait)
        self._history = ObservationHistory(
            FRAME_DIM,
            history_len,
            layout="term",
            term_sizes=FRAME_TERM_SIZES,
        )
        self._last_action: List[float] = [0.0] * ACTION_DIM
        self._cmd: List[float] = [0.0, 0.0, 0.0]
        self._gait_phase = 0.0
        self._recorder = recorder
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

    def set_recorder(self, recorder: Optional[ModelInputRecorder]) -> None:
        self._recorder = recorder

    def recorder(self) -> Optional[ModelInputRecorder]:
        return self._recorder

    def name(self) -> str:
        return "walk_v3"

    def observation_dim(self) -> int:
        return FRAME_DIM

    def history_len(self) -> int:
        return self._history.history_len

    def input_dim(self) -> int:
        return self._history.input_dim + FRAME_GAIT_CLOCK

    def controlled_joints(self) -> List[int]:
        return list(range(B1_JOINT_COUNT))

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"walk_v3: model not found: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"walk_v3: ONNX has no inputs/outputs: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def _gait_clock(self) -> List[float]:
        _GAIT_FREQ_EPS = 1.0e-3
        """Open-loop [cos φ, sin φ]; optional linear stop ramp (train parity)."""
        vx, vy, wz = self._cmd
        speed = math.hypot(vx, vy) + float(self._gait["yaw_weight"]) * abs(wz)
        moving = speed > float(self._gait["command_threshold"])
        if self._gait["adaptive_frequency"]:
            alpha = min(1.0, max(0.0, speed / max(float(self._gait["vel_ref"]), 1e-6)))
            target = float(self._gait["frequency_hz_min"]) + (
                float(self._gait["frequency_hz_max"])
                - float(self._gait["frequency_hz_min"])
            ) * alpha
        else:
            target = float(self._gait["frequency_hz"])

        dt = self._control_dt
        T = float(self._gait["stop_ramp_s"])

        if T <= 0.0:
            # Legacy: snap map; freeze phase when idle; optional zero clock.
            if moving:
                self._gait_freq = target
                self._gait_phase = (
                    self._gait_phase + 2.0 * math.pi * self._gait_freq * dt
                ) % (2.0 * math.pi)
                cos_s = math.cos(self._gait_phase)
                sin_s = math.sin(self._gait_phase)
            else:
                cos_s = math.cos(self._gait_phase)
                sin_s = math.sin(self._gait_phase)
                if self._gait["zero_clock_when_idle"]:
                    cos_s = 0.0
                    sin_s = 0.0
            self._gait_was_moving = moving
            return [cos_s, sin_s]

        if moving:
            self._gait_freq = target
            self._gait_stop_elapsed = 0.0
        else:
            if self._gait_was_moving:
                self._gait_stop_f0 = max(self._gait_freq, _GAIT_FREQ_EPS)
                self._gait_stop_elapsed = 0.0
            self._gait_stop_elapsed += dt
            frac = max(0.0, min(1.0, 1.0 - self._gait_stop_elapsed / T))
            self._gait_freq = self._gait_stop_f0 * frac

        self._gait_was_moving = moving
        if self._gait_freq > _GAIT_FREQ_EPS:
            self._gait_phase = (
                self._gait_phase + 2.0 * math.pi * self._gait_freq * dt
            ) % (2.0 * math.pi)
            cos_s = math.cos(self._gait_phase)
            sin_s = math.sin(self._gait_phase)
        else:
            cos_s = math.cos(self._gait_phase)
            sin_s = math.sin(self._gait_phase)
            if self._gait["zero_clock_when_idle"]:
                cos_s = 0.0
                sin_s = 0.0
        return [cos_s, sin_s]

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        if len(state.q) != B1_JOINT_COUNT or len(state.dq) != B1_JOINT_COUNT:
            raise ValueError("walk_v3: RobotState q/dq size mismatch")

        cmd = [float(x) for x in command[:FRAME_COMMAND]]
        while len(cmd) < FRAME_COMMAND:
            cmd.append(0.0)
        self._cmd = cmd

        joint_pos_all = joint_pos_to_relative(state.q)
        joint_pos_rel = joint_pos_all[OBS_BODY_JOINTS]
        joint_vel = np.asarray(state.dq, dtype=np.float64)[OBS_BODY_JOINTS]

        gyro = np.zeros(FRAME_ANG_VEL, dtype=np.float64)
        gyro[: min(FRAME_ANG_VEL, len(state.imu.gyro))] = state.imu.gyro[
            :FRAME_ANG_VEL
        ]
        grav = np.zeros(FRAME_PROJ_GRAV, dtype=np.float64)
        grav[: min(FRAME_PROJ_GRAV, len(state.projected_gravity))] = (
            state.projected_gravity[:FRAME_PROJ_GRAV]
        )
        last_a = np.asarray(self._last_action, dtype=np.float64)
        cmd_np = np.zeros(FRAME_COMMAND, dtype=np.float64)
        cmd_np[: len(cmd)] = cmd

        data = np.concatenate(
            [gyro, grav, joint_pos_rel, joint_vel, last_a, cmd_np]
        )
        if data.shape != (FRAME_DIM,):
            raise ValueError(
                f"walk_v3: built frame dim {data.size} != FRAME_DIM {FRAME_DIM}"
            )
        return Observation(data=data.tolist())

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        now = time.perf_counter()
        if self._settle_t0 is None:
            self._settle_t0 = now
            print(
                f"[walk_v3] settling to DEFAULT_JOINT_POS for "
                f"{self._settle_s:.1f}s..."
            )

        if (now - self._settle_t0) < self._settle_s:
            return default_pose_action()

        if not self._rl_started:
            self._history.reset()
            self._last_action = [0.0] * ACTION_DIM
            self._gait_phase = 0.0
            self._rl_started = True
            print("[walk_v3] settle done — starting RL walk")

        stacked = self._history.push(obs.data)
        gait = self._gait_clock()
        model_input = stacked + gait
        if len(model_input) != self.input_dim():
            raise ValueError(
                f"walk_v3: stacked dim {len(model_input)} != "
                f"input_dim {self.input_dim()}"
            )

        if self._recorder is not None:
            self._recorder.record(model_input)

        if self._session is None:
            raise RuntimeError("walk_v3: model not loaded; call load_model()")

        x = np.asarray(model_input, dtype=np.float32).reshape(1, -1)
        y = self._session.run(
            [self._output_name], {self._input_name: x}
        )[0]
        action = np.asarray(y, dtype=np.float64).reshape(-1)

        if len(action) != ACTION_DIM:
            raise ValueError(
                f"walk_v3: action dim {len(action)} != ACTION_DIM {ACTION_DIM}"
            )

        # Model emits relative leg targets; convert + clamp for the robot.
        q_abs = action_to_absolute(action, self._action_scale)
        q_abs = np.clip(q_abs, Q_ABS_LIMITS[:, 0], Q_ABS_LIMITS[:, 1])
        scale = self._action_scale if self._action_scale != 0.0 else 1.0
        self._last_action = [
            float(a) for a in (q_abs - DEFAULT_LEG_POS) / scale
        ]
        return full_body_action(q_abs)

    def reset(self) -> None:
        self._history.reset()
        self._last_action = [0.0] * ACTION_DIM
        self._cmd = [0.0, 0.0, 0.0]
        self._gait_phase = 0.0
        self._gait_freq = 0.0
        self._gait_was_moving = False
        self._gait_stop_f0 = 0.0
        self._gait_stop_elapsed = 0.0
        self._settle_t0 = None
        self._rl_started = False
