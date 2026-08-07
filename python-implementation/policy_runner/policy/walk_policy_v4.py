"""Walk policy v4 — 75-dim single-frame obs → 20-dim arm+leg action.

Frame layout (75):
  base_ang_vel (3)       IMU gyro (body frame)
  projected_gravity (3)
  joint_pos (22)         q − DEFAULT for all joints
  joint_vel (22)         dq (same order)
  actions (20)           last policy output (raw, before scale); arms+legs
  command (3)            [vx, vy, wz]
  gait_cycle (2)         [sin(2πφ), cos(2πφ)]

Gait: φ ← (φ + dt·f) mod 1; f fixed while walking; if still
(‖vx,vy‖+|wz| < 0.05) then f=0 and gait_cycle=[0,0].

Action: 20-D relative (arms 8 + legs 12, no head).
  q_abs = DEFAULT + scale * a_raw
  scale = 0.15 (arms), 1.0 (legs)

Startup: hold DEFAULT_JOINT_POS for settle_s, then RL.
Head held at DEFAULT during RL; arms+legs from policy.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from policy_runner.joint_gains import make_joint_cmd
from policy_runner.joint_index import JointIndex
from policy_runner.obs_record import ModelInputRecorder
from policy_runner.policy.base import Policy
from policy_runner.types import (
    B1_JOINT_COUNT,
    Action,
    JointCommand,
    Observation,
    RobotState,
)

# ---------------------------------------------------------------------------
# Joint groups
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

# Arms + legs (no head) — model action order.
ACTION_JOINTS = np.arange(
    int(JointIndex.LEFT_SHOULDER_PITCH), B1_JOINT_COUNT, dtype=np.int64
)
assert ACTION_JOINTS.shape == (20,)

HEAD_JOINTS = np.arange(0, int(JointIndex.LEFT_SHOULDER_PITCH), dtype=np.int64)
assert HEAD_JOINTS.shape == (2,)

# Per-action scale: arms 0.15, legs 1.0 (same order as ACTION_JOINTS).
ACTION_SCALE = np.asarray(
    [0.15] * 8 + [1.0] * 12,
    dtype=np.float64,
)
assert ACTION_SCALE.shape == (20,)

WALK_KP = {
    int(JointIndex.LEFT_HIP_PITCH): 80.0,
    int(JointIndex.LEFT_HIP_ROLL): 80.0,
    int(JointIndex.LEFT_HIP_YAW): 80.0,
    int(JointIndex.LEFT_KNEE_PITCH): 80.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 25.0,
    int(JointIndex.LEFT_ANKLE_ROLL): 25.0,
    int(JointIndex.RIGHT_HIP_PITCH): 80.0,
    int(JointIndex.RIGHT_HIP_ROLL): 80.0,
    int(JointIndex.RIGHT_HIP_YAW): 80.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 80.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 25.0,
    int(JointIndex.RIGHT_ANKLE_ROLL): 25.0,
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

ACTION_DIM = 20
ACTION_CLIP = 1.0  # training clip_actions

DEFAULT_SETTLE_S = 0.2

# Gait cycle (φ ∈ [0,1)). Still → f=0 and obs zeros.
GAIT_STILL_THRESHOLD = 0.05
GAIT_FREQUENCY_HZ = 1.5  # in [1.0, 2.0] while walking


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
        0.0,  # Left_Shoulder_Pitch
        -1.45,  # Left_Shoulder_Roll
        0.0,  # Left_Elbow_Pitch
        0.0,  # Left_Elbow_Yaw
        0.0,  # Right_Shoulder_Pitch
        1.45,  # Right_Shoulder_Roll
        0.0,  # Right_Elbow_Pitch
        0.0,  # Right_Elbow_Yaw
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

DEFAULT_ACTION_POS = DEFAULT_JOINT_POS[ACTION_JOINTS].copy()
assert DEFAULT_ACTION_POS.shape == (ACTION_DIM,)


def default_pose_action() -> Action:
    """Command every joint to DEFAULT_JOINT_POS."""
    return full_body_action(DEFAULT_ACTION_POS)


def full_body_action(action_q: Sequence[float]) -> Action:
    """
    Head held at DEFAULT; arms+legs at `action_q` (length 20, ACTION_JOINTS order).
    Legs use walk PD; head/arms use default joint_gains PD.
    """
    q_act = np.asarray(action_q, dtype=np.float64)
    if q_act.shape != (ACTION_DIM,):
        raise ValueError("full_body_action: expected 20 action qs")

    cmds: List[JointCommand] = [
        make_joint_cmd(int(i), float(DEFAULT_JOINT_POS[i])) for i in HEAD_JOINTS
    ]
    for i, joint_index in enumerate(ACTION_JOINTS):
        ji = int(joint_index)
        q = float(q_act[i])
        if ji in WALK_KP:
            cmds.append(make_walk_joint_cmd(ji, q))
        else:
            cmds.append(make_joint_cmd(ji, q))
    return Action(joint_cmds=cmds)


def joint_pos_to_relative(q_abs: Sequence[float]) -> np.ndarray:
    q = np.asarray(q_abs, dtype=np.float64)
    if q.shape != (B1_JOINT_COUNT,):
        raise ValueError("joint_pos_to_relative: expected 22 joints")
    return q - DEFAULT_JOINT_POS


def action_to_absolute(action: Sequence[float]) -> np.ndarray:
    """20-D raw relative action → absolute arm+leg q (per-joint scale)."""
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"action_to_absolute: expected {ACTION_DIM} dims")
    return DEFAULT_ACTION_POS + ACTION_SCALE * a


FRAME_ANG_VEL = 3
FRAME_PROJ_GRAV = 3
FRAME_JOINT_POS = 22
FRAME_JOINT_VEL = 22
FRAME_ACTIONS = 20
FRAME_COMMAND = 3
FRAME_GAIT_CYCLE = 2
FRAME_DIM = (
    FRAME_ANG_VEL
    + FRAME_PROJ_GRAV
    + FRAME_JOINT_POS
    + FRAME_JOINT_VEL
    + FRAME_ACTIONS
    + FRAME_COMMAND
    + FRAME_GAIT_CYCLE
)
assert FRAME_DIM == 75

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = _REPO_ROOT / "v4_models" / "model_31000 (1).onnx"


class WalkPolicyV4(Policy):
    """observation_dim = input_dim = 75 (single frame, no history)."""

    def __init__(
        self,
        control_dt: float = 0.02,
        model_path: Optional[Union[str, Path]] = None,
        history_len: int = 1,
        load_default_model: bool = True,
        recorder: Optional[ModelInputRecorder] = None,
        settle_s: float = DEFAULT_SETTLE_S,
        gait_frequency_hz: float = GAIT_FREQUENCY_HZ,
    ) -> None:
        del history_len  # v4 is single-frame
        self._control_dt = float(control_dt)
        self._settle_s = float(settle_s)
        self._gait_frequency_hz = float(gait_frequency_hz)
        self._last_action: List[float] = [0.0] * ACTION_DIM
        self._cmd: List[float] = [0.0, 0.0, 0.0]
        self._gait_phase = 0.0  # φ ∈ [0, 1)
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
        return "walk_v4"

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
            raise FileNotFoundError(f"walk_v4: model not found: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"walk_v4: ONNX has no inputs/outputs: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def _gait_cycle(self) -> List[float]:
        """[sin(2πφ), cos(2πφ)]; φ += dt·f; still → f=0 and [0, 0]."""
        if not self._rl_started:
            return [0.0, 0.0]
        vx, vy, wz = self._cmd
        still = (math.hypot(vx, vy) + abs(wz)) < GAIT_STILL_THRESHOLD
        if still:
            return [0.0, 0.0]
        f = self._gait_frequency_hz
        self._gait_phase = (self._gait_phase + self._control_dt * f) % 1.0
        ang = 2.0 * math.pi * self._gait_phase
        return [math.sin(ang), math.cos(ang)]

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        if len(state.q) != B1_JOINT_COUNT or len(state.dq) != B1_JOINT_COUNT:
            raise ValueError("walk_v4: RobotState q/dq size mismatch")

        cmd = [float(x) for x in command[:FRAME_COMMAND]]
        while len(cmd) < FRAME_COMMAND:
            cmd.append(0.0)
        self._cmd = cmd

        joint_pos_rel = joint_pos_to_relative(state.q)
        joint_vel = np.asarray(state.dq, dtype=np.float64)

        gyro = np.zeros(FRAME_ANG_VEL, dtype=np.float64)
        gyro[: min(FRAME_ANG_VEL, len(state.imu.gyro))] = state.imu.gyro[
            :FRAME_ANG_VEL
        ]
        grav = np.zeros(FRAME_PROJ_GRAV, dtype=np.float64)
        grav[: min(FRAME_PROJ_GRAV, len(state.projected_gravity))] = (
            state.projected_gravity[:FRAME_PROJ_GRAV]
        )
        last_a = np.asarray(self._last_action, dtype=np.float64)
        cmd_np = np.asarray(cmd, dtype=np.float64)
        gait = np.asarray(self._gait_cycle(), dtype=np.float64)

        data = np.concatenate(
            [gyro, grav, joint_pos_rel, joint_vel, last_a, cmd_np, gait]
        )
        if data.shape != (FRAME_DIM,):
            raise ValueError(
                f"walk_v4: built frame dim {data.size} != FRAME_DIM {FRAME_DIM}"
            )
        return Observation(data=data.tolist())

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        now = time.perf_counter()
        if self._settle_t0 is None:
            self._settle_t0 = now
            print(
                f"[walk_v4] settling to DEFAULT_JOINT_POS for "
                f"{self._settle_s:.1f}s..."
            )

        if (now - self._settle_t0) < self._settle_s:
            return default_pose_action()

        if not self._rl_started:
            self._last_action = [0.0] * ACTION_DIM
            self._gait_phase = 0.0
            self._rl_started = True
            print("[walk_v4] settle done — starting RL walk")

        model_input = list(obs.data)
        if len(model_input) != self.input_dim():
            raise ValueError(
                f"walk_v4: input dim {len(model_input)} != "
                f"input_dim {self.input_dim()}"
            )

        if self._recorder is not None:
            self._recorder.record(model_input)

        if self._session is None:
            raise RuntimeError("walk_v4: model not loaded; call load_model()")

        x = np.asarray(model_input, dtype=np.float32).reshape(1, -1)
        y = self._session.run(
            [self._output_name], {self._input_name: x}
        )[0]
        action = np.asarray(y, dtype=np.float64).reshape(-1)

        if len(action) != ACTION_DIM:
            raise ValueError(
                f"walk_v4: action dim {len(action)} != ACTION_DIM {ACTION_DIM}"
            )

        # Train parity: clip raw action to ±1, feedback that, then scale.
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        self._last_action = [float(a) for a in action]
        q_abs = action_to_absolute(action)
        return full_body_action(q_abs)

    def reset(self) -> None:
        self._last_action = [0.0] * ACTION_DIM
        self._cmd = [0.0, 0.0, 0.0]
        self._gait_phase = 0.0
        self._settle_t0 = None
        self._rl_started = False
