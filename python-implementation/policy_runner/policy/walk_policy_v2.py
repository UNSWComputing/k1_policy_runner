"""Walk policy v2 — 61-dim frame obs × history 3 → 12-dim leg action.

Frame layout (61):
  base_ang_vel (3)       IMU gyro
  projected_gravity (3)
  joint_pos (20)         arms+legs, q relative to DEFAULT (no head)
  joint_vel (20)         same joints
  actions (12)           last commanded action (clamped, relative)
  command (3)            twist (vx, vy, ωz)

Startup: hold all joints at DEFAULT_JOINT_POS for settle_s, then RL walk.
During RL, legs follow the policy; upper body is held at DEFAULT with
joint_gains PD.

Default model: <repo>/v2_models/model_31150.onnx
  input  "obs"     → [1, 183]  term-major: per term group, [t-2|t-1|t]
  output "actions" → [1, 12]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

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

WALK_KP = {
    int(JointIndex.LEFT_HIP_PITCH): 80.0,
    int(JointIndex.LEFT_HIP_ROLL): 80.0,
    int(JointIndex.LEFT_HIP_YAW): 80.0,
    int(JointIndex.LEFT_KNEE_PITCH): 80.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 20.0,
    int(JointIndex.LEFT_ANKLE_ROLL): 20.0,
    int(JointIndex.RIGHT_HIP_PITCH): 80.0,
    int(JointIndex.RIGHT_HIP_ROLL): 80.0,
    int(JointIndex.RIGHT_HIP_YAW): 80.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 80.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 20.0,
    int(JointIndex.RIGHT_ANKLE_ROLL): 20.0,
}
WALK_KD = {
    int(JointIndex.LEFT_HIP_PITCH): 4.0,
    int(JointIndex.LEFT_HIP_ROLL): 4.0,
    int(JointIndex.LEFT_HIP_YAW): 4.0,
    int(JointIndex.LEFT_KNEE_PITCH): 4.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 2.5,
    int(JointIndex.LEFT_ANKLE_ROLL): 2.5,
    int(JointIndex.RIGHT_HIP_PITCH): 4.0,
    int(JointIndex.RIGHT_HIP_ROLL): 4.0,
    int(JointIndex.RIGHT_HIP_YAW): 4.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 4.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 2.5,
    int(JointIndex.RIGHT_ANKLE_ROLL): 2.5,
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
        0.0,  # Head_Pitch
        0.3,  # Left_Shoulder_Pitch
        -1.65,  # Left_Shoulder_Roll
        2.0,  # Left_Elbow_Pitch
        -0.45,  # Left_Elbow_Yaw
        0.3,  # Right_Shoulder_Pitch
        1.65,  # Right_Shoulder_Roll
        2.0,  # Right_Elbow_Pitch
        0.45,  # Right_Elbow_Yaw
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



# ARM on the side
DEFAULT_JOINT_POS = np.asarray(
    [
            0.0,
    0.0,
    0.0,
    -1.452,
    0.009,
    -0.501,
    0.0,
    1.455,
    -0.026,
    0.497,
    -0.001,
    0.002,
    0.004,
    0.109,
    -0.085,
    -0.003,
    -0.000,
    -0.001,
    -0.001,
    0.109,
    -0.090,
    0.001,
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
    """12-D policy action → absolute leg q targets (scale + default offset)."""
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
FRAME_TERM_SIZES = (
    FRAME_ANG_VEL,
    FRAME_PROJ_GRAV,
    FRAME_JOINT_POS,
    FRAME_JOINT_VEL,
    FRAME_ACTIONS,
    FRAME_COMMAND,
)
FRAME_DIM = sum(FRAME_TERM_SIZES)  # 61
assert FRAME_DIM == 61

HISTORY_LEN = 3
MODEL_INPUT_DIM = FRAME_DIM * HISTORY_LEN  # 183

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_PATH = _REPO_ROOT / "v2_models" / "robocup_s2r_v40p.onnx"

class WalkPolicyV2(Policy):
    """
    observation_dim = 61 (one frame, no head)
    input_dim       = 183, term-group major (per term: [t-2|t-1|t])
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
    ) -> None:
        del control_dt
        self._action_scale = float(action_scale)
        self._settle_s = float(settle_s)
        self._history = ObservationHistory(
            FRAME_DIM,
            history_len,
            layout="term",
            term_sizes=FRAME_TERM_SIZES,
        )
        self._last_action: List[float] = [0.0] * ACTION_DIM
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
        return "walk_v2"

    def observation_dim(self) -> int:
        return FRAME_DIM

    def history_len(self) -> int:
        return self._history.history_len

    def input_dim(self) -> int:
        return self._history.input_dim

    def controlled_joints(self) -> List[int]:
        return list(range(B1_JOINT_COUNT))

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"walk_v2: model not found: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"walk_v2: ONNX has no inputs/outputs: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        if len(state.q) != B1_JOINT_COUNT or len(state.dq) != B1_JOINT_COUNT:
            raise ValueError("walk_v2: RobotState q/dq size mismatch")

        cmd = [float(x) for x in command[:FRAME_COMMAND]]
        while len(cmd) < FRAME_COMMAND:
            cmd.append(0.0)

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
                f"walk_v2: built frame dim {data.size} != FRAME_DIM {FRAME_DIM}"
            )
        return Observation(data=data.tolist())

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        now = time.perf_counter()
        if self._settle_t0 is None:
            self._settle_t0 = now
            print(
                f"[walk_v2] settling to DEFAULT_JOINT_POS for "
                f"{self._settle_s:.1f}s..."
            )

        if (now - self._settle_t0) < self._settle_s:
            return default_pose_action()

        if not self._rl_started:
            self._history.reset()
            self._last_action = [0.0] * ACTION_DIM
            self._rl_started = True
            print("[walk_v2] settle done — starting RL walk")

        model_input = self._history.push(obs.data)
        if len(model_input) != self.input_dim():
            raise ValueError(
                f"walk_v2: stacked dim {len(model_input)} != "
                f"input_dim {self.input_dim()}"
            )

        if self._recorder is not None:
            self._recorder.record(model_input)

        if self._session is None:
            raise RuntimeError("walk_v2: model not loaded; call load_model()")

        x = np.asarray(model_input, dtype=np.float32).reshape(1, -1)
        y = self._session.run(
            [self._output_name], {self._input_name: x}
        )[0]
        action = np.asarray(y, dtype=np.float64).reshape(-1)

        if len(action) != ACTION_DIM:
            raise ValueError(
                f"walk_v2: action dim {len(action)} != ACTION_DIM {ACTION_DIM}"
            )

        q_abs = action_to_absolute(action, self._action_scale)
        q_abs = np.clip(q_abs, Q_ABS_LIMITS[:, 0], Q_ABS_LIMITS[:, 1])
        scale = self._action_scale if self._action_scale != 0.0 else 1.0
        self._last_action = [
            float(a) for a in (q_abs - DEFAULT_LEG_POS) / scale
        ]
        print(
            "[walk_v2] ankle targets (rad): "
            f"L_pitch={q_abs[4]:.4f} L_roll={q_abs[5]:.4f} "
            f"R_pitch={q_abs[10]:.4f} R_roll={q_abs[11]:.4f}"
        )
        return full_body_action(q_abs)

    def reset(self) -> None:
        self._history.reset()
        self._last_action = [0.0] * ACTION_DIM
        self._settle_t0 = None
        self._rl_started = False
