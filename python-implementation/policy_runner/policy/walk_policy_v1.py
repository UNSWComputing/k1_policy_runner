"""Walk policy v1 — 65-dim frame obs × history 3 → 12-dim lower-body action.

Frame layout (65):
  base_ang_vel (3)       IMU gyro
  projected_gravity (3)
  joint_pos (22)         q relative to DEFAULT (offset layer)
  joint_vel (22)
  actions (12)           last policy action (12-D emit), no scale/offset
  command (3)            twist (vx, vy, ωz)

Model I/O is relative. Absolute robot commands via offset layer only:
  q_abs = DEFAULT + scale * a_rel

Model: ONNX (architecture + normalizer baked in). Default:
  <repo>/k1_v24_model_100800.onnx
  input  "obs"     → [1, 195]  term-major: per term group, [t-2|t-1|t]
  output "actions" → [1, 12]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from policy_runner.joint_index import JointIndex
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
# Offset layer — default / HOME pose (absolute rad), stored as numpy.
# Used only when converting action → robot q targets (not for last_action obs).
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

# Walk PD gains (legs only). Effort limits noted for reference / future use.
# Hip pitch/roll/yaw: kp=80 kd=4  effort 45/20/20
# Knee:               kp=80 kd=4  effort 40
# Ankle pitch/roll:   kp=25 kd=1  effort 20/15
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
    int(JointIndex.LEFT_ANKLE_PITCH): 1.0,
    int(JointIndex.LEFT_ANKLE_ROLL): 1.0,
    int(JointIndex.RIGHT_HIP_PITCH): 4.0,
    int(JointIndex.RIGHT_HIP_ROLL): 4.0,
    int(JointIndex.RIGHT_HIP_YAW): 4.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 4.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 1.0,
    int(JointIndex.RIGHT_ANKLE_ROLL): 1.0,
}
WALK_EFFORT_LIMIT = {
    int(JointIndex.LEFT_HIP_PITCH): 45.0,
    int(JointIndex.LEFT_HIP_ROLL): 20.0,
    int(JointIndex.LEFT_HIP_YAW): 20.0,
    int(JointIndex.LEFT_KNEE_PITCH): 40.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 20.0,
    int(JointIndex.LEFT_ANKLE_ROLL): 15.0,
    int(JointIndex.RIGHT_HIP_PITCH): 45.0,
    int(JointIndex.RIGHT_HIP_ROLL): 20.0,
    int(JointIndex.RIGHT_HIP_YAW): 20.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 40.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 20.0,
    int(JointIndex.RIGHT_ANKLE_ROLL): 15.0,
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

# Full 22-DoF default pose (JointIndex order). Obs joint_pos = q - this.
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
assert DEFAULT_JOINT_POS.shape == (B1_JOINT_COUNT,)

# Legs only — used for action → absolute q (walk controls 12 DoF).
DEFAULT_LEG_POS = DEFAULT_JOINT_POS[LEG_JOINTS].copy()
assert DEFAULT_LEG_POS.shape == (12,)

FRAME_ANG_VEL = 3
FRAME_PROJ_GRAV = 3
FRAME_JOINT_POS = 22
FRAME_JOINT_VEL = 22
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
FRAME_DIM = sum(FRAME_TERM_SIZES)  # 65
assert FRAME_DIM == 65

HISTORY_LEN = 3
MODEL_INPUT_DIM = FRAME_DIM * HISTORY_LEN  # 195
ACTION_DIM = 12

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = _REPO_ROOT / "k1_v24_model_105600.onnx"


def joint_pos_to_relative(q_abs: Sequence[float]) -> np.ndarray:
    """Absolute joint positions → relative to DEFAULT_JOINT_POS (model input)."""
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


class WalkPolicyV1(Policy):
    """
    observation_dim = 65 (one frame)
    input_dim       = 195, term-group major (per term: [t-2|t-1|t])
    last action     = self._last_action: same 12-D policy emit (no scale/offset)
    """

    def __init__(
        self,
        control_dt: float = 0.02,
        model_path: Optional[Union[str, Path]] = None,
        action_scale: float = 1.0,
        history_len: int = HISTORY_LEN,
        load_default_model: bool = True,
    ) -> None:
        del control_dt
        self._action_scale = float(action_scale)
        self._history = ObservationHistory(
            FRAME_DIM,
            history_len,
            layout="term",
            term_sizes=FRAME_TERM_SIZES,
        )
        # Previous policy output (12-D), as emitted — not q targets.
        self._last_action: List[float] = [0.0] * ACTION_DIM

        self._session: Optional[ort.InferenceSession] = None
        self._input_name = "obs"
        self._output_name = "actions"

        path = model_path
        if path is None and load_default_model:
            path = DEFAULT_MODEL_PATH
        if path is not None:
            self.load_model(str(path))

    def name(self) -> str:
        return "walk_v1"

    def observation_dim(self) -> int:
        return FRAME_DIM

    def history_len(self) -> int:
        return self._history.history_len

    def input_dim(self) -> int:
        return self._history.input_dim

    def controlled_joints(self) -> List[int]:
        return LEG_JOINTS.tolist()

    def load_model(self, model_path: str) -> None:
        """Load ONNX model (graph includes normalizer + MLP)."""
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"walk_v1: model not found: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"walk_v1: ONNX has no inputs/outputs: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        if len(state.q) != B1_JOINT_COUNT or len(state.dq) != B1_JOINT_COUNT:
            raise ValueError("walk_v1: RobotState q/dq size mismatch")

        cmd = [float(x) for x in command[:FRAME_COMMAND]]
        while len(cmd) < FRAME_COMMAND:
            cmd.append(0.0)

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
        # Same 12-D policy vector as last emit (no scale / default offset).
        last_a = np.asarray(self._last_action, dtype=np.float64)
        cmd_np = np.zeros(FRAME_COMMAND, dtype=np.float64)
        cmd_np[: len(cmd)] = cmd

        data = np.concatenate(
            [gyro, grav, joint_pos_rel, joint_vel, last_a, cmd_np]
        )
        if data.shape != (FRAME_DIM,):
            raise ValueError(
                f"walk_v1: built frame dim {data.size} != FRAME_DIM {FRAME_DIM}"
            )
        return Observation(data=data.tolist())

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        model_input = self._history.push(obs.data)
        if len(model_input) != self.input_dim():
            raise ValueError(
                f"walk_v1: stacked dim {len(model_input)} != "
                f"input_dim {self.input_dim()}"
            )

        if self._session is None:
            raise RuntimeError("walk_v1: model not loaded; call load_model()")

        x = np.asarray(model_input, dtype=np.float32).reshape(1, -1)
        y = self._session.run(
            [self._output_name], {self._input_name: x}
        )[0]
        action = np.asarray(y, dtype=np.float64).reshape(-1)

        if len(action) != ACTION_DIM:
            raise ValueError(
                f"walk_v1: action dim {len(action)} != ACTION_DIM {ACTION_DIM}"
            )

        # Feedback the emitted 12-D action into the next frame (no scale/offset).
        self._last_action = [float(a) for a in action]
        q_abs = action_to_absolute(action, self._action_scale)
        print(
            "[walk_v1] ankle targets (rad): "
            f"L_pitch={q_abs[4]:.4f} L_roll={q_abs[5]:.4f} "
            f"R_pitch={q_abs[10]:.4f} R_roll={q_abs[11]:.4f}"
        )
        return Action(
            joint_cmds=[
                make_walk_joint_cmd(int(joint_index), float(q_abs[i]))
                for i, joint_index in enumerate(LEG_JOINTS)
            ]
        )

    def reset(self) -> None:
        self._history.reset()
        self._last_action = [0.0] * ACTION_DIM
