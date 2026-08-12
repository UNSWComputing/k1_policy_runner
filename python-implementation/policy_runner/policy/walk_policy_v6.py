"""Walk policy v6 — mjlab Minimal K1, 73-dim obs → 20-dim action.

Same as walk_v5 without height_scan. Frame layout (73):
  base_ang_vel (3) + projected_gravity (3) + joint_pos (22) + joint_vel (22)
  + actions (20) + command (3)

default_joint_pos and action_scale from ONNX metadata.
Leg PD uses WALK_KP/KD (metadata joint_stiffness/damping is read then
overridden for walk joints). Arms use default joint_gains PD.
Head not policy-controlled (sparse 20-joint action).

Sent q is clipped to physical joint ranges (k1_22dof_scene.xml). last_action
is inverted from that clipped q, so an out-of-range raw action is not fed back.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from policy_runner.joint_gains import make_joint_cmd
from policy_runner.joint_index import JointIndex
from policy_runner.policy.base import Policy
from policy_runner.types import (
    B1_JOINT_COUNT,
    Action,
    JointCommand,
    Observation,
    RobotState,
)

ACTION_JOINTS = np.arange(
    int(JointIndex.LEFT_SHOULDER_PITCH), B1_JOINT_COUNT, dtype=np.int64
)
assert ACTION_JOINTS.shape == (20,)

HEAD_JOINTS = np.arange(0, int(JointIndex.LEFT_SHOULDER_PITCH), dtype=np.int64)
assert HEAD_JOINTS.shape == (2,)

ACTION_DIM = 20

# Physical q limits [lo, hi] in JointIndex order (assets/k1_22dof_scene.xml).
Q_ABS_LIMITS = np.asarray(
    [
        [-1.000, 1.000],  # Head_Yaw
        [-0.349, 0.855],  # Head_Pitch
        [-3.316, 1.220],  # Left_Shoulder_Pitch
        [-1.740, 1.570],  # Left_Shoulder_Roll
        [-2.270, 2.270],  # Left_Elbow_Pitch
        [-2.440, 0.000],  # Left_Elbow_Yaw
        [-3.316, 1.220],  # Right_Shoulder_Pitch
        [-1.570, 1.740],  # Right_Shoulder_Roll
        [-2.270, 2.270],  # Right_Elbow_Pitch
        [0.000, 2.440],  # Right_Elbow_Yaw
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
assert Q_ABS_LIMITS.shape == (B1_JOINT_COUNT, 2)

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
    int(JointIndex.LEFT_ANKLE_PITCH): 2.5,
    int(JointIndex.LEFT_ANKLE_ROLL): 2.5,
    int(JointIndex.RIGHT_HIP_PITCH): 4.0,
    int(JointIndex.RIGHT_HIP_ROLL): 4.0,
    int(JointIndex.RIGHT_HIP_YAW): 4.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 4.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 2.5,
    int(JointIndex.RIGHT_ANKLE_ROLL): 2.5,
}

# WEAKER
WALK_KP = {
    int(JointIndex.LEFT_HIP_PITCH): 44.0,
    int(JointIndex.LEFT_HIP_ROLL): 44.0,
    int(JointIndex.LEFT_HIP_YAW): 44.0,
    int(JointIndex.LEFT_KNEE_PITCH): 44.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 15.0,
    int(JointIndex.LEFT_ANKLE_ROLL): 15.0,
    int(JointIndex.RIGHT_HIP_PITCH): 44.0,
    int(JointIndex.RIGHT_HIP_ROLL): 44.0,
    int(JointIndex.RIGHT_HIP_YAW): 44.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 44.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 15.0,
    int(JointIndex.RIGHT_ANKLE_ROLL): 15.0,
}

WALK_KD = {
    int(JointIndex.LEFT_HIP_PITCH): 3.0,
    int(JointIndex.LEFT_HIP_ROLL): 4.0,
    int(JointIndex.LEFT_HIP_YAW): 4.0,
    int(JointIndex.LEFT_KNEE_PITCH): 4.0,
    int(JointIndex.LEFT_ANKLE_PITCH): 2.5,
    int(JointIndex.LEFT_ANKLE_ROLL): 2.5,
    int(JointIndex.RIGHT_HIP_PITCH): 3.0,
    int(JointIndex.RIGHT_HIP_ROLL): 4.0,
    int(JointIndex.RIGHT_HIP_YAW): 4.0,
    int(JointIndex.RIGHT_KNEE_PITCH): 4.0,
    int(JointIndex.RIGHT_ANKLE_PITCH): 2.5,
    int(JointIndex.RIGHT_ANKLE_ROLL): 2.5,
}

# +/-15 deg arm target clip (env_cfgs_minimal _ARM_CLIP).
_ARM_BAND = math.radians(15.0)
_ARM_CLIP_BY_INDEX: Dict[int, tuple[float, float]] = {
    int(JointIndex.LEFT_SHOULDER_PITCH): (-_ARM_BAND, _ARM_BAND),
    int(JointIndex.RIGHT_SHOULDER_PITCH): (-_ARM_BAND, _ARM_BAND),
    int(JointIndex.LEFT_SHOULDER_ROLL): (-1.45 - _ARM_BAND, -1.45 + _ARM_BAND),
    int(JointIndex.RIGHT_SHOULDER_ROLL): (1.45 - _ARM_BAND, 1.45 + _ARM_BAND),
    int(JointIndex.LEFT_ELBOW_PITCH): (-_ARM_BAND, _ARM_BAND),
    int(JointIndex.RIGHT_ELBOW_PITCH): (-_ARM_BAND, _ARM_BAND),
    int(JointIndex.LEFT_ELBOW_YAW): (-_ARM_BAND, _ARM_BAND),
    int(JointIndex.RIGHT_ELBOW_YAW): (-_ARM_BAND, _ARM_BAND),
}

FRAME_ANG_VEL = 3
FRAME_PROJ_GRAV = 3
FRAME_JOINT_POS = 22
FRAME_JOINT_VEL = 22
FRAME_ACTIONS = 20
FRAME_COMMAND = 3
FRAME_DIM = (
    FRAME_ANG_VEL
    + FRAME_PROJ_GRAV
    + FRAME_JOINT_POS
    + FRAME_JOINT_VEL
    + FRAME_ACTIONS
    + FRAME_COMMAND
)
assert FRAME_DIM == 73

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    _REPO_ROOT
    / "v6_models"
    / "2026-08-12_10-41-21_k1_minimal_scratch_v3_noheightscan.onnx"
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


def _parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _apply_walk_pd_overrides(kp: np.ndarray, kd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Metadata kp/kd read, then leg joints overwritten with custom walk gains."""
    return kp, kd # disable walk pd overrides
    kp = kp.copy()
    kd = kd.copy()
    for ji, val in WALK_KP.items():
        kp[ji] = val
    for ji, val in WALK_KD.items():
        kd[ji] = val
    return kp, kd


class WalkPolicyV6(Policy):
    """Single-frame obs (73). Controls 20 joints; head held elsewhere."""

    def __init__(
        self,
        control_dt: float = 0.02,
        model_path: Optional[Union[str, Path]] = None,
        load_default_model: bool = True,
    ) -> None:
        del control_dt

        self._default_joint_pos = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        self._default_action_pos = np.zeros(ACTION_DIM, dtype=np.float64)
        self._action_scale = np.ones(ACTION_DIM, dtype=np.float64)
        self._last_action: List[float] = [0.0] * ACTION_DIM

        self._session: Optional[ort.InferenceSession] = None
        self._input_name = "obs"
        self._output_name = "actions"
        self._input_dim = FRAME_DIM

        path = model_path
        if path is None and load_default_model:
            path = DEFAULT_MODEL_PATH
        if path is not None:
            self.load_model(str(path))

    def name(self) -> str:
        return "walk_v6"

    def observation_dim(self) -> int:
        return self._input_dim

    def history_len(self) -> int:
        return 1

    def input_dim(self) -> int:
        return self._input_dim

    def controlled_joints(self) -> List[int]:
        return [int(i) for i in ACTION_JOINTS]

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"walk_v6: model not found: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"walk_v6: ONNX has no inputs/outputs: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        shape = inputs[0].shape
        dims = [d for d in shape if isinstance(d, int) and d > 0]
        if not dims:
            raise RuntimeError(f"walk_v6: cannot parse obs shape {shape}")
        obs_dim = int(dims[-1])
        if obs_dim != FRAME_DIM:
            raise RuntimeError(
                f"walk_v6: unsupported ONNX obs dim {obs_dim} "
                f"(expected {FRAME_DIM})"
            )
        self._input_dim = FRAME_DIM

        out_dims = [d for d in outputs[0].shape if isinstance(d, int) and d > 0]
        if not out_dims or int(out_dims[-1]) != ACTION_DIM:
            raise RuntimeError(
                f"walk_v6: unexpected action dim {out_dims} (expected {ACTION_DIM})"
            )

        meta = dict(self._session.get_modelmeta().custom_metadata_map)
        self._apply_metadata(meta)
        print(
            f"[walk_v6] loaded {path.name}: in {self._input_name!r} [1, {self._input_dim}] "
            f"-> out {self._output_name!r} [1, {ACTION_DIM}]"
        )

    def _apply_metadata(self, meta: Dict[str, str]) -> None:
        if "default_joint_pos" not in meta:
            raise RuntimeError("walk_v6: ONNX metadata missing default_joint_pos")
        vals = _parse_csv_floats(meta["default_joint_pos"])
        if len(vals) != B1_JOINT_COUNT:
            raise RuntimeError(
                f"walk_v6: default_joint_pos has {len(vals)} entries, "
                f"expected {B1_JOINT_COUNT}"
            )
        self._default_joint_pos = np.asarray(vals, dtype=np.float64)
        self._default_action_pos = self._default_joint_pos[ACTION_JOINTS].copy()

        if "joint_stiffness" in meta and "joint_damping" in meta:
            kp = _parse_csv_floats(meta["joint_stiffness"])
            kd = _parse_csv_floats(meta["joint_damping"])
            if len(kp) != B1_JOINT_COUNT or len(kd) != B1_JOINT_COUNT:
                raise RuntimeError(
                    "walk_v6: joint_stiffness/joint_damping length mismatch"
                )
            _apply_walk_pd_overrides(
                np.asarray(kp, dtype=np.float64),
                np.asarray(kd, dtype=np.float64),
            )

        if "action_scale" in meta:
            scale = _parse_csv_floats(meta["action_scale"])
            if len(scale) != ACTION_DIM:
                raise RuntimeError(
                    f"walk_v6: action_scale has {len(scale)} entries, "
                    f"expected {ACTION_DIM}"
                )
            self._action_scale = np.asarray(scale, dtype=np.float64)

    def _joint_pos_relative(self, q_abs: Sequence[float]) -> np.ndarray:
        q = np.asarray(q_abs, dtype=np.float64)
        if q.shape != (B1_JOINT_COUNT,):
            raise ValueError("walk_v6: expected 22 joint positions")
        return q - self._default_joint_pos

    def _action_to_absolute(self, action: Sequence[float]) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (ACTION_DIM,):
            raise ValueError(f"walk_v6: expected {ACTION_DIM}-dim action")
        q = self._default_action_pos + self._action_scale * a
        for i, joint_idx in enumerate(ACTION_JOINTS):
            ji = int(joint_idx)
            if ji in _ARM_CLIP_BY_INDEX:
                lo, hi = _ARM_CLIP_BY_INDEX[ji]
                q[i] = float(np.clip(q[i], lo, hi))
        lo = Q_ABS_LIMITS[ACTION_JOINTS, 0]
        hi = Q_ABS_LIMITS[ACTION_JOINTS, 1]
        return np.clip(q, lo, hi)

    def _last_action_from_sent(self, q_abs: np.ndarray) -> List[float]:
        """Relative action matching the q that was actually sent (after clips)."""
        scale = np.where(np.abs(self._action_scale) > 1e-12, self._action_scale, 1.0)
        rel = (q_abs - self._default_action_pos) / scale
        return [float(x) for x in rel]

    def build_observation(
        self,
        state: RobotState,
        command: Sequence[float],
    ) -> Observation:
        if len(state.q) != B1_JOINT_COUNT or len(state.dq) != B1_JOINT_COUNT:
            raise ValueError("walk_v6: RobotState q/dq size mismatch")

        cmd = [float(x) for x in command[:FRAME_COMMAND]]
        while len(cmd) < FRAME_COMMAND:
            cmd.append(0.0)

        joint_pos_rel = self._joint_pos_relative(state.q)
        joint_vel = np.asarray(state.dq, dtype=np.float64)

        gyro = np.zeros(FRAME_ANG_VEL, dtype=np.float64)
        gyro[: min(FRAME_ANG_VEL, len(state.imu.gyro))] = state.imu.gyro[:FRAME_ANG_VEL]
        grav = np.zeros(FRAME_PROJ_GRAV, dtype=np.float64)
        grav[: min(FRAME_PROJ_GRAV, len(state.projected_gravity))] = (
            state.projected_gravity[:FRAME_PROJ_GRAV]
        )
        last_a = np.asarray(self._last_action, dtype=np.float64)
        cmd_np = np.asarray(cmd, dtype=np.float64)

        data = np.concatenate([gyro, grav, joint_pos_rel, joint_vel, last_a, cmd_np])
        if data.shape != (self._input_dim,):
            raise ValueError(
                f"walk_v6: built frame dim {data.size} != input_dim {self._input_dim}"
            )
        return Observation(data=data.tolist())

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)
        if self._session is None:
            raise RuntimeError("walk_v6: model not loaded; call load_model()")

        x = np.asarray(obs.data, dtype=np.float32).reshape(1, self._input_dim)
        raw = self._session.run([self._output_name], {self._input_name: x})[0]
        action = np.asarray(raw, dtype=np.float64).reshape(ACTION_DIM)

        q_abs = self._action_to_absolute(action)
        self._last_action = self._last_action_from_sent(q_abs)

        joint_cmds: List[JointCommand] = []
        for i, joint_idx in enumerate(ACTION_JOINTS):
            ji = int(joint_idx)
            q = float(q_abs[i])
            if ji in WALK_KP:
                joint_cmds.append(make_walk_joint_cmd(ji, q))
            else:
                joint_cmds.append(make_joint_cmd(ji, q))
        return Action(joint_cmds=joint_cmds)

    def reset(self) -> None:
        self._last_action = [0.0] * ACTION_DIM
