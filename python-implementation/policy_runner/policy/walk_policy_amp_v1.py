"""Walk AMP v1 — mjlab K1 AMP velocity policy, 75-dim obs → 22-dim action.

Frame layout (75), matching k1_amp actor observations:
  base_ang_vel (3) + projected_gravity (3) + joint_pos (22) + joint_vel (22)
  + actions (22) + command (3)

All 22 joints are policy-controlled (including head). No arm-target clip.
default_joint_pos, action_scale, and PD (joint_stiffness / joint_damping)
come from ONNX metadata. PD can be overridden at construction / CLI.

Sent q is clipped to physical joint ranges. last_action feeds back the raw
ONNX output before scale/offset/clip, matching mjlab training
(env.action_manager.action is the raw policy output).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from policy_runner.joint_index import JOINT_NAMES, JointIndex
from policy_runner.policy.base import Policy
from policy_runner.types import (
    B1_JOINT_COUNT,
    Action,
    JointCommand,
    Observation,
    RobotState,
)

ACTION_DIM = B1_JOINT_COUNT
ACTION_JOINTS = np.arange(ACTION_DIM, dtype=np.int64)

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

# Fallback PD if ONNX metadata is missing — matches k1_whirlwind training.
AMP_DEFAULT_KP = np.asarray(
    [
        4.0, 4.0,
        10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0,
        80.0, 80.0, 80.0, 80.0, 50.0, 50.0,
        80.0, 80.0, 80.0, 80.0, 50.0, 50.0,
    ],
    dtype=np.float64,
)
AMP_DEFAULT_KD = np.asarray(
    [
        0.25, 0.25,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
    ],
    dtype=np.float64,
)
assert AMP_DEFAULT_KP.shape == (B1_JOINT_COUNT,)
assert AMP_DEFAULT_KD.shape == (B1_JOINT_COUNT,)

# MuJoCo / mjlab joint names in JointIndex order (ROS uses AAHead_yaw / Head_pitch).
MJ_JOINT_NAMES: List[str] = [
    "Head_Yaw",
    "Head_Pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

# Standing base height at AMP HOME_KEYFRAME.
AMP_INIT_POS = (0.0, 0.0, 0.5125)

FRAME_ANG_VEL = 3
FRAME_PROJ_GRAV = 3
FRAME_JOINT_POS = 22
FRAME_JOINT_VEL = 22
FRAME_ACTIONS = 22
FRAME_COMMAND = 3
FRAME_DIM = (
    FRAME_ANG_VEL
    + FRAME_PROJ_GRAV
    + FRAME_JOINT_POS
    + FRAME_JOINT_VEL
    + FRAME_ACTIONS
    + FRAME_COMMAND
)
assert FRAME_DIM == 75

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = _REPO_ROOT / "amp_models" / "model_29999.onnx"

_PD_NAME_TO_INDEX: Dict[str, int] = {}
for _i, _name in enumerate(JOINT_NAMES):
    _PD_NAME_TO_INDEX[_name.lower()] = _i
for _i, _name in enumerate(MJ_JOINT_NAMES):
    _PD_NAME_TO_INDEX[_name.lower()] = _i
for _member in JointIndex:
    _PD_NAME_TO_INDEX[_member.name.lower()] = int(_member)


def parse_csv_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_pd_vector(raw: str, label: str) -> np.ndarray:
    """Parse a 22-value CSV into a PD vector."""
    vals = parse_csv_floats(raw)
    if len(vals) != B1_JOINT_COUNT:
        raise ValueError(
            f"walk_amp_v1: {label} has {len(vals)} entries, expected {B1_JOINT_COUNT}"
        )
    return np.asarray(vals, dtype=np.float64)


def parse_pd_overrides(raw: str) -> Dict[int, float]:
    """Parse sparse PD overrides: ``14:30,Left_Ankle_Roll=15``."""
    out: Dict[int, float] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            key, val = token.split(":", 1)
        elif "=" in token:
            key, val = token.split("=", 1)
        else:
            raise ValueError(
                f"walk_amp_v1: invalid PD override {token!r} "
                "(want idx:value or name=value)"
            )
        key = key.strip()
        value = float(val.strip())
        if key.isdigit() or (key.startswith("-") and key[1:].isdigit()):
            idx = int(key)
        else:
            idx = _PD_NAME_TO_INDEX.get(key.lower())
            if idx is None:
                raise ValueError(
                    f"walk_amp_v1: unknown joint in PD override: {key!r}"
                )
        if idx < 0 or idx >= B1_JOINT_COUNT:
            raise ValueError(f"walk_amp_v1: PD override index out of range: {idx}")
        out[idx] = value
    return out


def _as_pd_vector(
    values: Optional[Sequence[float]], label: str
) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (B1_JOINT_COUNT,):
        raise ValueError(
            f"walk_amp_v1: {label} has shape {arr.shape}, expected ({B1_JOINT_COUNT},)"
        )
    return arr


class WalkPolicyAmpV1(Policy):
    """Single-frame obs (75). Controls all 22 joints."""

    def __init__(
        self,
        control_dt: float = 0.02,
        model_path: Optional[Union[str, Path]] = None,
        load_default_model: bool = True,
        kp: Optional[Sequence[float]] = None,
        kd: Optional[Sequence[float]] = None,
        kp_override: Optional[Mapping[int, float]] = None,
        kd_override: Optional[Mapping[int, float]] = None,
    ) -> None:
        del control_dt

        self._default_joint_pos = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        self._action_scale = np.ones(ACTION_DIM, dtype=np.float64)
        self._kp = AMP_DEFAULT_KP.copy()
        self._kd = AMP_DEFAULT_KD.copy()
        self._last_action: List[float] = [0.0] * ACTION_DIM
        self._kp_replace = _as_pd_vector(kp, "kp")
        self._kd_replace = _as_pd_vector(kd, "kd")
        self._kp_override = dict(kp_override) if kp_override else {}
        self._kd_override = dict(kd_override) if kd_override else {}

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
        return "walk_amp_v1"

    def observation_dim(self) -> int:
        return self._input_dim

    def history_len(self) -> int:
        return 1

    def input_dim(self) -> int:
        return self._input_dim

    def controlled_joints(self) -> List[int]:
        return [int(i) for i in ACTION_JOINTS]

    @property
    def default_joint_pos(self) -> np.ndarray:
        return self._default_joint_pos.copy()

    @property
    def kp(self) -> np.ndarray:
        return self._kp.copy()

    @property
    def kd(self) -> np.ndarray:
        return self._kd.copy()

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"walk_amp_v1: model not found: {path}")

        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError(f"walk_amp_v1: ONNX has no inputs/outputs: {path}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        shape = inputs[0].shape
        dims = [d for d in shape if isinstance(d, int) and d > 0]
        if not dims:
            raise RuntimeError(f"walk_amp_v1: cannot parse obs shape {shape}")
        obs_dim = int(dims[-1])
        if obs_dim != FRAME_DIM:
            raise RuntimeError(
                f"walk_amp_v1: unsupported ONNX obs dim {obs_dim} "
                f"(expected {FRAME_DIM})"
            )
        self._input_dim = FRAME_DIM

        out_dims = [d for d in outputs[0].shape if isinstance(d, int) and d > 0]
        if not out_dims or int(out_dims[-1]) != ACTION_DIM:
            raise RuntimeError(
                f"walk_amp_v1: unexpected action dim {out_dims} "
                f"(expected {ACTION_DIM})"
            )

        meta = dict(self._session.get_modelmeta().custom_metadata_map)
        self._apply_metadata(meta)
        self._apply_pd_overrides()
        print(
            f"[walk_amp_v1] loaded {path.name}: in {self._input_name!r} "
            f"[1, {self._input_dim}] -> out {self._output_name!r} [1, {ACTION_DIM}]"
        )
        print(
            f"[walk_amp_v1] kp={np.array2string(self._kp, precision=2, separator=',')}"
        )
        print(
            f"[walk_amp_v1] kd={np.array2string(self._kd, precision=2, separator=',')}"
        )

    def _apply_metadata(self, meta: Dict[str, str]) -> None:
        if "default_joint_pos" not in meta:
            raise RuntimeError("walk_amp_v1: ONNX metadata missing default_joint_pos")
        vals = parse_csv_floats(meta["default_joint_pos"])
        if len(vals) != B1_JOINT_COUNT:
            raise RuntimeError(
                f"walk_amp_v1: default_joint_pos has {len(vals)} entries, "
                f"expected {B1_JOINT_COUNT}"
            )
        self._default_joint_pos = np.asarray(vals, dtype=np.float64)

        if "action_scale" not in meta:
            raise RuntimeError("walk_amp_v1: ONNX metadata missing action_scale")
        scale = parse_csv_floats(meta["action_scale"])
        if len(scale) != ACTION_DIM:
            raise RuntimeError(
                f"walk_amp_v1: action_scale has {len(scale)} entries, "
                f"expected {ACTION_DIM}"
            )
        self._action_scale = np.asarray(scale, dtype=np.float64)

        if "joint_stiffness" in meta and "joint_damping" in meta:
            kp = parse_csv_floats(meta["joint_stiffness"])
            kd = parse_csv_floats(meta["joint_damping"])
            if len(kp) != B1_JOINT_COUNT or len(kd) != B1_JOINT_COUNT:
                raise RuntimeError(
                    "walk_amp_v1: joint_stiffness/joint_damping length mismatch"
                )
            self._kp = np.asarray(kp, dtype=np.float64)
            self._kd = np.asarray(kd, dtype=np.float64)
        else:
            print(
                "[walk_amp_v1] ONNX metadata missing joint_stiffness/joint_damping; "
                "using AMP training defaults"
            )
            self._kp = AMP_DEFAULT_KP.copy()
            self._kd = AMP_DEFAULT_KD.copy()

    def _apply_pd_overrides(self) -> None:
        if self._kp_replace is not None:
            self._kp = self._kp_replace.copy()
            print("[walk_amp_v1] kp replaced from --kp / constructor")
        if self._kd_replace is not None:
            self._kd = self._kd_replace.copy()
            print("[walk_amp_v1] kd replaced from --kd / constructor")
        for idx, val in self._kp_override.items():
            self._kp[idx] = float(val)
        for idx, val in self._kd_override.items():
            self._kd[idx] = float(val)
        if self._kp_override or self._kd_override:
            print(
                f"[walk_amp_v1] sparse PD overrides: "
                f"kp={self._kp_override} kd={self._kd_override}"
            )

    def _joint_pos_relative(self, q_abs: Sequence[float]) -> np.ndarray:
        q = np.asarray(q_abs, dtype=np.float64)
        if q.shape != (B1_JOINT_COUNT,):
            raise ValueError("walk_amp_v1: expected 22 joint positions")
        return q - self._default_joint_pos

    def _action_to_absolute(self, action: Sequence[float]) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (ACTION_DIM,):
            raise ValueError(f"walk_amp_v1: expected {ACTION_DIM}-dim action")
        q = self._default_joint_pos + self._action_scale * a
        return np.clip(q, Q_ABS_LIMITS[:, 0], Q_ABS_LIMITS[:, 1])

    def build_observation(
        self,
        state: RobotState,
        command: Sequence[float],
    ) -> Observation:
        if len(state.q) != B1_JOINT_COUNT or len(state.dq) != B1_JOINT_COUNT:
            raise ValueError("walk_amp_v1: RobotState q/dq size mismatch")

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
                f"walk_amp_v1: built frame dim {data.size} != input_dim {self._input_dim}"
            )
        return Observation(data=data.tolist())

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)
        if self._session is None:
            raise RuntimeError("walk_amp_v1: model not loaded; call load_model()")

        x = np.asarray(obs.data, dtype=np.float32).reshape(1, self._input_dim)
        raw = self._session.run([self._output_name], {self._input_name: x})[0]
        action = np.asarray(raw, dtype=np.float64).reshape(ACTION_DIM)

        q_abs = self._action_to_absolute(action)
        # Feed back the raw network output, not the clip-corrected q.
        self._last_action = [float(v) for v in action]

        joint_cmds = [
            JointCommand(
                index=i,
                q=float(q_abs[i]),
                dq=0.0,
                tau=0.0,
                kp=float(self._kp[i]),
                kd=float(self._kd[i]),
                weight=1.0,
            )
            for i in range(ACTION_DIM)
        ]
        return Action(joint_cmds=joint_cmds)

    def reset(self) -> None:
        self._last_action = [0.0] * ACTION_DIM
