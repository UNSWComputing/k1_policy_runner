"""MuJoCo sim bridge for sim-to-sim policy tests.

Same RobotState / Action surface as ROS RobotBridge:
  latest_state() -> RobotState (22 joints + IMU + projected gravity)
  publish_action(Action) -> set PD targets (sparse; others hold default)
  step() -> run physics with PD torques via qfrc_applied
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from policy_runner.joint_index import JointIndex
from policy_runner.types import (
    B1_JOINT_COUNT,
    Action,
    ImuState,
    RobotState,
    compute_projected_gravity,
)

_ASSETS = Path(__file__).resolve().parents[3] / "assets"
DEFAULT_MJCF = _ASSETS / "k1_22dof_scene.xml"

# Match ParameterWalk / deploy standing pose (22-DoF, no waist).
DEFAULT_Q = np.asarray(
    [
        0.0,
        0.0,  # head
        0.2,
        -1.35,
        0.0,
        -0.5,  # left arm
        0.2,
        1.35,
        0.0,
        0.5,  # right arm
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.25,
        0.0,  # left leg
        -0.2,
        0.0,
        0.0,
        0.4,
        -0.25,
        0.0,  # right leg
    ],
    dtype=np.float64,
)
assert DEFAULT_Q.shape == (B1_JOINT_COUNT,)

# Soft hold for joints the policy does not own (head/arms).
# Arms need enough kp to hold walk_v1's bent-elbow default (~2 rad).
HOLD_KP = np.asarray(
    [
        20,
        20,
        40,
        50,
        40,
        20,
        40,
        50,
        40,
        20,
        100,
        100,
        100,
        100,
        50,
        50,
        100,
        100,
        100,
        100,
        50,
        50,
    ],
    dtype=np.float64,
)
HOLD_KD = np.asarray(
    [
        0.2,
        0.2,
        1.0,
        1.5,
        1.0,
        0.5,
        1.0,
        1.5,
        1.0,
        0.5,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        1,
    ],
    dtype=np.float64,
)

# Effort clip (Nm), aligned with deploy torque_limit legs / modest arms.
EFFORT_LIMIT = np.asarray(
    [
        7,
        7,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        10,
        60,
        25,
        30,
        60,
        24,
        15,
        60,
        25,
        30,
        60,
        24,
        15,
    ],
    dtype=np.float64,
)


def _quat_wxyz_to_rpy(quat: Sequence[float]) -> list[float]:
    """MuJoCo quat (w, x, y, z) → roll, pitch, yaw."""
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    # roll (x)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # yaw (z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


class MujocoBridge:
    """Local MuJoCo stand-in for RobotBridge (no ROS)."""

    def __init__(
        self,
        mjcf_path: Optional[str] = None,
        control_dt: float = 0.02,
        physics_dt: Optional[float] = None,
        init_pos: Sequence[float] = (0.0, 0.0, 0.58),
        init_quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        default_q: Optional[Sequence[float]] = None,
    ) -> None:
        path = Path(mjcf_path) if mjcf_path else DEFAULT_MJCF
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo MJCF not found: {path}")

        self.model = mujoco.MjModel.from_xml_path(str(path))
        if physics_dt is not None:
            self.model.opt.timestep = float(physics_dt)
        self.data = mujoco.MjData(self.model)

        self.control_dt = float(control_dt)
        self.decimation = max(
            1, int(round(self.control_dt / float(self.model.opt.timestep)))
        )

        if self.model.nq != 7 + B1_JOINT_COUNT or self.model.nv != 6 + B1_JOINT_COUNT:
            raise RuntimeError(
                f"Expected floating-base 22-DoF model "
                f"(nq={7 + B1_JOINT_COUNT}, nv={6 + B1_JOINT_COUNT}), "
                f"got nq={self.model.nq} nv={self.model.nv}"
            )

        # Confirm hinge joint name order matches JointIndex / policy order.
        for i in range(B1_JOINT_COUNT):
            jname = self.model.joint(i + 1).name  # 0 is freejoint
            mj_expected = _JOINT_INDEX_TO_MJ_NAME[i]
            if jname != mj_expected:
                raise RuntimeError(
                    f"MuJoCo joint order mismatch at {i}: got {jname}, "
                    f"expected {mj_expected} (JointIndex / policy order)"
                )

        self._default_q = (
            np.asarray(default_q, dtype=np.float64)
            if default_q is not None
            else DEFAULT_Q.copy()
        )
        if self._default_q.shape != (B1_JOINT_COUNT,):
            raise ValueError("default_q must have length 22")

        self._q_target = self._default_q.copy()
        self._kp = HOLD_KP.copy()
        self._kd = HOLD_KD.copy()
        self._last_tau = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        self._trunk_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "Trunk"
        )

        self.reset(init_pos=init_pos, init_quat_wxyz=init_quat_wxyz)

    @property
    def q_target(self) -> np.ndarray:
        return self._q_target.copy()

    @property
    def last_tau(self) -> np.ndarray:
        return self._last_tau.copy()

    def reset(
        self,
        init_pos: Sequence[float] = (0.0, 0.0, 0.58),
        init_quat_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.asarray(init_pos, dtype=np.float64)
        self.data.qpos[3:7] = np.asarray(init_quat_wxyz, dtype=np.float64)
        self.data.qpos[7:] = self._default_q
        self.data.qvel[:] = 0.0
        self._q_target[:] = self._default_q
        self._kp[:] = HOLD_KP
        self._kd[:] = HOLD_KD
        self._last_tau[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def has_state(self) -> bool:
        return True

    def latest_state(self) -> RobotState:
        q = self.data.qpos[7 : 7 + B1_JOINT_COUNT].astype(np.float64).tolist()
        dq = self.data.qvel[6 : 6 + B1_JOINT_COUNT].astype(np.float64).tolist()

        quat = self.data.qpos[3:7]
        rpy = _quat_wxyz_to_rpy(quat)
        # Free-joint angular velocity is in the body frame.
        gyro = self.data.qvel[3:6].astype(np.float64).tolist()
        # Linear acceleration from sensor if present, else zeros.
        acc = [0.0, 0.0, 0.0]
        sid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_acc"
        )
        if sid >= 0:
            adr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            acc = self.data.sensordata[adr : adr + dim].astype(np.float64).tolist()

        # Projected gravity from trunk rotation (world gravity [0,0,-1]).
        if self._trunk_body_id >= 0:
            R = self.data.xmat[self._trunk_body_id].reshape(3, 3)
            proj = (R.T @ np.array([0.0, 0.0, -1.0])).tolist()
        else:
            proj = compute_projected_gravity(rpy)

        return RobotState(
            q=q,
            dq=dq,
            imu=ImuState(rpy=rpy, gyro=gyro, acc=acc),
            projected_gravity=proj,
        )

    def publish_action(self, action: Action) -> None:
        """Apply sparse joint commands; untouched joints keep hold targets/gains."""
        # Reset to hold defaults each tick, then overlay policy cmds.
        self._q_target[:] = self._default_q
        self._kp[:] = HOLD_KP
        self._kd[:] = HOLD_KD
        for jc in action.joint_cmds:
            if jc.weight <= 0.0:
                continue
            if jc.index < 0 or jc.index >= B1_JOINT_COUNT:
                raise ValueError(f"joint index out of range: {jc.index}")
            self._q_target[jc.index] = float(jc.q)
            self._kp[jc.index] = float(jc.kp)
            self._kd[jc.index] = float(jc.kd)

    def step(self, n_substeps: Optional[int] = None) -> None:
        """Integrate physics for one control period (PD torque + mj_step)."""
        steps = self.decimation if n_substeps is None else int(n_substeps)
        for _ in range(steps):
            q = self.data.qpos[7 : 7 + B1_JOINT_COUNT]
            dq = self.data.qvel[6 : 6 + B1_JOINT_COUNT]
            tau = self._kp * (self._q_target - q) - self._kd * dq
            tau = np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)
            self._last_tau[:] = tau
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[6 : 6 + B1_JOINT_COUNT] = tau
            mujoco.mj_step(self.model, self.data)


# MuJoCo hinge names in JointIndex order (must match MJCF DFS order).
_JOINT_INDEX_TO_MJ_NAME = [
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

# Documented for readers; mapping table above is authoritative.
assert len(_JOINT_INDEX_TO_MJ_NAME) == B1_JOINT_COUNT
assert all(JointIndex(i).value == i for i in range(B1_JOINT_COUNT))
