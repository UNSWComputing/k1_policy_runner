"""Shared types for the Python policy runner (mirrors C++ types.hpp)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence

B1_JOINT_COUNT = 22
IMU_DIM = 9  # rpy(3) + gyro(3) + acc(3)
PROJECTED_GRAVITY_DIM = 3
# Default packed observation without command extras.
BASE_OBS_DIM = B1_JOINT_COUNT + IMU_DIM + PROJECTED_GRAVITY_DIM


@dataclass
class ImuState:
    """IMU from LowState.imu_state: rpy, gyro, acc (each length 3)."""

    rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gyro: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acc: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def as_list(self) -> List[float]:
        return [*self.rpy, *self.gyro, *self.acc]


def compute_projected_gravity(rpy: Sequence[float]) -> List[float]:
    """Gravity [0, 0, -1] expressed in the base frame from roll/pitch (yaw-invariant)."""
    roll = float(rpy[0]) if len(rpy) > 0 else 0.0
    pitch = float(rpy[1]) if len(rpy) > 1 else 0.0
    return [
        -math.sin(pitch),
        math.sin(roll) * math.cos(pitch),
        math.cos(roll) * math.cos(pitch),
    ]


@dataclass
class RobotState:
    """Robot state from /joint_states + /low_state IMU."""

    q: List[float] = field(default_factory=list)
    dq: List[float] = field(default_factory=list)
    imu: ImuState = field(default_factory=ImuState)
    # Unit gravity vector in the robot base frame (from IMU RPY).
    projected_gravity: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 1.0]
    )


@dataclass
class Observation:
    """Flat observation vector. Layout is policy-defined."""

    data: List[float] = field(default_factory=list)


@dataclass
class JointCommand:
    """One controlled joint. Partial-body policies emit only joints they own."""

    index: int = -1
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    weight: float = 1.0


@dataclass
class Action:
    """Sparse action: only joints this policy controls."""

    joint_cmds: List[JointCommand] = field(default_factory=list)


def pack_observation(
    state: RobotState, command: Sequence[float] = ()
) -> Observation:
    """Default layout: [joint_q..., imu..., projected_gravity..., optional command...]."""
    if len(state.q) != B1_JOINT_COUNT:
        raise ValueError("RobotState.q size mismatch")
    data = list(state.q)
    data.extend(state.imu.as_list())
    data.extend(state.projected_gravity)
    data.extend(float(x) for x in command)
    return Observation(data=data)
