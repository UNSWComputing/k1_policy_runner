"""Shared types for the Python policy runner (mirrors C++ types.hpp)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

B1_JOINT_COUNT = 22


@dataclass
class RobotState:
    """Full-body robot state from /joint_states. Always sized to the robot."""

    q: List[float] = field(default_factory=list)
    dq: List[float] = field(default_factory=list)


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
