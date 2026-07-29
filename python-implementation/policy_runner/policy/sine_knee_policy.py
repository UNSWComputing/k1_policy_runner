"""Hard-coded sine-wave knee-pitch policy (mirrors C++ SineKneePolicy)."""

from __future__ import annotations

import math
from typing import List, Sequence

from policy_runner.joint_gains import make_joint_cmd
from policy_runner.joint_index import JointIndex
from policy_runner.policy.base import Policy
from policy_runner.types import (
    BASE_OBS_DIM,
    Action,
    Observation,
    RobotState,
    pack_observation,
)

LEFT_KNEE_PITCH = int(JointIndex.LEFT_KNEE_PITCH)
RIGHT_KNEE_PITCH = int(JointIndex.RIGHT_KNEE_PITCH)

# Knee pitch limits: 0° … 133° (absolute rad).
_KNEE_MIN = math.radians(0.0)
_KNEE_MAX = math.radians(133.0)
_KNEE_BASE = 0.5 * (_KNEE_MIN + _KNEE_MAX)  # mid ≈ 66.5°
_KNEE_AMP = 0.5 * (_KNEE_MAX - _KNEE_MIN)  # ±66.5°
_PERIOD_S = 5.0 
_FREQ_HZ = 1.0 / _PERIOD_S


class SineKneePolicy(Policy):
    """Knee pitches (13, 19) follow a sine wave. Sparse — other joints weight=0."""

    def __init__(self, control_dt: float = 0.02) -> None:
        self._control_dt = control_dt
        self._time_s = 0.0

    def name(self) -> str:
        return "sine_knee"

    def input_dim(self) -> int:
        return BASE_OBS_DIM

    def controlled_joints(self) -> List[int]:
        return [LEFT_KNEE_PITCH, RIGHT_KNEE_PITCH]

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        return pack_observation(state, command)

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        phase = 2.0 * math.pi * _FREQ_HZ * self._time_s
        knee_q = _KNEE_BASE + _KNEE_AMP * math.sin(phase)

        self._time_s += self._control_dt
        return Action(
            joint_cmds=[
                make_joint_cmd(LEFT_KNEE_PITCH, knee_q),
                make_joint_cmd(RIGHT_KNEE_PITCH, knee_q),
            ]
        )

    def reset(self) -> None:
        self._time_s = 0.0
