"""Sine-wave head yaw scan; pitch held fixed. Sparse — pairs with walk policies."""

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

HEAD_YAW = int(JointIndex.HEAD_YAW)
HEAD_PITCH = int(JointIndex.HEAD_PITCH)

# Yaw oscillates about center; pitch held constant.
_YAW_CENTER = 0.0
_YAW_AMP = 0.8  # rad (~46 deg); joint limit ±1.0
_PITCH_FIXED = 0.2
_FREQ_HZ = 0.25  # slow left–right scan


class SineHeadPolicy(Policy):
    """Head yaw (0) sine-scans L/R; pitch (1) held fixed. Sparse output."""

    def __init__(self, control_dt: float = 0.02) -> None:
        self._control_dt = control_dt
        self._time_s = 0.0

    def name(self) -> str:
        return "sine_head"

    def input_dim(self) -> int:
        return BASE_OBS_DIM

    def controlled_joints(self) -> List[int]:
        return [HEAD_YAW, HEAD_PITCH]

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        del command  # unused; keep frame at BASE_OBS_DIM (no cmd extras)
        return pack_observation(state)

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        phase = 2.0 * math.pi * _FREQ_HZ * self._time_s
        yaw_q = _YAW_CENTER + _YAW_AMP * math.sin(phase)

        self._time_s += self._control_dt
        return Action(
            joint_cmds=[
                make_joint_cmd(HEAD_YAW, yaw_q),
                make_joint_cmd(HEAD_PITCH, _PITCH_FIXED),
            ]
        )

    def reset(self) -> None:
        self._time_s = 0.0
