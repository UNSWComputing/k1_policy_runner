"""Sine-wave ankle-pitch policy; all other joints held at latched start pose."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from policy_runner.joint_gains import make_joint_cmd
from policy_runner.joint_index import JointIndex
from policy_runner.policy.base import Policy
from policy_runner.types import (
    BASE_OBS_DIM,
    B1_JOINT_COUNT,
    Action,
    Observation,
    RobotState,
    pack_observation,
)

LEFT_ANKLE_PITCH = int(JointIndex.LEFT_ANKLE_PITCH)
RIGHT_ANKLE_PITCH = int(JointIndex.RIGHT_ANKLE_PITCH)

# Oscillation about the latched ankle pitch (absolute rad).
_ANKLE_AMP = 0.5
_PERIOD_S = 5.0
_FREQ_HZ = 1.0 / _PERIOD_S


class SineAnklePolicy(Policy):
    """
    Ankle pitches (14, 20) follow a sine wave about the start pose.
    All other joints are held at their latched start values.
    """

    def __init__(self, control_dt: float = 0.02) -> None:
        self._control_dt = control_dt
        self._time_s = 0.0
        self._hold_captured = False
        self._hold_q: Optional[List[float]] = None

    def name(self) -> str:
        return "sine_ankle"

    def input_dim(self) -> int:
        return BASE_OBS_DIM

    def controlled_joints(self) -> List[int]:
        return list(range(B1_JOINT_COUNT))

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        del command  # unused; keep frame at BASE_OBS_DIM (no cmd extras)
        return pack_observation(state)

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        if not self._hold_captured:
            self._hold_q = list(obs.data[:B1_JOINT_COUNT])
            self._hold_captured = True

        hold = self._hold_q or obs.data[:B1_JOINT_COUNT]
        phase = 2.0 * math.pi * _FREQ_HZ * self._time_s
        delta = _ANKLE_AMP * math.sin(phase)
        left_q = float(hold[LEFT_ANKLE_PITCH]) + delta
        right_q = float(hold[RIGHT_ANKLE_PITCH]) + delta

        self._time_s += self._control_dt
        cmds = []
        for i in range(B1_JOINT_COUNT):
            if i == LEFT_ANKLE_PITCH:
                q = left_q
            elif i == RIGHT_ANKLE_PITCH:
                q = right_q
            else:
                q = float(hold[i])
            cmds.append(make_joint_cmd(i, q))
        return Action(joint_cmds=cmds)

    def reset(self) -> None:
        self._time_s = 0.0
        self._hold_captured = False
        self._hold_q = None
