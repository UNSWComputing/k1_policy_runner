"""Hard-coded sine-wave elbow policy (mirrors C++ SineArmPolicy)."""

from __future__ import annotations

import math
from typing import List, Sequence

from policy_runner.policy.arm_demo_common import (
    LEFT_ELBOW,
    RIGHT_ELBOW,
    make_elbow_cmd,
)
from policy_runner.policy.base import Policy
from policy_runner.types import (
    BASE_OBS_DIM,
    Action,
    Observation,
    RobotState,
    pack_observation,
)

_ELBOW_BASE = 0.0
_ELBOW_AMP = 0.4
_FREQ_HZ = 0.5


class SineArmPolicy(Policy):
    """Elbows (5, 9) follow a sine wave. Sparse output — other joints weight=0."""

    def __init__(self, control_dt: float = 0.02) -> None:
        self._control_dt = control_dt
        self._time_s = 0.0

    def name(self) -> str:
        return "sine_arm"

    def input_dim(self) -> int:
        return BASE_OBS_DIM

    def controlled_joints(self) -> List[int]:
        return [LEFT_ELBOW, RIGHT_ELBOW]

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        return pack_observation(state, command)

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        phase = 2.0 * math.pi * _FREQ_HZ * self._time_s
        elbow_q = _ELBOW_BASE + _ELBOW_AMP * math.sin(phase)

        self._time_s += self._control_dt
        return Action(
            joint_cmds=[
                make_elbow_cmd(LEFT_ELBOW, elbow_q),
                make_elbow_cmd(RIGHT_ELBOW, elbow_q),
            ]
        )

    def reset(self) -> None:
        self._time_s = 0.0
