"""Hard-coded step-function elbow policy (mirrors C++ StepArmPolicy)."""

from __future__ import annotations

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

_ELBOW_A = 0.0
_ELBOW_B = -0.6


class StepArmPolicy(Policy):
    """Elbows (5, 9) step between two angles. Sparse output — other joints weight=0."""

    def __init__(self, control_dt: float = 0.02, period_s: float = 2.0) -> None:
        self._control_dt = control_dt
        self._period_s = period_s
        self._time_s = 0.0

    def name(self) -> str:
        return "step_arm"

    def input_dim(self) -> int:
        return BASE_OBS_DIM

    def controlled_joints(self) -> List[int]:
        return [LEFT_ELBOW, RIGHT_ELBOW]

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        return pack_observation(state, command)

    def infer(self, obs: Observation) -> Action:
        if len(obs.data) != self.input_dim():
            raise ValueError("StepArmPolicy: observation size mismatch")

        use_b = int(self._time_s / self._period_s) % 2 == 1
        elbow_q = _ELBOW_B if use_b else _ELBOW_A

        self._time_s += self._control_dt
        return Action(
            joint_cmds=[
                make_elbow_cmd(LEFT_ELBOW, elbow_q),
                make_elbow_cmd(RIGHT_ELBOW, elbow_q),
            ]
        )

    def reset(self) -> None:
        self._time_s = 0.0
