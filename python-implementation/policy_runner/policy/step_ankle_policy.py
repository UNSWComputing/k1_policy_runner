"""Hard-coded step-function ankle pitch/roll policy."""

from __future__ import annotations

from typing import List, Sequence

from policy_runner.joint_index import JointIndex
from policy_runner.policy.base import Policy
from policy_runner.types import (
    BASE_OBS_DIM,
    Action,
    JointCommand,
    Observation,
    RobotState,
    pack_observation,
)

LEFT_ANKLE_PITCH = int(JointIndex.LEFT_ANKLE_PITCH)
LEFT_ANKLE_ROLL = int(JointIndex.LEFT_ANKLE_ROLL)
RIGHT_ANKLE_PITCH = int(JointIndex.RIGHT_ANKLE_PITCH)
RIGHT_ANKLE_ROLL = int(JointIndex.RIGHT_ANKLE_ROLL)

# Absolute rad limits for the step extremes.
_PITCH_A = -0.8
_PITCH_B = 0.35
_ROLL_A = -0.345
_ROLL_B = 0.345
_PITCH_CENTER = 0.5 * (_PITCH_A + _PITCH_B)
_ROLL_CENTER = 0.5 * (_ROLL_A + _ROLL_B)

# Match walk_v1 ankle PD for this test.
_ANKLE_KP = 25.0
_ANKLE_KD = 1.0

# Which DOF to step: "pitch" | "roll". The other is held at range center.
_TEST_AXIS = "pitch"


def _make_ankle_cmd(index: int, q: float) -> JointCommand:
    return JointCommand(
        index=index,
        q=q,
        dq=0.0,
        tau=0.0,
        kp=_ANKLE_KP,
        kd=_ANKLE_KD,
        weight=1.0,
    )


class StepAnklePolicy(Policy):
    """Step one ankle axis: low → mid → high; hold the other centered."""

    def __init__(
        self,
        control_dt: float = 0.02,
        period_s: float = 10.0,
        axis: str = _TEST_AXIS,
    ) -> None:
        if axis not in ("pitch", "roll"):
            raise ValueError(f"step_ankle axis must be 'pitch' or 'roll', got {axis!r}")
        self._control_dt = control_dt
        self._period_s = period_s
        self._axis = axis
        self._time_s = 0.0

    def name(self) -> str:
        return "step_ankle"

    def input_dim(self) -> int:
        return BASE_OBS_DIM

    def controlled_joints(self) -> List[int]:
        return [
            LEFT_ANKLE_PITCH,
            LEFT_ANKLE_ROLL,
            RIGHT_ANKLE_PITCH,
            RIGHT_ANKLE_ROLL,
        ]

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        return pack_observation(state, command)

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        # Cycle: low limit → mid → high limit → …
        phase = int(self._time_s / self._period_s) % 3
        if self._axis == "pitch":
            pitch_levels = (_PITCH_A, _PITCH_CENTER, _PITCH_B)
            pitch_q = pitch_levels[phase]
            roll_q = _ROLL_CENTER
        else:
            roll_levels = (_ROLL_A, _ROLL_CENTER, _ROLL_B)
            pitch_q = _PITCH_CENTER
            roll_q = roll_levels[phase]

        self._time_s += self._control_dt
        return Action(
            joint_cmds=[
                _make_ankle_cmd(LEFT_ANKLE_PITCH, pitch_q),
                _make_ankle_cmd(LEFT_ANKLE_ROLL, roll_q),
                _make_ankle_cmd(RIGHT_ANKLE_PITCH, pitch_q),
                _make_ankle_cmd(RIGHT_ANKLE_ROLL, roll_q),
            ]
        )

    def reset(self) -> None:
        self._time_s = 0.0
