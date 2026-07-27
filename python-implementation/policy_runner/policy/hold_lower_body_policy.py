"""Hold lower-body joints at the pose latched when control starts."""

from __future__ import annotations

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

# Lower body: left leg 10-15, right leg 16-21.
_LOWER_BODY_JOINTS = list(
    range(int(JointIndex.LEFT_HIP_PITCH), int(JointIndex.RIGHT_ANKLE_ROLL) + 1)
)


class HoldLowerBodyPolicy(Policy):
    """Hold legs at start pose. Sparse — upper body stays weight=0."""

    def __init__(self, control_dt: float = 0.02) -> None:
        del control_dt  # unused; kept for factory API consistency
        self._hold_captured = False
        self._hold_q: Optional[List[float]] = None

    def name(self) -> str:
        return "hold_lower"

    def input_dim(self) -> int:
        return BASE_OBS_DIM

    def controlled_joints(self) -> List[int]:
        return list(_LOWER_BODY_JOINTS)

    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        return pack_observation(state, command)

    def infer(self, obs: Observation) -> Action:
        self.assert_frame_observation(obs)

        # Latch joint positions only (not IMU / command extras).
        if not self._hold_captured:
            self._hold_q = list(obs.data[:B1_JOINT_COUNT])
            self._hold_captured = True

        hold = self._hold_q or obs.data[:B1_JOINT_COUNT]
        return Action(
            joint_cmds=[make_joint_cmd(i, float(hold[i])) for i in _LOWER_BODY_JOINTS]
        )

    def reset(self) -> None:
        self._hold_captured = False
        self._hold_q = None
