"""Abstract policy interface (mirrors C++ policy.hpp)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from policy_runner.types import Action, Observation, RobotState


class Policy(ABC):
    """
    Input shape differs per model (joints + optional extras).
    Output is always sparse joint commands — not necessarily all joints.
    """

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def input_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def controlled_joints(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def build_observation(
        self, state: RobotState, command: Sequence[float]
    ) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def infer(self, obs: Observation) -> Action:
        raise NotImplementedError

    def reset(self) -> None:
        pass
