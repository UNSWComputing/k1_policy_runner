"""Abstract policy interface (mirrors C++ policy.hpp)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from policy_runner.types import Action, Observation, RobotState


class Policy(ABC):
    """
    Input shape differs per model (joints + optional extras).
    Output is always sparse joint commands — not necessarily all joints.

    Dimensionality:
      - observation_dim(): size of one frame from build_observation()
      - input_dim(): size consumed by the model / Infer after any history stacking
      - history_len(): number of frames kept inside the policy (default 1)

    History buffers live inside the policy. The runner only passes one frame
    Observation per step; policies that need history append/check internally.
    """

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def input_dim(self) -> int:
        """Model / Infer input size (may be observation_dim * history_len)."""
        raise NotImplementedError

    def observation_dim(self) -> int:
        """Single-frame observation size from build_observation(). Default: no history."""
        return self.input_dim()

    def history_len(self) -> int:
        """Number of frames this policy stacks. Default: 1 (no history)."""
        return 1

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

    def assert_frame_observation(self, obs: Observation) -> None:
        """Sanity-check one frame from build_observation (not the stacked model input)."""
        got = len(obs.data)
        expected = self.observation_dim()
        if got != expected:
            raise ValueError(
                f"{self.name()}: frame observation dim {got} != "
                f"observation_dim {expected} "
                f"(input_dim={self.input_dim()}, history_len={self.history_len()})"
            )
