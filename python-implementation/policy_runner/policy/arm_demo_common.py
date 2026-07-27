"""Shared helpers for elbow-only arm demos (mirrors C++ arm_demo_common.hpp)."""

from __future__ import annotations

from policy_runner.joint_gains import make_joint_cmd
from policy_runner.joint_index import JointIndex
from policy_runner.types import JointCommand

# Elbow joints per robot joint table (README): indices 5 and 9.
LEFT_ELBOW = int(JointIndex.LEFT_ELBOW_YAW)  # 5
RIGHT_ELBOW = int(JointIndex.RIGHT_ELBOW_YAW)  # 9


def make_elbow_cmd(index: int, q: float) -> JointCommand:
    return make_joint_cmd(index, q)
