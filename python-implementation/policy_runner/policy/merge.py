"""Merge sparse Actions from multiple policies."""

from __future__ import annotations

from typing import Iterable

from policy_runner.types import Action, JointCommand


def merge_actions(actions: Iterable[Action]) -> Action:
    """Combine sparse actions. Later policies override earlier ones on conflicts."""
    by_index: dict[int, JointCommand] = {}
    for action in actions:
        for cmd in action.joint_cmds:
            by_index[cmd.index] = cmd
    return Action(joint_cmds=list(by_index.values()))
