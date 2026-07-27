"""Joint indices and canonical /joint_states names (22-DoF, no waist)."""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List


class JointIndex(IntEnum):
    HEAD_YAW = 0
    HEAD_PITCH = 1
    LEFT_SHOULDER_PITCH = 2
    LEFT_SHOULDER_ROLL = 3
    LEFT_ELBOW_PITCH = 4
    LEFT_ELBOW_YAW = 5
    RIGHT_SHOULDER_PITCH = 6
    RIGHT_SHOULDER_ROLL = 7
    RIGHT_ELBOW_PITCH = 8
    RIGHT_ELBOW_YAW = 9
    LEFT_HIP_PITCH = 10
    LEFT_HIP_ROLL = 11
    LEFT_HIP_YAW = 12
    LEFT_KNEE_PITCH = 13
    LEFT_ANKLE_PITCH = 14
    LEFT_ANKLE_ROLL = 15
    RIGHT_HIP_PITCH = 16
    RIGHT_HIP_ROLL = 17
    RIGHT_HIP_YAW = 18
    RIGHT_KNEE_PITCH = 19
    RIGHT_ANKLE_PITCH = 20
    RIGHT_ANKLE_ROLL = 21


# Canonical names in JointIndex order — must match /joint_states exactly.
JOINT_NAMES: List[str] = [
    "AAHead_yaw",
    "Head_pitch",
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def joint_name_to_index() -> Dict[str, int]:
    return {name: i for i, name in enumerate(JOINT_NAMES)}
