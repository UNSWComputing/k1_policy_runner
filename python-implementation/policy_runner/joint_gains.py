"""Default PD gains for hard-coded policies.

From README K1_CFG joint_stiffness / joint_damping (22-DoF, no waist).
"""

from __future__ import annotations

from policy_runner.types import B1_JOINT_COUNT, JointCommand

DEFAULT_KP = (
    4.0, 4.0,
    4.0, 4.0, 4.0, 4.0,
    4.0, 4.0, 4.0, 4.0,
    80.0, 80.0, 80.0, 80.0, 30.0, 30.0,
    80.0, 80.0, 80.0, 80.0, 30.0, 30.0,
)

DEFAULT_KD = (
    1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
)

assert len(DEFAULT_KP) == B1_JOINT_COUNT
assert len(DEFAULT_KD) == B1_JOINT_COUNT


def make_joint_cmd(index: int, q: float) -> JointCommand:
    return JointCommand(
        index=index,
        q=q,
        dq=0.0,
        tau=0.0,
        kp=DEFAULT_KP[index],
        kd=DEFAULT_KD[index],
        weight=1.0,
    )
