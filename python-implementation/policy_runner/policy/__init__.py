from policy_runner.policy.base import Policy
from policy_runner.policy.hold_lower_body_policy import HoldLowerBodyPolicy
from policy_runner.policy.merge import merge_actions
from policy_runner.policy.obs_history import ObservationHistory
from policy_runner.policy.sine_arm_policy import SineArmPolicy
from policy_runner.policy.sine_knee_policy import SineKneePolicy
from policy_runner.policy.step_arm_policy import StepArmPolicy

try:
    from policy_runner.policy.step_ankle_policy import StepAnklePolicy
except ImportError:  # pragma: no cover
    StepAnklePolicy = None  # type: ignore

try:
    from policy_runner.policy.walk_policy import WalkPolicy
except ImportError:  # pragma: no cover
    WalkPolicy = None  # type: ignore

try:
    from policy_runner.policy.walk_policy_v1 import WalkPolicyV1
except ImportError:  # pragma: no cover
    WalkPolicyV1 = None  # type: ignore

try:
    from policy_runner.policy.walk_policy_v2 import WalkPolicyV2
except ImportError:  # pragma: no cover
    WalkPolicyV2 = None  # type: ignore

__all__ = [
    "Policy",
    "HoldLowerBodyPolicy",
    "ObservationHistory",
    "SineArmPolicy",
    "SineKneePolicy",
    "StepAnklePolicy",
    "StepArmPolicy",
    "WalkPolicy",
    "WalkPolicyV1",
    "WalkPolicyV2",
    "merge_actions",
]
