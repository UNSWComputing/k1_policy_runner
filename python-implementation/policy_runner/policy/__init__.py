from policy_runner.policy.base import Policy
from policy_runner.policy.hold_lower_body_policy import HoldLowerBodyPolicy
from policy_runner.policy.merge import merge_actions
from policy_runner.policy.obs_history import ObservationHistory
from policy_runner.policy.sine_ankle_policy import SineAnklePolicy
from policy_runner.policy.sine_arm_policy import SineArmPolicy
from policy_runner.policy.sine_hip_policy import SineHipPolicy
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

try:
    from policy_runner.policy.walk_policy_v3 import WalkPolicyV3
except ImportError:  # pragma: no cover
    WalkPolicyV3 = None  # type: ignore

try:
    from policy_runner.policy.walk_policy_v4 import WalkPolicyV4
except ImportError:  # pragma: no cover
    WalkPolicyV4 = None  # type: ignore

try:
    from policy_runner.policy.walk_policy_v5 import WalkPolicyV5
except ImportError:  # pragma: no cover
    WalkPolicyV5 = None  # type: ignore

__all__ = [
    "Policy",
    "HoldLowerBodyPolicy",
    "ObservationHistory",
    "SineAnklePolicy",
    "SineArmPolicy",
    "SineHipPolicy",
    "SineKneePolicy",
    "StepAnklePolicy",
    "StepArmPolicy",
    "WalkPolicy",
    "WalkPolicyV1",
    "WalkPolicyV2",
    "WalkPolicyV3",
    "WalkPolicyV4",
    "WalkPolicyV5",
    "merge_actions",
]
