"""ROS2 robot bridge: /joint_states in, /joint_ctrl (LowCmd) out."""

from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from booster_interface.msg import LowCmd, MotorCmd

from policy_runner.joint_index import joint_name_to_index
from policy_runner.types import B1_JOINT_COUNT, Action, RobotState


class RobotBridge(Node):
    """
    Thin ROS2 bridge mirroring the C++ RobotBridge API.

    Subscribes: /joint_states (sensor_msgs/JointState)
    Publishes:  /joint_ctrl   (booster_interface/msg/LowCmd)
    """

    def __init__(
        self,
        joint_state_topic: str = "/joint_states",
        joint_ctrl_topic: str = "/joint_ctrl",
    ) -> None:
        super().__init__("policy_runner")

        self._state_lock = threading.Lock()
        self._latest_state = RobotState(
            q=[0.0] * B1_JOINT_COUNT,
            dq=[0.0] * B1_JOINT_COUNT,
        )
        self._has_state = False
        self._name_to_index = joint_name_to_index()

        self._pub = self.create_publisher(LowCmd, joint_ctrl_topic, 10)
        self._sub = self.create_subscription(
            JointState, joint_state_topic, self._on_joint_state, 10
        )

        self.get_logger().info(
            f"Subscribed to {joint_state_topic}, publishing to {joint_ctrl_topic}"
        )

    def has_state(self) -> bool:
        return self._has_state

    def latest_state(self) -> RobotState:
        with self._state_lock:
            return RobotState(
                q=list(self._latest_state.q),
                dq=list(self._latest_state.dq),
            )

    def publish_action(self, action: Action) -> None:
        """Write sparse Action onto a full LowCmd. Uncontrolled joints get weight=0."""
        msg = LowCmd()
        msg.cmd_type = getattr(LowCmd, "CMD_TYPE_PARALLEL", 0)
        msg.motor_cmd = [MotorCmd() for _ in range(B1_JOINT_COUNT)]

        for m in msg.motor_cmd:
            m.mode = 0
            m.q = 0.0
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0
            m.weight = 0.0

        for jc in action.joint_cmds:
            if jc.index < 0 or jc.index >= B1_JOINT_COUNT:
                raise ValueError(f"PublishAction: joint index out of range: {jc.index}")
            m = msg.motor_cmd[jc.index]
            m.q = float(jc.q)
            m.dq = float(jc.dq)
            m.tau = float(jc.tau)
            m.kp = float(jc.kp)
            m.kd = float(jc.kd)
            m.weight = float(jc.weight)

        self._pub.publish(msg)

    def _on_joint_state(self, msg: JointState) -> None:
        q = [0.0] * B1_JOINT_COUNT
        dq = [0.0] * B1_JOINT_COUNT

        if msg.name:
            for i, name in enumerate(msg.name):
                idx = self._name_to_index.get(name)
                if idx is None:
                    continue
                if i < len(msg.position):
                    q[idx] = float(msg.position[i])
                if i < len(msg.velocity):
                    dq[idx] = float(msg.velocity[i])
        else:
            # Positional fallback: assume JointIndex order.
            n = min(len(msg.position), B1_JOINT_COUNT)
            for i in range(n):
                q[i] = float(msg.position[i])
            n_dq = min(len(msg.velocity), B1_JOINT_COUNT)
            for i in range(n_dq):
                dq[i] = float(msg.velocity[i])

        with self._state_lock:
            self._latest_state = RobotState(q=q, dq=dq)
            self._has_state = True


def spin_bridge_in_background(bridge: RobotBridge) -> threading.Thread:
    """Spin the ROS node on a daemon thread so the control loop can run in main."""

    def _spin() -> None:
        rclpy.spin(bridge)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread
