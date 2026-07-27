"""ROS2 robot bridge: /joint_states + /low_state (IMU) in, /joint_ctrl out."""

from __future__ import annotations

import threading
from typing import Sequence

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from booster_interface.msg import LowCmd, LowState, MotorCmd

from policy_runner.joint_index import joint_name_to_index
from policy_runner.types import (
    B1_JOINT_COUNT,
    Action,
    ImuState,
    RobotState,
    compute_projected_gravity,
)


def _as_vec3(values: Sequence[float]) -> list[float]:
    out = [0.0, 0.0, 0.0]
    n = min(3, len(values))
    for i in range(n):
        out[i] = float(values[i])
    return out


class RobotBridge(Node):
    """
    Thin ROS2 bridge mirroring the C++ RobotBridge API.

    Subscribes: /joint_states (sensor_msgs/JointState)
                /low_state    (booster_interface/msg/LowState) for IMU
    Publishes:  /joint_ctrl   (booster_interface/msg/LowCmd)
    """

    def __init__(
        self,
        joint_state_topic: str = "/joint_states",
        joint_ctrl_topic: str = "/joint_ctrl",
        low_state_topic: str = "/low_state",
    ) -> None:
        super().__init__("policy_runner")

        self._state_lock = threading.Lock()
        self._latest_state = RobotState(
            q=[0.0] * B1_JOINT_COUNT,
            dq=[0.0] * B1_JOINT_COUNT,
            imu=ImuState(),
            projected_gravity=[0.0, 0.0, -1.0],
        )
        self._has_joint_state = False
        self._has_imu = False
        self._name_to_index = joint_name_to_index()

        self._pub = self.create_publisher(LowCmd, joint_ctrl_topic, 10)
        self._joint_sub = self.create_subscription(
            JointState, joint_state_topic, self._on_joint_state, 10
        )
        self._low_state_sub = self.create_subscription(
            LowState, low_state_topic, self._on_low_state, 10
        )

        self.get_logger().info(
            f"Subscribed to {joint_state_topic} and {low_state_topic}, "
            f"publishing to {joint_ctrl_topic}"
        )

    def has_state(self) -> bool:
        return self._has_joint_state and self._has_imu

    def latest_state(self) -> RobotState:
        with self._state_lock:
            imu = self._latest_state.imu
            return RobotState(
                q=list(self._latest_state.q),
                dq=list(self._latest_state.dq),
                imu=ImuState(
                    rpy=list(imu.rpy),
                    gyro=list(imu.gyro),
                    acc=list(imu.acc),
                ),
                projected_gravity=list(self._latest_state.projected_gravity),
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
            n = min(len(msg.position), B1_JOINT_COUNT)
            for i in range(n):
                q[i] = float(msg.position[i])
            n_dq = min(len(msg.velocity), B1_JOINT_COUNT)
            for i in range(n_dq):
                dq[i] = float(msg.velocity[i])

        with self._state_lock:
            self._latest_state.q = q
            self._latest_state.dq = dq
            self._has_joint_state = True

    def _on_low_state(self, msg: LowState) -> None:
        imu_msg = msg.imu_state
        imu = ImuState(
            rpy=_as_vec3(imu_msg.rpy),
            gyro=_as_vec3(imu_msg.gyro),
            acc=_as_vec3(imu_msg.acc),
        )
        projected_gravity = compute_projected_gravity(imu.rpy)
        with self._state_lock:
            self._latest_state.imu = imu
            self._latest_state.projected_gravity = projected_gravity
            self._has_imu = True


def spin_bridge_in_background(bridge: RobotBridge) -> threading.Thread:
    """Spin the ROS node on a daemon thread so the control loop can run in main."""

    def _spin() -> None:
        rclpy.spin(bridge)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread
