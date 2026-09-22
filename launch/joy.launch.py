"""Launch joy_node + teleop_twist_joy for policy_runner walk commands.

joy_node publishes sensor_msgs/Joy on policy_runner/joy.
teleop_twist_joy converts that to geometry_msgs/Twist on /cmd_vel.

  ros2 launch launch/joy.launch.py
  ros2 launch launch/joy.launch.py joy_dev:=0

Button indices and max speeds live in config/joy.yaml.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = str(_REPO_ROOT / "config" / "joy.yaml")


def generate_launch_description() -> LaunchDescription:
    joy_dev = LaunchConfiguration("joy_dev")
    joy_topic = LaunchConfiguration("joy_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    config_file = LaunchConfiguration("config_file")
    publish_stamped_twist = LaunchConfiguration("publish_stamped_twist")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "joy_dev",
                default_value="0",
                description="SDL joystick device_id (ros2 run joy joy_enumerate_devices)",
            ),
            DeclareLaunchArgument(
                "joy_topic",
                default_value="policy_runner/joy",
                description="Joy topic published by joy_node and read by teleop",
            ),
            DeclareLaunchArgument(
                "cmd_vel_topic",
                default_value="/cmd_vel",
                description="Twist topic for walk velocity (policy_runner default)",
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=_DEFAULT_CONFIG,
                description="YAML with button/axis indices and max speeds",
            ),
            DeclareLaunchArgument(
                "publish_stamped_twist",
                default_value="false",
                description="If true, teleop publishes TwistStamped instead of Twist",
            ),
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node",
                output="screen",
                parameters=[
                    config_file,
                    {"device_id": joy_dev},
                ],
                remappings=[("joy", joy_topic)],
            ),
            Node(
                package="teleop_twist_joy",
                executable="teleop_node",
                name="teleop_twist_joy_node",
                output="screen",
                parameters=[
                    config_file,
                    {"publish_stamped_twist": publish_stamped_twist},
                ],
                remappings=[
                    ("joy", joy_topic),
                    ("cmd_vel", cmd_vel_topic),
                ],
            ),
        ]
    )
