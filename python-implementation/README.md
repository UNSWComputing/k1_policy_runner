# Python Policy Runner (ROS2)

Same logic as the C++ pipeline, but I/O goes through ROS 2:

| Direction | Topic | Type |
|---|---|---|
| Input | `/joint_states` | `sensor_msgs/msg/JointState` |
| Output | `/joint_ctrl` | `booster_interface/msg/LowCmd` |

## Requirements

- Python 3.10+
- ROS 2 (Humble or later) with `rclpy`, `sensor_msgs`
- `booster_interface` ROS 2 package (provides `LowCmd` / `MotorCmd`)

Source your ROS 2 workspace before running:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
# and your workspace overlay that contains booster_interface, if needed
source /path/to/ws/install/setup.bash
```

## Layout

```
python-implementation/
  policy_runner/
    types.py
    joint_index.py
    policy/          # Policy interface + sine/step demos
    robot/           # RobotBridge (ROS2)
  policy_runner_main.py
```

## Run

From `python-implementation/`:

```bash
python3 policy_runner_main.py sine_arm
python3 policy_runner_main.py sine_knee
python3 policy_runner_main.py sine_arm,sine_knee   # parallel, merged
```

Optional topic overrides:

```bash
python3 policy_runner_main.py sine_arm,sine_knee \
  --joint-states-topic /joint_states \
  --joint-ctrl-topic /joint_ctrl
```

### Safety sequence

1. Put the robot in **Prepare** mode.
2. Start the script; wait until `/joint_states` is received.
3. Press **ENTER** to begin publishing `/joint_ctrl`.
4. Switch the robot to **Custom** mode.
5. Stop with Ctrl+C.

## Policies

| Name | Behavior | Animated joints |
|---|---|---|
| `sine_arm` | Elbows follow a sine wave | Left + right elbow (5, 9) |
| `step_arm` | Elbows step between two angles | Left + right elbow (5, 9) |
| `sine_knee` | Knee pitches follow a sine wave | Left + right knee pitch (13, 19) |

Policies emit **sparse** actions. Pass several comma-separated names to run them in parallel; actions are merged by joint index (later policy wins on conflicts). Unowned joints stay `weight = 0`.

`/joint_states` is mapped by joint **name** using `JOINT_NAMES` in `joint_index.py`; otherwise positions are assumed to be in `JointIndex` order.
