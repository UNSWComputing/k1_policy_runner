# Python Policy Runner (ROS2)

Same logic as the C++ pipeline, but I/O goes through ROS 2:

| Direction | Topic | Type |
|---|---|---|
| Input | `/joint_states` | `sensor_msgs/msg/JointState` |
| Input | `/low_state` | `booster_interface/msg/LowState` (IMU: rpy, gyro, acc) |
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
python3 policy_runner_main.py hold_lower
python3 policy_runner_main.py sine_arm,hold_lower   # arms move, legs hold
python3 policy_runner_main.py walk_v1
```

Optional topic overrides:

```bash
python3 policy_runner_main.py sine_arm,sine_knee \
  --joint-states-topic /joint_states \
  --joint-ctrl-topic /joint_ctrl
```

### Walk v1 + recording

Requires `onnxruntime` (see `requirements.txt`). Default model: repo-root `k1_v24_model_105600.onnx`.

```bash
# Record 195-D model inputs while running (saved on Ctrl+C)
python3 policy_runner_main.py walk_v1 --record-obs ../runs/walk_obs

# Optional: choose ONNX weights
python3 policy_runner_main.py walk_v1 \
  --model-path ../k1_v24_model_105600.onnx \
  --record-obs ../runs/walk_obs
```

Outputs:

| File | Contents |
|---|---|
| `runs/walk_obs.npz` | `obs` `[T, 195]`, `timestamps` `[T]`, layout `meta` |
| `runs/walk_obs.json` | Human-readable layout (term names / sizes / history) |

Recording starts only after the settle phase ends (RL steps). Inspect / plot from the **repo root**:

```bash
python3 joint_record.py runs/walk_obs.npz
python3 joint_record.py runs/walk_obs.npz --term joint_pos
python3 joint_record.py runs/walk_obs.npz --term actions
python3 joint_record.py runs/walk_obs.npz --plot
python3 joint_record.py runs/walk_obs.npz --plot --save runs/walk --no-show
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
| `hold_lower` | Hold legs at start pose | Lower body (10–21) |
| `walk` | Stub history example | Lower body (10–21) |
| `walk_v1` | ONNX walk: 5s settle to default pose, then RL (legs) + hold upper body | Full body (0–21) |

Policies emit **sparse** actions. Pass several comma-separated names to run them in parallel; actions are merged by joint index (later policy wins on conflicts). Unowned joints stay `weight = 0`.

Default observation layout: `[joint_q (22), imu_rpy/gyro/acc (9), projected_gravity (3)]` (= 34 dims), plus any optional command extras.

**History / dims:** `observation_dim` is one frame from `build_observation`; `input_dim` is what the model consumes (may be `observation_dim * history_len`). Keep history inside the policy and check frames with `assert_frame_observation` / `ObservationHistory` — see `walk_policy.py`.

`/joint_states` is mapped by joint **name** using `JOINT_NAMES` in `joint_index.py`; otherwise positions are assumed to be in `JointIndex` order.
