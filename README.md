# Policy Runner

Pipeline to run control policies on the Booster B1.

- **C++** — Booster low-level SDK (`LowState` / `LowCmd` DDS)
- **Python** — ROS 2 (`/joint_states` → `/joint_ctrl`); see [`python-implementation/README.md`](python-implementation/README.md)

## Requirements

- CMake ≥ 3.16
- C++17 compiler
- [Booster Robotics SDK](https://github.com/BoosterRobotics/booster_robotics_sdk) (headers + `libbooster_robotics_sdk`)

Install the SDK once (copies headers/libs into `/usr/local`):

```bash
cd /path/to/booster_robotics_sdk
sudo ./install.sh
```

## Build (C++)

From the repo root:

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

If the SDK is **not** installed system-wide, point CMake at the SDK checkout:

```bash
cmake .. -DBOOSTER_SDK_DIR=/path/to/booster_robotics_sdk
cmake --build .
```

`BOOSTER_SDK_DIR` must contain `include/` and `lib/<arch>/libbooster_robotics_sdk.a`.

## Run (C++)

```bash
./policy_runner <policies> [networkInterface]
```

| Argument | Description |
|---|---|
| `policies` | Comma-separated: `sine_arm`, `step_arm`, `sine_knee`, `hold_lower` |
| `networkInterface` | Optional NIC name (same as the SDK examples). Omit to use `Init(0)`. |

Examples:

```bash
./policy_runner sine_arm
./policy_runner sine_knee
./policy_runner hold_lower
./policy_runner sine_arm,hold_lower       # arms move, legs hold
./policy_runner sine_arm,sine_knee eth0
```

### Safety sequence

Same as `example/b1_low_sdk_example.cpp`:

1. Put the robot in **Prepare** mode.
2. Start `policy_runner`.
3. Wait until it reports that `LowState` is available.
4. Press **ENTER** to begin publishing commands.
5. Switch the robot to **Custom** mode (API or controller).
6. Stop with Ctrl+C when done.

## Run (Python / ROS2)

```bash
cd python-implementation
source /opt/ros/$ROS_DISTRO/setup.bash   # + booster_interface overlay if needed
python3 policy_runner_main.py sine_arm
python3 policy_runner_main.py sine_knee
python3 policy_runner_main.py hold_lower
python3 policy_runner_main.py sine_arm,hold_lower   # arms move, legs hold
```

Topics: `/joint_states` + `/low_state` (IMU) → `/joint_ctrl` `booster_interface/msg/LowCmd`.  
Details: [`python-implementation/README.md`](python-implementation/README.md).

## Policies

| Name | Behavior | Animated joints |
|---|---|---|
| `sine_arm` | Elbows follow a sine wave | Left + right elbow (5, 9) |
| `step_arm` | Elbows step between two angles | Left + right elbow (5, 9) |
| `sine_knee` | Knee pitches follow a sine wave | Left + right knee pitch (13, 19) |
| `hold_lower` | Hold legs at start pose | Lower body (10–21) |

Policies emit **sparse** actions. Pass several comma-separated names to run them in parallel; actions are merged by joint index (later policy wins on conflicts). Unowned joints stay `weight = 0`.

Default observation layout: `[joint_q (22), imu_rpy/gyro/acc (9), projected_gravity (3)]` (= 34 dims), plus any optional command extras.

## Layout

```
src/include/
  types.hpp              # RobotState, Observation, Action
  joint_index.hpp
  policy/                # Policy interface + demos
  robot/                 # RobotBridge (DDS I/O)
src/policy/              # Policy implementations
src/robot/               # RobotBridge implementation
src/policy_runner.cpp    # Main control loop
python-implementation/   # ROS2 Python port
example/                 # Upstream SDK usage samples
```

## Robot joints (reference)

Limits below are degrees (max / min). See `src/include/joint_index.hpp` (22-DoF, no waist).

| Index | Joint | Max | Min |
|---|---|---|---|
| 0 | Head Yaw | 59 | -59 |
| 1 | Head Pitch | 49 | -19 |
| 2 | Left Shoulder Pitch | 69 | -169 |
| 3 | Left Shoulder Roll | 94 | -94 |
| 4 | Left Shoulder Yaw | 109 | -109 |
| 5 | Left Elbow | 39 | -129 |
| 6 | Right Shoulder Pitch | 69 | -169 |
| 7 | Right Shoulder Roll | 94 | -94 |
| 8 | Right Shoulder Yaw | 109 | -109 |
| 9 | Right Elbow | 129 | -39 |
| 10 | Left Hip Pitch | 128 | -170 |
| 11 | Left Hip Roll | 89 | -22 |
| 12 | Left Hip Yaw | 59 | -59 |
| 13 | Left Knee | 133 | 0 |
| 14 | Left Ankle Up | 38 | -17 |
| 15 | Left Ankle Down | 41 | -16 |
| 16 | Right Hip Pitch | 128 | -170 |
| 17 | Right Hip Roll | 22 | -89 |
| 18 | Right Hip Yaw | 59 | -59 |
| 19 | Right Knee | 133 | 0 |
| 20 | Right Ankle Up | 38 | -17 |
| 21 | Right Ankle Down | 41 | -16 |


example stiffness and damp

K1_CFG = RobotCfg(
    name="Booster_K1",
    joint_names=[
        "AAHead_yaw",
        "Head_pitch",
        "ALeft_Shoulder_Pitch",
        "Left_Shoulder_Roll",
        "Left_Elbow_Pitch",
        "Left_Elbow_Yaw",
        "ARight_Shoulder_Pitch",
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
    ],
    body_names=[
        "Trunk",
        "Head_1",
        "Head_2",
        "Left_Arm_1",
        "Left_Arm_2",
        "Left_Arm_3",
        "left_hand_link",
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "Left_Hip_Pitch",
        "Left_Hip_Roll",
        "Left_Hip_Yaw",
        "Left_Shank",
        "Left_Ankle_Cross",
        "left_foot_link",
        "Right_Hip_Pitch",
        "Right_Hip_Roll",
        "Right_Hip_Yaw",
        "Right_Shank",
        "Right_Ankle_Cross",
        "right_foot_link",
    ],
    joint_stiffness=[
        4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        4.0, 4.0, 4.0, 4.0,
        80., 80.0, 80., 80., 30., 30.,
        80., 80.0, 80., 80., 30., 30.,
    ],
    joint_damping=[
        1., 1.,
        1., 1., 1., 1.,
        1., 1., 1., 1.,
        2., 2., 2., 2., 2., 2.,
        2., 2., 2., 2., 2., 2.,
    ],
    default_joint_pos=[
        0, 0,
        0.0, -1.3, 0, -0.,
        0.0, 1.3, 0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.,
        -0.0, 0, 0, 0.0, -0.0, 0.
    ],
    effort_limit=[
        6, 6,
        14, 14, 14, 14,
        14, 14, 14, 14,
        30, 35, 20, 40, 20, 20,
        30, 35, 20, 40, 20, 20,
    ],
    sim_joint_names=[       # joint order in isaacsim/isaaclab
        "AAHead_yaw",
        "ALeft_Shoulder_Pitch",
        "ARight_Shoulder_Pitch",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Head_pitch",
        "Left_Shoulder_Roll",
        "Right_Shoulder_Roll",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Elbow_Pitch",
        "Right_Elbow_Pitch",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "Left_Elbow_Yaw",
        "Right_Elbow_Yaw",
        "Left_Knee_Pitch",
        "Right_Knee_Pitch",
        "Left_Ankle_Pitch",
        "Right_Ankle_Pitch",
        "Left_Ankle_Roll",
        "Right_Ankle_Roll",
    ],
    sim_body_names=[    # body order in isaacsim/isaaclab
        "Trunk",
        "Head_1",
        "Left_Arm_1",
        "Right_Arm_1",
        "Left_Hip_Pitch",
        "Right_Hip_Pitch",
        "Head_2",
        "Left_Arm_2",
        "Right_Arm_2",
        "Left_Hip_Roll",
        "Right_Hip_Roll",
        "Left_Arm_3",
        "Right_Arm_3",
        "Left_Hip_Yaw",
        "Right_Hip_Yaw",
        "left_hand_link",
        "right_hand_link",
        "Left_Shank",
        "Right_Shank",
        "Left_Ankle_Cross",
        "Right_Ankle_Cross",
        "left_foot_link",
        "right_foot_link",
    ],
    # {BOOSTER_ASSETS_DIR} will be replaced with
    # booster_assets.BOOSTER_ASSETS_DIR by MujocoController
    mjcf_path="{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.xml",
    prepare_state=PrepareStateCfg(
        stiffness=[
            40., 40.,
            40., 50., 20., 20,
            40., 50., 20., 20,
            350., 350., 180., 350., 250., 250.,
            350., 350., 180., 350., 250., 250.,
        ],
        damping=[
            1.5, 1.5,
            0.5, 1.5, 0.2, 0.2,
            0.5, 1.5, 0.2, 0.2,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
            7.5, 7.5, 3., 5.5, 5.0, 5.0,
        ],
        joint_pos=[
            0, 0,
            0.0, -1.3, 0, -0.,
            0.0, 1.3, 0, 0.,
            -0.0, 0, 0, 0.105, -0.10, 0.,
            -0.0, 0, 0, 0.105, -0.10, 0.
        ],
    ),
)
