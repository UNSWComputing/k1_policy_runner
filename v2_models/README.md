# Walk v2 models

Default: `model_31150.onnx`

## Frame layout (61) — no head joints

| Term | Dim | Notes |
|------|-----|--------|
| base_ang_vel | 3 | IMU gyro |
| projected_gravity | 3 | from IMU attitude |
| joint_pos | 20 | arms+legs, `q − DEFAULT` (no head) |
| joint_vel | 20 | same joints |
| actions | 12 | last policy emit (clamped, relative) |
| command | 3 | twist (vx, vy, ωz) |

## Model input

`61 × history 3 = 183`, term-major (per term: `[t-2 | t-1 | t]`), same stacking as walk_v1.

## Output

`actions` → `[1, 12]` leg targets (relative); runner applies default offset + clamp.

## Run

```bash
cd python-implementation
python3 policy_runner_main.py walk_v2
python3 policy_runner_main.py walk_v2 --model-path ../v2_models/model_31150.onnx
```
