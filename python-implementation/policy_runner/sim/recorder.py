"""Record MuJoCo sim-to-sim runs for offline debugging (npz + optional mp4)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mujoco
import numpy as np

from policy_runner.sim.mujoco_bridge import MujocoBridge
from policy_runner.types import Observation, RobotState


class SimRecorder:
    """Buffers per-step state; writes `.npz` and optionally an `.mp4`."""

    def __init__(
        self,
        bridge: MujocoBridge,
        out_dir: str | Path,
        *,
        record_video: bool = True,
        video_size: tuple[int, int] = (640, 480),
        fps: float = 50.0,
        run_name: Optional[str] = None,
    ) -> None:
        self.bridge = bridge
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_name = run_name or f"mujoco_{stamp}"
        self.fps = float(fps)
        self.record_video = bool(record_video)

        self._rows: Dict[str, List[np.ndarray]] = {
            "sim_t": [],
            "root_pos": [],
            "root_quat_wxyz": [],
            "q": [],
            "dq": [],
            "q_target": [],
            "tau": [],
            "gyro": [],
            "projected_gravity": [],
            "rpy": [],
            "obs": [],
        }

        self._renderer: Optional[mujoco.Renderer] = None
        self._video_writer = None
        self._video_path: Optional[Path] = None
        if self.record_video:
            self._setup_video(video_size)

    def _setup_video(self, video_size: tuple[int, int]) -> None:
        import imageio

        h, w = int(video_size[1]), int(video_size[0])
        self._renderer = mujoco.Renderer(self.bridge.model, height=h, width=w)
        self._video_path = self.out_dir / f"{self.run_name}.mp4"
        self._video_writer = imageio.get_writer(
            str(self._video_path),
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        # Camera tracks the robot.
        self._cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self._cam)
        self._cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self._cam.trackbodyid = self.bridge._trunk_body_id
        self._cam.distance = 2.5
        self._cam.elevation = -20
        self._cam.azimuth = 135

    def log(
        self,
        sim_t: float,
        state: RobotState,
        obs: Optional[Observation] = None,
    ) -> None:
        data = self.bridge.data
        self._rows["sim_t"].append(np.asarray([sim_t], dtype=np.float64))
        self._rows["root_pos"].append(data.qpos[0:3].copy())
        self._rows["root_quat_wxyz"].append(data.qpos[3:7].copy())
        self._rows["q"].append(np.asarray(state.q, dtype=np.float64))
        self._rows["dq"].append(np.asarray(state.dq, dtype=np.float64))
        self._rows["q_target"].append(self.bridge.q_target)
        self._rows["tau"].append(self.bridge.last_tau)
        self._rows["gyro"].append(np.asarray(state.imu.gyro, dtype=np.float64))
        self._rows["projected_gravity"].append(
            np.asarray(state.projected_gravity, dtype=np.float64)
        )
        self._rows["rpy"].append(np.asarray(state.imu.rpy, dtype=np.float64))
        if obs is not None:
            self._rows["obs"].append(np.asarray(obs.data, dtype=np.float64))
        else:
            self._rows["obs"].append(np.zeros(0, dtype=np.float64))

        if self._renderer is not None and self._video_writer is not None:
            self._renderer.update_scene(data, camera=self._cam)
            frame = self._renderer.render()
            self._video_writer.append_data(frame)

    def close(self) -> Dict[str, Any]:
        npz_path = self.out_dir / f"{self.run_name}.npz"
        payload: Dict[str, Any] = {}
        for key, values in self._rows.items():
            if not values:
                continue
            # obs may be empty arrays if never set — skip stacking empties of mixed size
            if key == "obs" and (len(values[0]) == 0):
                continue
            payload[key] = np.stack(values, axis=0)

        payload["control_dt"] = np.asarray(
            [1.0 / self.fps if self.fps > 0 else 0.02], dtype=np.float64
        )
        np.savez_compressed(npz_path, **payload)

        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        return {
            "npz": str(npz_path),
            "video": str(self._video_path) if self._video_path else None,
            "steps": int(payload.get("sim_t", np.zeros(0)).shape[0]),
        }
