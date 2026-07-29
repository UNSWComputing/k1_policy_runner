#!/usr/bin/env python3
"""Log /joint_states + /joint_ctrl, or plot a saved log offline.

Log (ROS2):
  python3 joint_log.py
  python3 joint_log.py --output runs/joint_log

Plot offline (no flags — just the .npz path):
  python3 joint_log.py runs/joint_log.npz

Plot GUI: tick boxes select joints; solid = measured state_q, dashed = cmd_q.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PY = Path(__file__).resolve().parent / "python-implementation"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from policy_runner.joint_index import JOINT_NAMES, joint_name_to_index
from policy_runner.types import B1_JOINT_COUNT


# ---------------------------------------------------------------------------
# Offline load / plot
# ---------------------------------------------------------------------------


def load_joint_log(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if p.suffix == "":
        p = p.with_suffix(".npz")
    if not p.is_file():
        raise FileNotFoundError(f"joint log not found: {p}")

    data = np.load(p, allow_pickle=False)
    meta_raw = data["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_raw = meta_raw.item()
    meta = json.loads(str(meta_raw))
    names = [str(n) for n in data["joint_names"].tolist()]
    return {
        "path": p,
        "meta": meta,
        "joint_names": names,
        "state_t": np.asarray(data["state_t"], dtype=np.float64),
        "state_q": np.asarray(data["state_q"], dtype=np.float64),
        "state_dq": np.asarray(data["state_dq"], dtype=np.float64),
        "cmd_t": np.asarray(data["cmd_t"], dtype=np.float64),
        "cmd_q": np.asarray(data["cmd_q"], dtype=np.float64),
        "cmd_dq": np.asarray(data["cmd_dq"], dtype=np.float64),
        "cmd_tau": np.asarray(data["cmd_tau"], dtype=np.float64),
        "cmd_kp": np.asarray(data["cmd_kp"], dtype=np.float64),
        "cmd_kd": np.asarray(data["cmd_kd"], dtype=np.float64),
        "cmd_weight": np.asarray(data["cmd_weight"], dtype=np.float64),
    }


def plot_joint_log(path: Path) -> int:
    """Open a checkbox GUI for measured vs commanded joint positions."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons

    log = load_joint_log(path)
    names: List[str] = log["joint_names"]
    n = len(names)
    state_t = log["state_t"]
    state_q = log["state_q"]
    cmd_t = log["cmd_t"]
    cmd_q = log["cmd_q"]

    print(
        f"file: {log['path']}\n"
        f"states: {state_q.shape[0]}  cmds: {cmd_q.shape[0]}  "
        f"duration_s: {log['meta'].get('duration_s', 0):.2f}"
    )

    # Default: legs selected (indices 10–21), upper body off.
    active = [i >= 10 for i in range(n)]

    fig = plt.figure(figsize=(12, 7))
    fig.canvas.manager.set_window_title(f"joint_log — {log['path'].name}")
    ax = fig.add_axes([0.28, 0.12, 0.68, 0.78])
    rax = fig.add_axes([0.02, 0.08, 0.22, 0.84])
    rax.set_title("Joints", fontsize=10)
    rax.set_facecolor("#f5f5f5")

    colors = plt.cm.tab20(np.linspace(0, 1, n))
    state_lines = []
    cmd_lines = []
    for i in range(n):
        (ls,) = ax.plot(
            state_t,
            state_q[:, i] if state_q.shape[0] else [],
            color=colors[i],
            linewidth=1.2,
            label=f"{names[i]} state",
            visible=active[i],
        )
        (lc,) = ax.plot(
            cmd_t,
            cmd_q[:, i] if cmd_q.shape[0] else [],
            color=colors[i],
            linewidth=1.2,
            linestyle="--",
            label=f"{names[i]} cmd",
            visible=active[i],
        )
        state_lines.append(ls)
        cmd_lines.append(lc)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("q (rad)")
    ax.set_title("solid = state_q   dashed = cmd_q")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    checks = CheckButtons(rax, names, active)

    # Shrink checkbox label font so 22 joints fit.
    for label in checks.labels:
        label.set_fontsize(8)

    def _on_check(label: str) -> None:
        idx = names.index(label)
        vis = not state_lines[idx].get_visible()
        state_lines[idx].set_visible(vis)
        cmd_lines[idx].set_visible(vis)
        # Rebuild legend from visible lines only.
        handles = []
        labels_out = []
        for i in range(n):
            if state_lines[i].get_visible():
                handles.extend([state_lines[i], cmd_lines[i]])
                labels_out.extend([f"{names[i]} state", f"{names[i]} cmd"])
        ax.legend(handles, labels_out, loc="upper right", fontsize=7, ncol=2)
        fig.canvas.draw_idle()

    checks.on_clicked(_on_check)
    plt.show()
    return 0


# ---------------------------------------------------------------------------
# ROS2 logger
# ---------------------------------------------------------------------------


class JointLogger:
    """Created only when logging; ROS imports stay inside run_logger()."""

    def __init__(
        self,
        node: Any,
        output: Path,
        joint_state_topic: str,
        joint_ctrl_topic: str,
        JointState: Any,
        LowCmd: Any,
    ) -> None:
        self._node = node
        self._output = Path(output)
        self._name_to_index = joint_name_to_index()
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()

        self._state_t: List[float] = []
        self._state_q: List[np.ndarray] = []
        self._state_dq: List[np.ndarray] = []

        self._cmd_t: List[float] = []
        self._cmd_q: List[np.ndarray] = []
        self._cmd_dq: List[np.ndarray] = []
        self._cmd_tau: List[np.ndarray] = []
        self._cmd_kp: List[np.ndarray] = []
        self._cmd_kd: List[np.ndarray] = []
        self._cmd_weight: List[np.ndarray] = []

        self._state_count = 0
        self._cmd_count = 0
        self._rate_t0 = time.perf_counter()

        node.create_subscription(
            JointState, joint_state_topic, self._on_joint_state, 50
        )
        node.create_subscription(LowCmd, joint_ctrl_topic, self._on_joint_ctrl, 50)
        node.create_timer(1.0, self._on_rate_timer)
        node.get_logger().info(
            f"Logging {joint_state_topic} + {joint_ctrl_topic} → {self._output}"
        )

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    def _on_joint_state(self, msg: Any) -> None:
        q = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        dq = np.zeros(B1_JOINT_COUNT, dtype=np.float64)

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
            q[:n] = np.asarray(msg.position[:n], dtype=np.float64)
            n_dq = min(len(msg.velocity), B1_JOINT_COUNT)
            dq[:n_dq] = np.asarray(msg.velocity[:n_dq], dtype=np.float64)

        with self._lock:
            self._state_t.append(self._now())
            self._state_q.append(q)
            self._state_dq.append(dq)
            self._state_count += 1

    def _on_joint_ctrl(self, msg: Any) -> None:
        q = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        dq = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        tau = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        kp = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        kd = np.zeros(B1_JOINT_COUNT, dtype=np.float64)
        weight = np.zeros(B1_JOINT_COUNT, dtype=np.float64)

        n = min(len(msg.motor_cmd), B1_JOINT_COUNT)
        for i in range(n):
            m = msg.motor_cmd[i]
            q[i] = float(m.q)
            dq[i] = float(m.dq)
            tau[i] = float(m.tau)
            kp[i] = float(m.kp)
            kd[i] = float(m.kd)
            weight[i] = float(m.weight)

        with self._lock:
            self._cmd_t.append(self._now())
            self._cmd_q.append(q)
            self._cmd_dq.append(dq)
            self._cmd_tau.append(tau)
            self._cmd_kp.append(kp)
            self._cmd_kd.append(kd)
            self._cmd_weight.append(weight)
            self._cmd_count += 1

    def _on_rate_timer(self) -> None:
        elapsed = time.perf_counter() - self._rate_t0
        if elapsed <= 0.0:
            return
        with self._lock:
            n_state = self._state_count
            n_cmd = self._cmd_count
            self._state_count = 0
            self._cmd_count = 0
            n_state_tot = len(self._state_t)
            n_cmd_tot = len(self._cmd_t)
        self._rate_t0 = time.perf_counter()
        self._node.get_logger().info(
            f"rate: joint_states={n_state / elapsed:.1f} Hz  "
            f"joint_ctrl={n_cmd / elapsed:.1f} Hz  "
            f"(logged {n_state_tot} states, {n_cmd_tot} cmds)"
        )

    def save(self) -> Path:
        with self._lock:
            state_t = np.asarray(self._state_t, dtype=np.float64)
            state_q = (
                np.stack(self._state_q, axis=0)
                if self._state_q
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )
            state_dq = (
                np.stack(self._state_dq, axis=0)
                if self._state_dq
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )
            cmd_t = np.asarray(self._cmd_t, dtype=np.float64)
            cmd_q = (
                np.stack(self._cmd_q, axis=0)
                if self._cmd_q
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )
            cmd_dq = (
                np.stack(self._cmd_dq, axis=0)
                if self._cmd_dq
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )
            cmd_tau = (
                np.stack(self._cmd_tau, axis=0)
                if self._cmd_tau
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )
            cmd_kp = (
                np.stack(self._cmd_kp, axis=0)
                if self._cmd_kp
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )
            cmd_kd = (
                np.stack(self._cmd_kd, axis=0)
                if self._cmd_kd
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )
            cmd_weight = (
                np.stack(self._cmd_weight, axis=0)
                if self._cmd_weight
                else np.zeros((0, B1_JOINT_COUNT), dtype=np.float64)
            )

        out = self._output
        if out.suffix == "":
            out = out.with_suffix(".npz")
        out.parent.mkdir(parents=True, exist_ok=True)

        meta: Dict[str, Any] = {
            "joint_names": list(JOINT_NAMES),
            "joint_count": B1_JOINT_COUNT,
            "num_state_samples": int(state_q.shape[0]),
            "num_cmd_samples": int(cmd_q.shape[0]),
            "duration_s": float(
                max(
                    state_t[-1] if len(state_t) else 0.0,
                    cmd_t[-1] if len(cmd_t) else 0.0,
                )
            ),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "notes": (
                "state_* from /joint_states; cmd_* from /joint_ctrl LowCmd.motor_cmd "
                "in JointIndex order. Timestamps are seconds since logger start."
            ),
        }

        np.savez_compressed(
            out,
            state_t=state_t,
            state_q=state_q,
            state_dq=state_dq,
            cmd_t=cmd_t,
            cmd_q=cmd_q,
            cmd_dq=cmd_dq,
            cmd_tau=cmd_tau,
            cmd_kp=cmd_kp,
            cmd_kd=cmd_kd,
            cmd_weight=cmd_weight,
            joint_names=np.asarray(JOINT_NAMES),
            meta=json.dumps(meta),
        )
        meta_path = out.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._node.get_logger().info(
            f"saved {meta['num_state_samples']} states + "
            f"{meta['num_cmd_samples']} cmds → {out}"
        )
        return out


def run_logger(argv: Optional[Sequence[str]] = None) -> int:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    from booster_interface.msg import LowCmd

    parser = argparse.ArgumentParser(
        description="Log /joint_states and /joint_ctrl to an .npz file."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="runs/joint_log",
        help="Output path prefix or .npz (default: runs/joint_log)",
    )
    parser.add_argument(
        "--joint-states-topic",
        default="/joint_states",
        help="JointState topic (default: /joint_states)",
    )
    parser.add_argument(
        "--joint-ctrl-topic",
        default="/joint_ctrl",
        help="LowCmd topic (default: /joint_ctrl)",
    )
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = Node("joint_log")
    logger = JointLogger(
        node=node,
        output=Path(args.output),
        joint_state_topic=args.joint_states_topic,
        joint_ctrl_topic=args.joint_ctrl_topic,
        JointState=JointState,
        LowCmd=LowCmd,
    )
    try:
        print("Logging… Ctrl+C to stop and save.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if len(logger._state_t) > 0 or len(logger._cmd_t) > 0:
            logger.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def _resolve_plot_path(arg: str) -> Optional[Path]:
    p = Path(arg)
    candidates = [p]
    if p.suffix == "":
        candidates.append(p.with_suffix(".npz"))
    elif p.suffix == ".json":
        candidates.append(p.with_suffix(".npz"))
    for c in candidates:
        if c.is_file() and c.suffix == ".npz":
            return c
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    # Offline plot: python3 joint_log.py path/to/log.npz
    if len(args) == 1 and not args[0].startswith("-"):
        plot_path = _resolve_plot_path(args[0])
        if plot_path is not None:
            return plot_joint_log(plot_path)
        print(f"File not found: {args[0]}", file=sys.stderr)
        return 1

    return run_logger(args)


if __name__ == "__main__":
    raise SystemExit(main())
