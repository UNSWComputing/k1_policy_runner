#!/usr/bin/env python3
"""Load / inspect / plot walk_v1 195-D model-input recordings.

Recording (during a run):
  python3 policy_runner_main.py walk_v1 --record-obs ../runs/walk_obs

Inspect:
  python3 joint_record.py path/to/walk_obs.npz
  python3 joint_record.py path/to/walk_obs.npz --term joint_pos

Plot joint_pos + last policy actions:
  python3 joint_record.py path/to/walk_obs.npz --plot
  python3 joint_record.py path/to/walk_obs.npz --plot --save plots/walk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Import layout helpers from the python-implementation package.
_PY = Path(__file__).resolve().parent / "python-implementation"
if str(_PY) not in sys.path:
    sys.path.insert(0, str(_PY))

from policy_runner.joint_index import JOINT_NAMES  # noqa: E402
from policy_runner.obs_record import (  # noqa: E402
    TERM_NAMES,
    load_recording,
    slice_term,
)

# 12-D policy action order (matches walk_policy_v1.LEG_JOINTS).
ACTION_NAMES = [
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
]


def _plot_channels(
    t: np.ndarray,
    series: np.ndarray,
    names: Sequence[str],
    title: str,
    ylabel: str,
    save_path: Optional[Path] = None,
):
    """Build one multi-channel figure; caller is responsible for show/close."""
    import matplotlib.pyplot as plt

    n = series.shape[1]
    if len(names) != n:
        raise ValueError(f"names ({len(names)}) != series cols ({n})")

    ncols = 2 if n > 6 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5 * ncols, 1.6 * nrows),
        sharex=True,
        squeeze=False,
        num=title,
    )
    fig.suptitle(title, fontsize=12)

    for i in range(n):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        ax.plot(t, series[:, i], linewidth=1.0)
        ax.set_ylabel(names[i], fontsize=8)
        ax.grid(True, alpha=0.3)
        if r == nrows - 1:
            ax.set_xlabel("time (s)")

    for i in range(n, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.text(0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=10)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.96))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"saved {save_path}")
    return fig


def plot_recording(
    obs: np.ndarray,
    timestamps: np.ndarray,
    history_index: int = -1,
    save_prefix: Optional[Path] = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    t = timestamps if len(timestamps) == obs.shape[0] else np.arange(obs.shape[0])

    joint_pos = slice_term(obs, "joint_pos", history_index=history_index)
    actions = slice_term(obs, "actions", history_index=history_index)

    joint_save = (
        Path(f"{save_prefix}_joint_pos.png") if save_prefix is not None else None
    )
    action_save = (
        Path(f"{save_prefix}_actions.png") if save_prefix is not None else None
    )

    # Build both figures before show() so both windows appear together.
    _plot_channels(
        t,
        joint_pos,
        JOINT_NAMES,
        title=f"joint_pos (relative to default, hist={history_index})",
        ylabel="rad (rel)",
        save_path=joint_save,
    )
    _plot_channels(
        t,
        actions,
        ACTION_NAMES,
        title=f"last policy actions (raw emit, hist={history_index})",
        ylabel="action",
        save_path=action_save,
    )

    if show:
        print("Showing 2 figures: joint_pos + last policy actions (close windows to exit)")
        plt.show()
    else:
        plt.close("all")



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect / plot a walk_v1 model-input recording (.npz)."
    )
    parser.add_argument("recording", help="Path to .npz from --record-obs")
    parser.add_argument(
        "--term",
        choices=TERM_NAMES,
        default=None,
        help="Print stats for one term (newest history step)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=-1,
        help="History index within term: 0=oldest … -1=newest (default)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot joint_pos (22) and last actions (12) vs time",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PREFIX",
        help="Save plots as PREFIX_joint_pos.png and PREFIX_actions.png",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive windows (use with --save)",
    )
    args = parser.parse_args(argv)

    obs, timestamps, meta = load_recording(args.recording)
    print(f"file: {args.recording}")
    print(f"steps: {obs.shape[0]}  dim: {obs.shape[1]}")
    print(f"duration_s: {timestamps[-1] if len(timestamps) else 0:.3f}")
    print(f"layout: {meta.get('layout')}  terms: {meta.get('term_names')}")

    if args.plot:
        prefix = Path(args.save) if args.save else None
        plot_recording(
            obs,
            timestamps,
            history_index=args.history,
            save_prefix=prefix,
            show=not args.no_show,
        )
        return 0

    if args.term is None:
        print("\nTerms (use --term NAME to inspect, or --plot):")
        for name, size in zip(meta["term_names"], meta["term_sizes"]):
            print(
                f"  {name:20s}  per-frame={size}  "
                f"in-195={size * meta['history_len']}"
            )
        return 0

    series = slice_term(obs, args.term, history_index=args.history)
    print(f"\n{args.term} (history_index={args.history}) shape={series.shape}")
    print(f"  mean: {series.mean(axis=0)}")
    print(f"  min:  {series.min(axis=0)}")
    print(f"  max:  {series.max(axis=0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
