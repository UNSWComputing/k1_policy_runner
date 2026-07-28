"""Record walk_v1 model inputs (195-D) with layout metadata for later analysis."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Must match walk_policy_v1.FRAME_TERM_SIZES / HISTORY_LEN / term layout.
TERM_NAMES: Tuple[str, ...] = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
)
TERM_SIZES: Tuple[int, ...] = (3, 3, 22, 22, 12, 3)
FRAME_DIM = sum(TERM_SIZES)  # 65
HISTORY_LEN = 3
MODEL_INPUT_DIM = FRAME_DIM * HISTORY_LEN  # 195
LAYOUT = "term"  # per term: [t-2 | t-1 | t]


def walk_v1_layout_meta() -> Dict[str, Any]:
    """Schema describing how to slice a 195-D model input."""
    return {
        "layout": LAYOUT,
        "frame_dim": FRAME_DIM,
        "history_len": HISTORY_LEN,
        "model_input_dim": MODEL_INPUT_DIM,
        "term_names": list(TERM_NAMES),
        "term_sizes": list(TERM_SIZES),
        "notes": (
            "Term-major: for each term, concat history oldest→newest "
            "[t-2|t-1|t]. joint_pos is relative to DEFAULT_JOINT_POS. "
            "actions is last raw policy emit (no scale/offset/clamp)."
        ),
    }


def term_byte_range(term: str) -> Tuple[int, int]:
    """Inclusive-start / exclusive-end index range of `term` in a 195-D vector."""
    if term not in TERM_NAMES:
        raise KeyError(f"unknown term {term!r}; expected one of {TERM_NAMES}")
    offset = 0
    for name, size in zip(TERM_NAMES, TERM_SIZES):
        span = size * HISTORY_LEN
        if name == term:
            return offset, offset + span
        offset += span
    raise RuntimeError("unreachable")


def slice_term(
    model_input: Sequence[float],
    term: str,
    history_index: int = -1,
) -> np.ndarray:
    """
    Extract one term from a 195-D (or batch) model input.

    history_index: 0 = oldest (t-2), 1 = mid (t-1), -1/2 = newest (t).
    """
    x = np.asarray(model_input, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[-1] != MODEL_INPUT_DIM:
        raise ValueError(
            f"expected last dim {MODEL_INPUT_DIM}, got {x.shape[-1]}"
        )

    lo, _ = term_byte_range(term)
    size = TERM_SIZES[TERM_NAMES.index(term)]
    h = history_index if history_index >= 0 else HISTORY_LEN + history_index
    if not (0 <= h < HISTORY_LEN):
        raise IndexError(f"history_index {history_index} out of range")

    start = lo + h * size
    out = x[:, start : start + size]
    return out[0] if np.asarray(model_input).ndim == 1 else out


class ModelInputRecorder:
    """Append 195-D model inputs in memory; flush to .npz (+ .json meta)."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path) if path is not None else None
        self._rows: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._t0 = time.perf_counter()

    def __len__(self) -> int:
        return len(self._rows)

    def record(self, model_input: Sequence[float]) -> None:
        row = np.asarray(model_input, dtype=np.float32).reshape(-1)
        if row.shape != (MODEL_INPUT_DIM,):
            raise ValueError(
                f"ModelInputRecorder: expected {MODEL_INPUT_DIM} dims, "
                f"got {row.size}"
            )
        self._rows.append(row.copy())
        self._timestamps.append(time.perf_counter() - self._t0)

    def as_array(self) -> np.ndarray:
        if not self._rows:
            return np.zeros((0, MODEL_INPUT_DIM), dtype=np.float32)
        return np.stack(self._rows, axis=0)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        out = Path(path) if path is not None else self.path
        if out is None:
            raise ValueError("ModelInputRecorder.save: no path given")
        out = out.with_suffix(".npz") if out.suffix == "" else out
        out.parent.mkdir(parents=True, exist_ok=True)

        obs = self.as_array()
        ts = np.asarray(self._timestamps, dtype=np.float64)
        meta = walk_v1_layout_meta()
        meta["num_steps"] = int(obs.shape[0])
        meta["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        np.savez_compressed(out, obs=obs, timestamps=ts, meta=json.dumps(meta))
        meta_path = out.with_suffix(".json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(
            f"[obs_record] saved {obs.shape[0]} steps × {obs.shape[1]} dims "
            f"→ {out}"
        )
        return out

    def reset(self) -> None:
        self._rows.clear()
        self._timestamps.clear()
        self._t0 = time.perf_counter()


def load_recording(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load (obs [T,195], timestamps [T], meta dict) from a .npz recording."""
    p = Path(path)
    data = np.load(p, allow_pickle=False)
    obs = np.asarray(data["obs"])
    timestamps = np.asarray(data["timestamps"])
    meta_raw = data["meta"]
    if isinstance(meta_raw, np.ndarray):
        meta_raw = meta_raw.item()
    meta = json.loads(str(meta_raw))
    return obs, timestamps, meta
