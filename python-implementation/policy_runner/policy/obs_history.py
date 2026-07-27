"""Ring buffer that stacks single-frame observations into a flat model input."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Literal, Optional, Sequence, Tuple

HistoryLayout = Literal["frame", "term"]


class ObservationHistory:
    """
    Keeps the last `history_len` frames of size `frame_dim`.

    push() validates each frame against frame_dim and returns a flat vector of
    length frame_dim * history_len. Until the buffer is full, missing frames are
    filled by repeating the oldest available frame (or zeros if empty).

    layout:
      "frame" — [frame_{t-H+1} | ... | frame_t]
      "term"  — for each term group (sizes in term_sizes):
                [term_{t-H+1} | ... | term_t]
                e.g. gravity [0,0,-1] × 3 → [0,0,-1, 0,0,-1, 0,0,-1]
    """

    def __init__(
        self,
        frame_dim: int,
        history_len: int,
        layout: HistoryLayout = "frame",
        term_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        if frame_dim <= 0:
            raise ValueError("frame_dim must be positive")
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        if layout not in ("frame", "term"):
            raise ValueError(f"layout must be 'frame' or 'term', got {layout!r}")

        self.frame_dim = frame_dim
        self.history_len = history_len
        self.layout = layout
        self._frames: Deque[List[float]] = deque(maxlen=history_len)

        if layout == "term":
            if not term_sizes:
                raise ValueError("term layout requires term_sizes")
            sizes = tuple(int(s) for s in term_sizes)
            if any(s <= 0 for s in sizes):
                raise ValueError(f"term_sizes must be positive, got {sizes}")
            if sum(sizes) != frame_dim:
                raise ValueError(
                    f"sum(term_sizes)={sum(sizes)} != frame_dim={frame_dim}"
                )
            self.term_sizes: Tuple[int, ...] = sizes
        else:
            self.term_sizes = ()

    @property
    def input_dim(self) -> int:
        return self.frame_dim * self.history_len

    def reset(self, fill: Optional[Sequence[float]] = None) -> None:
        self._frames.clear()
        if fill is not None:
            self.push(fill)

    def push(self, frame: Sequence[float]) -> List[float]:
        if len(frame) != self.frame_dim:
            raise ValueError(
                f"ObservationHistory: frame dim {len(frame)} != "
                f"frame_dim {self.frame_dim}"
            )
        self._frames.append(list(frame))
        return self.as_input()

    def as_input(self) -> List[float]:
        """Flattened history, oldest → newest within the chosen layout."""
        if not self._frames:
            return [0.0] * self.input_dim

        frames = list(self._frames)
        while len(frames) < self.history_len:
            frames.insert(0, list(frames[0]))

        if self.layout == "frame":
            flat: List[float] = []
            for f in frames:
                flat.extend(f)
        else:
            # Term-group major: for each term, concat that slice over history.
            flat = []
            offset = 0
            for size in self.term_sizes:
                for h in range(self.history_len):
                    flat.extend(frames[h][offset : offset + size])
                offset += size

        if len(flat) != self.input_dim:
            raise ValueError(
                f"ObservationHistory: stacked dim {len(flat)} != "
                f"input_dim {self.input_dim}"
            )
        return flat
