"""Optional video plus trajectory recording for UR3 evaluation episodes."""

import csv
from enum import Enum
from pathlib import Path

import cv2
import numpy as np


class VideoCodec(str, Enum):
    """Supported video codec fourcc codes."""

    MJPG = "MJPG"


class EpisodeRecorder:
    """Records per-step frames and trajectory rows for one episode."""

    BUFFER_SIZE = 10
    VIDEO_FPS = 10

    def __init__(
        self,
        environment_id: str,
        language_instruction: str,
        trajectory_columns: list[str],
        frame_skip: int = 5,
    ):
        self.safe_instruction = language_instruction.replace(" ", "_").replace("/", "-")
        self.trajectory_columns = trajectory_columns
        self.frame_skip = frame_skip
        self.step_counter = 0
        self.frames_buffer: list[np.ndarray] = []
        self.trajectory_rows: list[dict[str, float]] = []
        self.writer: cv2.VideoWriter | None = None
        self.filepath: Path | None = None
        self.environment_id = environment_id
        self.num_saves = 0

    def _init_writer(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self.environment_id}_unknown_"
            f"{self.safe_instruction}_{self.num_saves}.avi"
        )
        self.filepath = output_directory / filename
        height, width = self.frames_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*VideoCodec.MJPG.value)
        self.writer = cv2.VideoWriter(
            str(self.filepath),
            fourcc,
            self.VIDEO_FPS,
            (width, height),
        )

    def _flush_buffer(self, output_directory: Path) -> None:
        if not self.frames_buffer:
            return
        if self.writer is None:
            self._init_writer(output_directory)
        for frame in self.frames_buffer:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(bgr)
        self.frames_buffer = []

    def add_observation(
        self,
        frame: np.ndarray | None,
        trajectory_row: dict[str, float],
        reward: float,
        output_directory: Path,
    ) -> None:
        trajectory_row["reward"] = reward
        self.trajectory_rows.append(trajectory_row)
        if frame is not None and self.step_counter % self.frame_skip == 0:
            self.frames_buffer.append(frame)
            if len(self.frames_buffer) >= self.BUFFER_SIZE:
                self._flush_buffer(output_directory)
        self.step_counter += 1

    def save(
        self,
        reward: float,
        p1: float,
        p2: float,
        behavior_order: str,
        output_directory: Path,
    ) -> None:
        """Flush files and include the episode metrics in filenames."""
        output_directory.mkdir(parents=True, exist_ok=True)
        safe_order = behavior_order.replace("->", "-")
        file_prefix = (
            f"{self.environment_id}_reward={reward:.4f}_"
            f"p1={p1:.1f}_p2={p2:.1f}_order={safe_order}_"
            f"{self.safe_instruction}_{self.num_saves}"
        )
        if self.frames_buffer or self.writer is not None:
            self._flush_buffer(output_directory)
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            if self.filepath is not None:
                self.filepath.rename(output_directory / f"{file_prefix}.avi")
                self.filepath = None

        csv_columns = self.trajectory_columns + ["reward"]
        csv_path = output_directory / f"{file_prefix}_trajectory.csv"
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerows(self.trajectory_rows)
        self.num_saves += 1

