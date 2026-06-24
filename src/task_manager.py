"""
Task manager: job state tracking, checkpoint/resume logic,
and progress persistence to .progress.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

from src.utils import output_exists_and_valid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single video-to-text processing task."""
    video_path: Path
    audio_path: Path | None = None
    status: str = "pending"  # pending | extracting | transcribing | done | failed
    error_message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


@dataclass
class ProgressSnapshot:
    """Serialisable progress state for resume."""
    total: int
    completed: list[str]   # list of video names (stems)
    failed: list[str]
    updated_at: str


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------

class TaskManager:
    """
    Manages task lifecycle, checkpointing, and resume logic.

    On startup, scans output_dir to build the DoneSet and filters
    already-processed videos from the work queue.
    """

    def __init__(self, output_dir: Path, progress_file: Path | None = None):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_file = progress_file or (self._output_dir / ".progress.json")
        self._tasks: dict[str, Task] = {}       # keyed by video stem
        self._done_set: set[str] = set()

    # ------------------------------------------------------------------
    # Build work queue
    # ------------------------------------------------------------------

    def build_queue(self, video_paths: list[Path], force: bool = False) -> list[Task]:
        """
        Build the pending task list, filtering already-completed videos.

        Args:
            video_paths: All discovered video files.
            force: If True, re-process even if output exists.

        Returns:
            List of Task objects with status='pending' (needs processing).
        """
        # Scan existing outputs
        self._done_set = self._scan_completed()

        tasks: list[Task] = []
        for vp in video_paths:
            stem = vp.stem
            task = Task(video_path=vp)
            self._tasks[stem] = task

            if not force and stem in self._done_set:
                task.status = "done"
                logger.info(f"Skip (already done): {stem}")
            else:
                tasks.append(task)

        return tasks

    def _scan_completed(self) -> set[str]:
        """Scan output_dir for valid output files → DoneSet."""
        done: set[str] = set()
        if not self._output_dir.exists():
            return done

        # Check both flat and per-video subfolder layouts
        patterns = [
            self._output_dir.glob("*.srt"),          # flat: output/demo.srt
            self._output_dir.glob("*/*.srt"),        # subfolder: output/demo/demo.srt
        ]
        for pattern in patterns:
            for srt_file in pattern:
                stem = srt_file.stem
                if output_exists_and_valid(stem, srt_file.parent):
                    txt_file = srt_file.parent / f"{stem}.txt"
                    if txt_file.exists() and txt_file.stat().st_size > 0:
                        done.add(stem)

        # Load progress file for additional context
        saved = self._load_progress()
        if saved:
            done |= set(saved.completed)
            # Remove known failures so they get retried
            done -= set(saved.failed)

        return done

    # ------------------------------------------------------------------
    # Progress persistence
    # ------------------------------------------------------------------

    def save_progress(self) -> None:
        """Write current progress to .progress.json."""
        completed = [
            stem for stem, t in self._tasks.items() if t.status == "done"
        ]
        failed = [
            stem for stem, t in self._tasks.items() if t.status == "failed"
        ]
        snap = ProgressSnapshot(
            total=len(self._tasks),
            completed=completed,
            failed=failed,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self._progress_file.write_text(
                json.dumps(snap.__dict__, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(f"Failed to save progress: {exc}")

    def _load_progress(self) -> ProgressSnapshot | None:
        """Load previous progress file if it exists."""
        if not self._progress_file.exists():
            return None
        try:
            data = json.loads(self._progress_file.read_text(encoding="utf-8"))
            return ProgressSnapshot(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(f"Progress file corrupted, ignoring: {exc}")
            return None

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def mark_started(self, video_path: Path, status: str = "transcribing") -> None:
        stem = video_path.stem
        if stem in self._tasks:
            self._tasks[stem].status = status
            self._tasks[stem].started_at = datetime.now(timezone.utc).isoformat()

    def mark_done(self, video_path: Path) -> None:
        stem = video_path.stem
        if stem in self._tasks:
            self._tasks[stem].status = "done"
            self._tasks[stem].finished_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, video_path: Path, error: str) -> None:
        stem = video_path.stem
        if stem in self._tasks:
            self._tasks[stem].status = "failed"
            self._tasks[stem].error_message = error
            self._tasks[stem].finished_at = datetime.now(timezone.utc).isoformat()
        logger.error(f"Task failed [{stem}]: {error}")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def done_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == "done")

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == "failed")

    @property
    def total_count(self) -> int:
        return len(self._tasks)

    def summary(self) -> str:
        return (
            f"Tasks: {self.done_count}/{self.total_count} done"
            + (f", {self.failed_count} failed" if self.failed_count else "")
        )
