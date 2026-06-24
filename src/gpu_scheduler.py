"""
Stage 2: GPU scheduler with semaphore-based concurrency control.
Uses multiprocessing (spawn) to isolate CUDA contexts per worker.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

from src.config import PipelineConfig
from src.text_formatter import Segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel sent through the task queue to signal worker shutdown
_SHUTDOWN = "__SHUTDOWN__"
_WORKER_STARTUP_TIMEOUT = 120  # seconds for model loading


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TaskRequest = tuple[Path, Path]          # (audio_path, video_path)
TaskResult = tuple[Path, list[Segment]]  # (video_path, segments)
TaskError = tuple[Path, str]             # (video_path, error_message)


# ---------------------------------------------------------------------------
# Worker process body
# ---------------------------------------------------------------------------

def _worker_process(
    model_path: str,
    language: str,
    beam_size: int,
    vad_filter: bool,
    compute_type: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    semaphore: mp.Semaphore,
    worker_id: int,
) -> None:
    """
    Child process entry point.

    - Acquires semaphore slot before processing (controls GPU concurrency).
    - Loads WhisperModel once, then loops on task_queue.
    - Sends (audio_path, segments) or (audio_path, error_string) to result_queue.
    """
    # Ensure nvidia CUDA DLLs are on the DLL search path BEFORE any import
    # of faster_whisper / ctranslate2. Must happen here in the child process
    # since spawn mode starts a fresh interpreter.
    import os as _os
    import sys as _sys

    dll_dirs: set[str] = set()
    for p in _sys.path:
        nvidia_root = _os.path.join(p, "nvidia")
        if not _os.path.isdir(nvidia_root):
            continue
        for pkg in _os.listdir(nvidia_root):
            for sub in ("bin", "lib"):
                d = _os.path.join(nvidia_root, pkg, sub)
                if _os.path.isdir(d):
                    _os.add_dll_directory(d)
                    dll_dirs.add(d)

    # Prepend nvidia bin dirs to PATH — ctranslate2's native loader
    # uses LoadLibrary which relies on PATH, not LOAD_LIBRARY_SEARCH.
    _path_parts = _os.environ.get("PATH", "").split(_os.pathsep)
    _os.environ["PATH"] = _os.pathsep.join(sorted(dll_dirs) + _path_parts)

    # Safe to import now that DLL search paths are configured
    from src.transcribe_worker import transcribe_worker

    logger.info(f"Worker-{worker_id} started, loading model...")

    # Pre-load model once (transcribe_worker caches internally)
    try:
        # Warm-up: load model by transcribing a tiny dummy? No —
        # the first real transcription will trigger model load via _load_model cache.
        pass
    except Exception:
        pass

    while True:
        # Wait for GPU slot
        semaphore.acquire()
        logger.debug(f"Worker-{worker_id} acquired semaphore")

        try:
            item = task_queue.get()
            if item == _SHUTDOWN:
                logger.info(f"Worker-{worker_id} received shutdown")
                break

            audio_path, video_path = item
            logger.info(f"Worker-{worker_id} processing: {video_path.name}")

            try:
                segments = transcribe_worker(
                    model_path=str(model_path),
                    audio_path=audio_path,
                    language=language,
                    beam_size=beam_size,
                    vad_filter=vad_filter,
                    compute_type=compute_type,
                )
                result_queue.put((audio_path, segments))
            except Exception as exc:
                logger.error(f"Worker-{worker_id} error on {audio_path.name}: {exc}")
                result_queue.put((audio_path, str(exc)))

        except Exception as exc:
            logger.error(f"Worker-{worker_id} queue error: {exc}")
        finally:
            semaphore.release()

    logger.info(f"Worker-{worker_id} exiting")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class GpuScheduler:
    """
    GPU-memory-aware task scheduler.

    Spawns N worker processes, feeds them audio paths via a task queue,
    and collects results via a result queue.

    Semaphore ensures at most max_workers GPUs are active simultaneously.
    """

    def __init__(self, config: PipelineConfig):
        self._cfg = config
        self._task_queue: mp.Queue | None = None
        self._result_queue: mp.Queue | None = None
        self._semaphore: mp.Semaphore | None = None
        self._workers: list[mp.Process] = []
        self._ctx = mp.get_context("spawn")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        tasks: list[tuple[Path, Path]],  # [(audio_path, video_path), ...]
        *,
        progress_callback: callable = None,  # (received: int, total: int) -> None
    ) -> dict[Path, list[Segment]]:
        """
        Run ASR transcription on all queued audio files.

        Args:
            tasks: List of (audio_path, video_path) tuples to process.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dict mapping audio_path → list of Segments.
            Failed tasks are excluded from the dict (errors are logged).
        """
        if not tasks:
            logger.info("No tasks to process")
            return {}

        self._setup_queues()
        self._start_workers(task_count=len(tasks))
        self._enqueue_tasks(tasks)
        results = self._collect_results(expected=len(tasks), progress_callback=progress_callback)
        self._shutdown()
        return results

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _setup_queues(self) -> None:
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._semaphore = self._ctx.Semaphore(self._cfg.max_workers)

    def _start_workers(self, task_count: int) -> None:
        # Don't spawn more workers than tasks — idle workers are noise
        num_workers = max(1, min(self._cfg.max_workers, task_count))
        for i in range(num_workers):
            p = self._ctx.Process(
                target=_worker_process,
                args=(
                    str(self._cfg.model_path),
                    self._cfg.language,
                    self._cfg.beam_size,
                    self._cfg.vad_filter,
                    self._cfg.compute_type,
                    self._task_queue,
                    self._result_queue,
                    self._semaphore,
                    i,
                ),
                name=f"asr-worker-{i}",
                daemon=True,
            )
            p.start()
            self._workers.append(p)
            logger.info(f"Spawned Worker-{i} (pid={p.pid})")

    def _enqueue_tasks(self, tasks: list[tuple[Path, Path]]) -> None:
        """Push tasks + sentinel values (one per worker) into the queue."""
        for task in tasks:
            self._task_queue.put(task)

        # One sentinel per worker so each exits cleanly
        for _ in self._workers:
            self._task_queue.put(_SHUTDOWN)

    def _collect_results(
        self, expected: int, *, progress_callback: callable = None
    ) -> dict[Path, list[Segment]]:
        """Drain result queue until expected count reached."""
        results: dict[Path, list[Segment]] = {}
        received = 0
        errors = 0

        while received < expected:
            try:
                item = self._result_queue.get(timeout=5)
                audio_path, payload = item

                if isinstance(payload, str):
                    # Error string
                    logger.error(f"FAILED: {audio_path.name} — {payload}")
                    errors += 1
                else:
                    results[audio_path] = payload

                received += 1
                if progress_callback:
                    progress_callback(received, expected)
            except Exception:
                # Timeout — check if workers are still alive
                alive = sum(1 for w in self._workers if w.is_alive())
                if alive == 0:
                    logger.error("All workers died unexpectedly")
                    break
                logger.info(
                    f"Transcribing... ({received}/{expected} done, {alive} worker(s) active)"
                )
                if progress_callback:
                    progress_callback(received, expected)

        logger.info(
            f"Transcription complete: {len(results)} success, {errors} failed "
            f"(out of {expected})"
        )
        return results

    def _shutdown(self) -> None:
        """Join all worker processes."""
        for w in self._workers:
            w.join(timeout=10)
            if w.is_alive():
                logger.warning(f"Force terminating {w.name}")
                w.terminate()
                w.join(timeout=3)
        self._workers.clear()
        logger.info("All workers shut down")
