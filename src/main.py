#!/usr/bin/env python
"""
ASR Pipeline CLI — end-to-end video-to-text for Japanese videos.

Usage:
    uv run python src/main.py --input ./videos --output ./subtitles
    uv run python src/main.py --input . --output ./output --verbose
"""
from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
import time
from pathlib import Path

from src.config import PipelineConfig
from src.utils import scan_video_files
from src.audio_extractor import AudioExtractor
from src.gpu_scheduler import GpuScheduler
from src.text_formatter import TextFormatter
from src.task_manager import TaskManager
from src.monitor import GpuMonitor

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested = False


def _on_sigint(signum, frame):
    global _shutdown_requested
    print("\n⚠ Ctrl+C received. Finishing current tasks and exiting...")
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asr-pipeline",
        description="Offline batch Japanese video speech-to-text with GPU parallel inference.",
    )
    # Required
    p.add_argument("--input", required=True, help="Input video directory")
    p.add_argument("--output", required=True, help="Output text/subtitle directory")

    # Optional overrides
    p.add_argument("--config", default="./config.yaml", help="Path to config.yaml")
    p.add_argument("--model", default=None, help="Model size: large-v3-turbo | large-v3 | medium")
    p.add_argument("--workers", type=int, default=None, help="Max parallel GPU workers")
    p.add_argument("--temp-dir", default=None, help="Temp audio directory")
    p.add_argument("--language", default=None, help="Target language (ISO 639-1)")
    p.add_argument("--beam-size", type=int, default=None, help="Beam search width (1-10)")
    p.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    p.add_argument("--compute-type", default=None, help="float16 | int8_float16 | int8")
    p.add_argument("--no-cleanup", action="store_true", help="Keep temp audio files")
    p.add_argument("--chunk-duration", type=int, default=None,
                   help="Split audio > N seconds into chunks for parallel (default 900, 0=disabled)")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    p.add_argument("--force", action="store_true", help="Re-process all files (ignore checkpoint)")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Parse CLI ---
    parser = build_argparser()
    cli = parser.parse_args()

    # --- Setup logging ---
    log_level = logging.DEBUG if cli.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Build config ---
    cli_dict = {
        "config": cli.config,
        "model": cli.model,
        "workers": cli.workers,
        "temp_dir": cli.temp_dir,
        "language": cli.language,
        "beam_size": cli.beam_size,
        "vad_filter": not cli.no_vad,
        "compute_type": cli.compute_type,
        "chunk_duration": cli.chunk_duration,
        "cleanup_temp": not cli.no_cleanup,
        "verbose": cli.verbose,
    }
    # Clean None values so they don't override YAML defaults
    cli_dict = {k: v for k, v in cli_dict.items() if v is not None}

    try:
        config = PipelineConfig.build(cli_dict)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"Config loaded: model={config.model_size}, workers={config.max_workers}, "
                f"language={config.language}, beam={config.beam_size}, "
                f"compute={config.compute_type}, vad={config.vad_filter}, "
                f"chunk_duration={config.chunk_duration}s")

    # --- Signal handling ---
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # --- Scan input ---
    logger.info(f"Scanning input directory: {config.input_dir}")
    video_paths = scan_video_files(config.input_dir, config.video_extensions)
    if not video_paths:
        logger.warning(f"No video files found in {config.input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(video_paths)} video(s)")

    # --- Task manager (resume checkpoint) ---
    task_mgr = TaskManager(config.output_dir)
    tasks = task_mgr.build_queue(video_paths, force=cli.force)
    if not tasks:
        logger.info("All videos already processed. Nothing to do.")
        sys.exit(0)

    logger.info(f"Pending tasks: {len(tasks)} / {len(video_paths)} total")

    # --- Stage 1: Audio extraction + chunking ---
    temp_dir = config.effective_temp_dir
    logger.info(f"Temp audio directory: {temp_dir}")

    extractor = AudioExtractor(temp_dir)

    # chunk_jobs: list of (offset, chunk_audio_path, video_path)
    # For non-chunked videos, offset=0.0 and chunk_audio_path is the full WAV.
    chunk_jobs: list[tuple[float, Path, Path]] = []
    # Track which videos need chunk merging: video_path -> list[(offset, Path)]
    video_chunks: dict[Path, list[tuple[float, Path]]] = {}

    for task in tasks:
        if _shutdown_requested:
            break
        task_mgr.mark_started(task.video_path, status="extracting")
        try:
            wav_path = extractor.extract(task.video_path)

            # Determine if chunking is beneficial
            chunk_sec = config.chunk_duration
            if chunk_sec > 0:
                chunks = extractor.split_wav(wav_path, chunk_sec)
            else:
                chunks = [(0.0, wav_path)]

            video_chunks[task.video_path] = chunks
            for offset, chunk_path in chunks:
                chunk_jobs.append((offset, chunk_path, task.video_path))
        except Exception as exc:
            task_mgr.mark_failed(task.video_path, str(exc))

    if _shutdown_requested:
        _graceful_exit(task_mgr, config, 2)

    if not chunk_jobs:
        logger.error("No audio files extracted. Exiting.")
        sys.exit(1)

    logger.info(f"Audio extraction complete: {len(chunk_jobs)} chunk(s) ready "
                f"(from {len(video_chunks)} video(s))")

    # --- Stage 2: GPU ASR ---
    gpu_monitor = GpuMonitor(gpu_index=0, interval=1.0)
    gpu_monitor.start()

    # Scheduler works with (audio_path, video_path) — we flatten chunks
    scheduler_tasks = [(cp, vp) for _, cp, vp in chunk_jobs]
    scheduler = GpuScheduler(config)
    start_time = time.time()
    raw_results = scheduler.process(scheduler_tasks)
    elapsed = time.time() - start_time

    gpu_monitor.stop()

    # --- Stage 3: Merge & write ---
    formatter = TextFormatter()
    output_dir = config.output_dir

    total_segments = 0

    for video_path, chunks_info in video_chunks.items():
        if _shutdown_requested:
            break

        # Collect chunk results for this video
        chunk_results: list[tuple[float, list[Segment]]] = []
        all_chunks_done = True
        for offset, chunk_path in chunks_info:
            if chunk_path in raw_results:
                chunk_results.append((offset, raw_results[chunk_path]))
            else:
                all_chunks_done = False

        if not all_chunks_done or not chunk_results:
            task_mgr.mark_failed(video_path, "Some chunks failed transcription")
            continue

        # Combine chunk segments with offset correction
        if len(chunk_results) > 1:
            segments = formatter.combine_chunk_segments(chunk_results)
            logger.info(f"Combined {len(chunk_results)} chunks → {len(segments)} segments")
        else:
            segments = chunk_results[0][1]

        # Write to per-video subfolder
        video_out_dir = output_dir / video_path.stem
        video_out_dir.mkdir(parents=True, exist_ok=True)
        written = formatter.write_all(
            segments,
            base_path=video_out_dir / video_path.stem,
            formats=config.output_formats,
        )
        total_segments += len(segments)
        task_mgr.mark_done(video_path)
        logger.info(
            f"✓ {video_path.stem}: {len(segments)} segments → "
            f"{', '.join(p.suffix for p in written)}"
        )

    # Mark failed: any video still pending and not in results
    for video_path in video_chunks:
        t = task_mgr._tasks.get(video_path.stem)
        if t and t.status not in ("done", "failed"):
            task_mgr.mark_failed(video_path, "No output from ASR worker")

    # --- Save progress (into each video subfolder) ---
    task_mgr.save_progress()

    # --- Summary ---
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | "
        f"{task_mgr.summary()} | "
        f"Total segments: {total_segments}"
    )

    # --- Cleanup ---
    if config.cleanup_temp and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except Exception as exc:
            logger.warning(f"Failed to clean temp dir: {exc}")

    # --- Exit code ---
    if task_mgr.failed_count > 0 and task_mgr.done_count == 0:
        sys.exit(3)  # fatal
    elif task_mgr.failed_count > 0:
        sys.exit(2)  # partial failure
    else:
        sys.exit(0)  # success


def _graceful_exit(task_mgr: TaskManager, config: PipelineConfig, code: int) -> None:
    """Save progress and exit cleanly on interrupt."""
    logger.info("Saving progress before exit...")
    task_mgr.save_progress()
    # Don't clean temp on interrupt — user may want to resume
    sys.exit(code)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows multiprocessing guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
