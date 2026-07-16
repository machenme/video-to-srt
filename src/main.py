#!/usr/bin/env python
"""
ASR Pipeline CLI — end-to-end video-to-text for Japanese videos.

Usage:
    uv run python -m src.main --input ./videos --output ./subtitles
    uv run src/main.py --input . --output ./output --verbose
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import shutil
import signal
import threading
import time

from src.config import PipelineConfig
from src.translator import EdgeTranslator, translate_srt, TranslationError
from src.utils import scan_video_files
from src.audio_extractor import AudioExtractor
from src.gpu_scheduler import GpuScheduler
from src.text_formatter import Segment, TextFormatter
from src.task_manager import TaskManager
from src.monitor import GpuMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reusable pipeline function (used by both CLI and GUI)
# ---------------------------------------------------------------------------

def run_one_video(
    config: PipelineConfig,
    video_path: Path,
    *,
    progress_callback: callable = None,
    cancel_event: threading.Event = None,
) -> tuple[bool, int, str]:
    """
    Process a single video through the full pipeline.

    Args:
        config: Validated PipelineConfig.
        video_path: Path to the source video file.
        progress_callback: Optional callable(stage, current, total) for progress.
        cancel_event: Optional threading.Event; set to request graceful stop.

    Returns:
        (success, segment_count, error_message)
    """
    _check_cancelled = lambda: cancel_event and cancel_event.is_set()
    temp_dir = config.effective_temp_dir
    extractor = AudioExtractor(temp_dir)

    # --- Stage 1: Audio extraction + chunking ---
    if _check_cancelled():
        return (False, 0, "Cancelled before extraction")

    logger.info(f"Extracting audio: {video_path.name}")
    if progress_callback:
        progress_callback("extracting", 0, 100)

    try:
        wav_path = extractor.extract(video_path)
    except Exception as exc:
        return (False, 0, str(exc))

    chunk_sec = config.chunk_duration
    if chunk_sec == 0:
        # Auto: split evenly by worker count so all workers finish simultaneously
        dur = extractor.get_duration(wav_path)
        chunk_sec = max(30, int(dur / config.max_workers))
        logger.info(f"Auto chunk: duration={dur:.0f}s, workers={config.max_workers} → {chunk_sec}s/chunk")
    if chunk_sec > 0:
        try:
            chunks = extractor.split_wav(wav_path, chunk_sec)
        except Exception as exc:
            return (False, 0, str(exc))
    else:
        chunks = [(0.0, wav_path)]

    logger.info(f"Audio ready: {len(chunks)} chunk(s)")

    if _check_cancelled():
        return (False, 0, "Cancelled after extraction")

    # --- Stage 2: GPU ASR ---
    gpu_monitor = GpuMonitor(gpu_index=0, interval=1.0)
    gpu_monitor.start()

    scheduler_tasks = [(cp, video_path) for _, cp in chunks]
    scheduler = GpuScheduler(config)
    start_time = time.time()

    if progress_callback:
        progress_callback("transcribing", 0, len(scheduler_tasks))

    raw_results = scheduler.process(
        scheduler_tasks,
        progress_callback=lambda received, total: (
            progress_callback("transcribing", received, total)
            if progress_callback else None
        ),
    )
    elapsed = time.time() - start_time
    gpu_monitor.stop()

    if _check_cancelled():
        return (False, 0, "Cancelled after transcription")

    # --- Stage 3: Merge & write ---
    formatter = TextFormatter()
    output_dir = config.output_dir

    # Collect chunk results
    chunk_results: list[tuple[float, list[Segment]]] = []
    all_done = True
    for offset, chunk_path in chunks:
        if chunk_path in raw_results:
            chunk_results.append((offset, raw_results[chunk_path]))
        else:
            all_done = False

    if not all_done or not chunk_results:
        return (False, 0, "Some chunks failed transcription")

    if len(chunk_results) > 1:
        segments = formatter.combine_chunk_segments(chunk_results)
        logger.info(f"Combined {len(chunk_results)} chunks → {len(segments)} segments")
    else:
        segments = chunk_results[0][1]

    video_out_dir = output_dir
    video_out_dir.mkdir(parents=True, exist_ok=True)
    written = formatter.write_all(
        segments,
        base_path=video_out_dir / video_path.stem,
        formats=config.output_formats,
    )

    # --- Stage 3b: Translation (optional) ---
    translated_path: Path | None = None
    if config.translate_to:
        srt_path = video_out_dir / f"{video_path.stem}.srt"
        if srt_path.exists():
            logger.info(
                f"Translating SRT: {srt_path.name} → {config.translate_to}"
            )
            try:
                provider = EdgeTranslator()
                translated_path = translate_srt(
                    srt_path,
                    config.translate_to,
                    provider=provider,
                    source_lang=config.language if config.language != "auto" else "auto",
                )
                written.append(translated_path)
            except TranslationError as exc:
                logger.error(f"Translation failed: {exc}")

    logger.info(
        f"✓ {video_path.stem}: {len(segments)} segments → "
        f"{', '.join(p.suffix for p in written)}  ({elapsed:.1f}s)"
    )

    if progress_callback:
        progress_callback("done", len(segments), len(segments))

    return (True, len(segments), "")


# ---------------------------------------------------------------------------
# Batch runner (CLI)
# ---------------------------------------------------------------------------

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
    p.add_argument("--translate", default=None, metavar="LANG",
                   help="Auto-translate SRT to target language (ISO 639-1, e.g. zh)")
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
        "translate_to": cli.translate,
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

    total_segments = 0
    failed_count = 0
    start_time = time.time()

    for task in tasks:
        if _shutdown_requested:
            break
        task_mgr.mark_started(task.video_path)
        ok, seg_count, err = run_one_video(config, task.video_path)
        if ok:
            task_mgr.mark_done(task.video_path)
            total_segments += seg_count
        else:
            task_mgr.mark_failed(task.video_path, err)
            failed_count += 1

    elapsed = time.time() - start_time

    # --- Cleanup ---
    if config.cleanup_temp and config.effective_temp_dir.exists():
        try:
            shutil.rmtree(config.effective_temp_dir)
        except Exception as exc:
            logger.warning(f"Failed to clean temp dir: {exc}")

    # --- Summary ---
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | "
        f"Tasks: {task_mgr.done_count}/{task_mgr.total_count} done"
        + (f", {failed_count} failed" if failed_count else "")
        + f" | Total segments: {total_segments}"
    )

    # --- Save progress ---
    task_mgr.save_progress()

    # --- Exit code ---
    if failed_count > 0 and task_mgr.done_count == 0:
        sys.exit(3)
    elif failed_count > 0:
        sys.exit(2)
    else:
        sys.exit(0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows multiprocessing guard
    import multiprocessing
    multiprocessing.freeze_support()
    main()
