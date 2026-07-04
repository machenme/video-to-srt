"""
Stage 2 sub-module: Single GPU worker process.
Loads WhisperModel once at startup, loops consuming audio paths,
produces segment lists.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.text_formatter import Segment

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker entry point (runs in child process)
# ---------------------------------------------------------------------------

def transcribe_worker(
    model_path: str,
    audio_path: Path,
    language: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
    compute_type: str = "float16",
) -> list[Segment]:
    """
    Transcribe a single audio file using faster-whisper.

    This function is called inside the child process. The model is loaded
    ONCE per process and reused across calls via the worker loop in gpu_scheduler.

    Args:
        model_path: Path to CTranslate2 model directory.
        audio_path: Path to 16kHz mono WAV file.
        language: ISO 639-1 language code.
        beam_size: Beam search width.
        vad_filter: Enable Silero VAD.
        compute_type: "float16", "int8_float16", etc.

    Returns:
        List of Segment objects with start/end timestamps and text.
    """
    # Deferred import: only the child process imports faster_whisper
    from faster_whisper import WhisperModel

    # --- load model with fallback ---
    model = _load_model(model_path, compute_type)

    # "auto" → None so faster-whisper auto-detects the language
    lang = None if language == "auto" else language

    logger.info(f"Transcribing: {audio_path.name}")

    segments_raw, info = model.transcribe(
        str(audio_path),
        language=lang,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )

    logger.info(
        f"[{audio_path.stem}] Detected language: {info.language} "
        f"(p={info.language_probability:.2f}), "
        f"duration={info.duration:.1f}s"
    )

    segments: list[Segment] = []
    for seg in segments_raw:
        segments.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            avg_logprob=seg.avg_logprob,
        ))

    logger.info(f"[{audio_path.stem}] → {len(segments)} segments")
    return segments


# ---------------------------------------------------------------------------
# Model loader (per-process singleton)
# ---------------------------------------------------------------------------

_model_cache: dict[str, object] = {}


def _load_model(model_path: str, compute_type: str):
    """Load WhisperModel with fallback on compute_type."""
    from faster_whisper import WhisperModel

    global _model_cache
    cache_key = f"{model_path}:{compute_type}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Try primary compute_type, fallback to int8_float16
    for ct in (compute_type, "int8_float16"):
        try:
            logger.info(f"Loading WhisperModel from {model_path} (compute_type={ct})")
            model = WhisperModel(
                model_path,
                device="cuda",
                compute_type=ct,
            )
            _model_cache[cache_key] = model
            return model
        except Exception as exc:
            logger.warning(f"Failed with compute_type={ct}: {exc}")
            if ct == compute_type:
                continue
            raise

    raise RuntimeError(f"Failed to load model from {model_path}")
