"""
Utility functions: video file scanning, file integrity checks, SRT validation.
"""
from __future__ import annotations

import re
from pathlib import Path


def scan_video_files(
    directory: Path,
    extensions: list[str],
    recursive: bool = True,
) -> list[Path]:
    """
    Recursively scan a directory for video files matching the given extensions.

    Returns a sorted list of absolute paths.
    """
    ext_set = {e.lower().lstrip(".") for e in extensions}
    pattern = "*" if recursive else "[!.]*"
    results: list[Path] = []

    for ext in ext_set:
        glob_pattern = f"**/{pattern}.{ext}" if recursive else f"{pattern}.{ext}"
        results.extend(directory.glob(glob_pattern))

    # Deduplicate and sort
    seen: set[str] = set()
    unique: list[Path] = []
    for p in sorted(results, key=lambda x: x.name):
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p.resolve())
    return unique


def is_file_valid(path: Path, min_bytes: int = 100) -> bool:
    """Check if a file exists, is > min_bytes, and is readable."""
    try:
        return path.exists() and path.is_file() and path.stat().st_size > min_bytes
    except OSError:
        return False


def is_srt_valid(srt_path: Path) -> bool:
    """
    Basic SRT structural validation.
    Checks that the file is non-empty and has at least one properly formed entry.
    """
    if not is_file_valid(srt_path, min_bytes=50):
        return False

    try:
        content = srt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    # SRT entry pattern: number, timestamp line, at least one text line, blank line
    # Broad check: must have at least one timestamp line
    timestamp_pattern = re.compile(
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}"
    )
    return bool(timestamp_pattern.search(content))


def output_exists_and_valid(video_name: str, output_dir: Path) -> bool:
    """
    Check if all expected output files for a video exist and are valid.
    Returns True only if SRT (required) exists and passes validation.
    """
    srt_path = output_dir / f"{video_name}.srt"
    return is_srt_valid(srt_path)


def format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """
    Convert seconds to timestamp string.

    fmt='srt':  HH:MM:SS,mmm
    fmt='md':   [HH:MM:SS]
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)

    if fmt == "srt":
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    else:
        return f"[{h:02d}:{m:02d}:{s:02d}]"
