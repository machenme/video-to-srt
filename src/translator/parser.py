"""
SRT file parser — reads a standard SRT file into a list of :class:`SrtCue`.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.translator.types import ParseError, SrtCue

# "HH:MM:SS,mmm --> HH:MM:SS,mmm"  (comma or dot separator)
_TIME_LINE = re.compile(
    r"^(\d{2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def parse_srt(path: str | Path) -> list[SrtCue]:
    """
    Parse an SRT file.

    Returns:
        List of :class:`SrtCue` ordered by appearance (1-based index).

    Raises:
        ParseError: The file is empty or structurally invalid.
        FileNotFoundError: *path* does not exist.
    """
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    cues: list[SrtCue] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # skip blank lines between cues
        if not line:
            i += 1
            continue

        # expect cue number
        try:
            index = int(line)
        except ValueError:
            raise ParseError(
                f"Expected cue number at line {i + 1}, got: {line!r}"
            )

        i += 1
        if i >= len(lines):
            raise ParseError(f"Cue {index}: missing timestamp line")

        # timestamp line
        m = _TIME_LINE.match(lines[i].strip())
        if not m:
            raise ParseError(
                f"Cue {index}: invalid timestamp at line {i + 1}: "
                f"{lines[i].strip()!r}"
            )
        start, end = m.group(1), m.group(2)

        i += 1
        # collect text lines until blank or EOF
        text_parts: list[str] = []
        while i < len(lines) and lines[i].strip():
            text_parts.append(lines[i].strip())
            i += 1

        if not text_parts:
            raise ParseError(f"Cue {index}: no text at line {i + 1}")

        cues.append(SrtCue(
            index=index,
            start=start,
            end=end,
            text="\n".join(text_parts),
        ))

    if not cues:
        raise ParseError(f"SRT file is empty or contains no valid cues: {file_path}")

    return cues
