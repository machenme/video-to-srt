"""
Bilingual SRT writer — produces a single SRT file with original + translation lines.
"""
from __future__ import annotations

from pathlib import Path

from src.translator.types import SrtCue


def write_bilingual_srt(cues: list[SrtCue], output_path: str | Path) -> None:
    """
    Write bilingual SRT to *output_path*.

    Format per cue:

        1
        00:00:01,234 --> 00:00:05,678
        原文
        译文

    Un-translated cues repeat the original text in the translation slot.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for cue in cues:
        translation = cue.translation or cue.text  # fallback to original
        lines.append(str(cue.index))
        lines.append(f"{cue.start} --> {cue.end}")
        lines.append(cue.text)
        lines.append(translation)
        lines.append("")  # blank separator

    out_path.write_text("\n".join(lines), encoding="utf-8")
