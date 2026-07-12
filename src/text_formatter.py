"""
Stage 3: Text formatter.
Converts segment lists to SRT, plain text, and Markdown with timestamps.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.utils import format_timestamp


# ---------------------------------------------------------------------------
# Subtitle split constants
# ---------------------------------------------------------------------------

# Japanese sentence-ending punctuation (splits here)
_SENTENCE_END = re.compile(r"[。！？!?\n]")
# Max characters per subtitle line (Japanese)
_MAX_CHARS_PER_SUB = 40
# Max seconds a single subtitle should stay on screen
_MAX_SUB_DURATION = 7.0


def _split_by_length(text: str, t_start: float, t_end: float) -> list[Segment]:
    """Split text into subtitle chunks by char count, distributing time evenly."""
    return _split_by_char_count_with_time(text, t_start, t_end, _MAX_CHARS_PER_SUB)


def _split_by_char_count(text: str, max_chars: int) -> list[str]:
    """Split a string into chunks of at most max_chars characters."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _split_by_char_count_with_time(
    text: str, t_start: float, t_end: float, max_chars: int
) -> list[Segment]:
    """Split text into subtitle-sized chunks and distribute timing proportionally."""
    chunks = _split_by_char_count(text, max_chars)
    if not chunks:
        return []
    dur = t_end - t_start
    total = len(text)
    if total == 0:
        return [Segment(t_start, t_end, text)]
    results: list[Segment] = []
    t = t_start
    for chunk in chunks:
        chunk_dur = min((len(chunk) / total) * dur, _MAX_SUB_DURATION)
        chunk_end = min(t + chunk_dur, t_end)
        results.append(Segment(round(t, 3), round(chunk_end, 3), chunk.strip()))
        t = chunk_end
    return results


# ---------------------------------------------------------------------------
# Segment type (lightweight dict-like, avoids heavy deps)
# ---------------------------------------------------------------------------

class Segment:
    """A single transcribed segment with timestamp and text."""

    __slots__ = ("start", "end", "text", "avg_logprob")

    def __init__(self, start: float, end: float, text: str, avg_logprob: float = 0.0):
        self.start = start
        self.end = end
        self.text = text.strip()
        self.avg_logprob = avg_logprob

    def __repr__(self) -> str:
        return f"Segment({self.start:.1f}-{self.end:.1f}: {self.text[:40]}...)"


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class TextFormatter:
    """Formats a list of Segments into various output formats."""

    # ------------------------------------------------------------------
    # Subtitle-friendly splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split_for_srt(segments: list[Segment]) -> list[Segment]:
        """
        Split long VAD segments into subtitle-friendly short lines.

        Rules:
        - Split on sentence-ending punctuation (。！？)
        - Max ~40 chars per subtitle
        - Max ~7 seconds display duration
        - Timestamps distributed proportionally within the parent segment
        """
        result: list[Segment] = []
        for seg in segments:
            dur = seg.end - seg.start
            if dur <= _MAX_SUB_DURATION and len(seg.text) <= _MAX_CHARS_PER_SUB:
                result.append(seg)
                continue

            # Split on sentence boundaries
            parts = _SENTENCE_END.split(seg.text)
            # Keep the delimiter with the preceding text
            sentences: list[str] = []
            for match in _SENTENCE_END.finditer(seg.text):
                # This is a bit tricky — we need to find actual sentence boundaries
                pass

            # Simpler: find all positions of sentence-ending punctuation
            boundaries: list[int] = []
            text = seg.text
            for m in _SENTENCE_END.finditer(text):
                boundaries.append(m.end())

            if not boundaries:
                # No sentence boundaries — split by char count
                subs = _split_by_length(text, seg.start, seg.end)
                result.extend(subs)
                continue

            # Split by sentence boundaries, merging short sentences together
            chunks: list[str] = []
            chunk_chars: list[int] = []  # char count per chunk
            start = 0
            for b in boundaries:
                sentence = text[start:b]
                if chunks and chunk_chars[-1] + len(sentence) <= _MAX_CHARS_PER_SUB:
                    chunks[-1] += sentence
                    chunk_chars[-1] += len(sentence)
                else:
                    chunks.append(sentence)
                    chunk_chars.append(len(sentence))
                start = b
            # Last piece (no trailing punctuation)
            if start < len(text):
                sentence = text[start:]
                if chunks and chunk_chars[-1] + len(sentence) <= _MAX_CHARS_PER_SUB:
                    chunks[-1] += sentence
                    chunk_chars[-1] += len(sentence)
                else:
                    chunks.append(sentence)
                    chunk_chars.append(len(sentence))

            # Further split any chunk that's too long
            final_chunks: list[str] = []
            for chunk in chunks:
                if len(chunk) > _MAX_CHARS_PER_SUB:
                    final_chunks.extend(_split_by_char_count(chunk, _MAX_CHARS_PER_SUB))
                else:
                    final_chunks.append(chunk)

            # Distribute timestamps proportionally
            total_chars = sum(len(c) for c in final_chunks)
            if total_chars == 0:
                result.append(seg)
                continue

            t = seg.start
            for chunk in final_chunks:
                chunk_dur = (len(chunk) / total_chars) * dur
                # Clamp duration
                chunk_dur = min(chunk_dur, _MAX_SUB_DURATION)
                chunk_end = min(t + chunk_dur, seg.end)

                result.append(Segment(
                    start=round(t, 3),
                    end=round(chunk_end, 3),
                    text=chunk.strip(),
                    avg_logprob=seg.avg_logprob,
                ))
                t = chunk_end

        return result

    # ------------------------------------------------------------------
    # SRT
    # ------------------------------------------------------------------

    @staticmethod
    def to_srt(segments: list[Segment]) -> str:
        """
        Generate standard SRT subtitle content.

        Format:
            1
            00:00:01,234 --> 00:00:05,678
            Text line

            2
            ...
        """
        if not segments:
            return ""

        lines: list[str] = []
        for i, seg in enumerate(segments, start=1):
            start_ts = format_timestamp(seg.start, fmt="srt")
            end_ts = format_timestamp(seg.end, fmt="srt")
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text)
            lines.append("")  # blank separator

        return "\n".join(lines).rstrip("\n") + "\n"

    # ------------------------------------------------------------------
    # Plain text
    # ------------------------------------------------------------------

    @staticmethod
    def to_plaintext(segments: list[Segment]) -> str:
        """
        Generate clean continuous text without timestamps.
        Japanese text: segments are joined with no space.
        """
        return "".join(seg.text for seg in segments).strip() + "\n"

    # ------------------------------------------------------------------
    # Markdown with timestamps
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown(segments: list[Segment]) -> str:
        """
        Generate Markdown with [HH:MM:SS] timestamp markers.
        Useful for human review / correction.
        """
        if not segments:
            return ""

        lines: list[str] = []
        for seg in segments:
            ts = format_timestamp(seg.start, fmt="md")
            lines.append(f"{ts} {seg.text}")

        return "\n".join(lines).strip() + "\n"

    # ------------------------------------------------------------------
    # Segment combining (for chunked parallel transcription)
    # ------------------------------------------------------------------

    @staticmethod
    def combine_chunk_segments(
        chunk_results: list[tuple[float, list[Segment]]]
    ) -> list[Segment]:
        """
        Combine segments from multiple audio chunks with offset correction.

        Only merges segments that *actually overlap in time* (happens at chunk
        boundaries where VAD detects the same speech from both sides).
        Adjacent segments with a gap are kept separate — their original
        faster-whisper timestamps are precise and should be preserved.

        Returns:
            Single sorted list of Segments with corrected absolute timestamps.
        """
        all_segments: list[Segment] = []
        for offset, segments in chunk_results:
            for seg in segments:
                seg.start += offset
                seg.end += offset
                all_segments.append(seg)

        all_segments.sort(key=lambda s: s.start)

        # Only merge segments whose time ranges actually overlap
        # (cross-chunk boundary duplication).
        merged: list[Segment] = []
        for seg in all_segments:
            if merged and seg.start < merged[-1].end:
                # Actual time overlap — merge into previous
                merged[-1].end = max(merged[-1].end, seg.end)
                # If the overlap is substantial, the text is likely duplicate;
                # keep the longer version to avoid double text.
                if seg.end - seg.start > merged[-1].end - merged[-1].start:
                    merged[-1].text = seg.text
                merged[-1].avg_logprob = max(merged[-1].avg_logprob, seg.avg_logprob)
            else:
                merged.append(seg)

        return merged

    # ------------------------------------------------------------------
    # Write all formats
    # ------------------------------------------------------------------

    def write_all(self, segments: list[Segment], base_path: Path, formats: list[str] | None = None) -> list[Path]:
        """
        Write all requested formats to disk.

        Args:
            segments: Transcribed segments.
            base_path: Output base path (e.g. output_dir/video_name). Extensions are appended.
            formats: List of formats to output ("srt", "txt", "md"). Default: all three.

        Returns:
            List of written file paths.
        """
        if formats is None:
            formats = ["srt", "txt", "md"]

        writers = {
            "srt": (".srt", self.to_srt),
            "txt": (".txt", self.to_plaintext),
            "md": (".md", self.to_markdown),
        }

        written: list[Path] = []
        base = Path(base_path)
        # Use the last path component as-is — caller is responsible for passing
        # a stem (no extension).  Don't re-strip via .stem because filenames
        # like "123.com.xxx" contain dots that are NOT an output-format suffix.
        stem = base.name
        parent = base.parent

        for fmt_key in formats:
            if fmt_key not in writers:
                continue
            ext, writer_fn = writers[fmt_key]
            out_path = parent / f"{stem}{ext}"

            # SRT gets subtitle-friendly short segments; TXT/MD get raw text
            if fmt_key == "srt":
                sub_segments = self.split_for_srt(segments)
                content = writer_fn(sub_segments)
            else:
                content = writer_fn(segments)

            out_path.write_text(content, encoding="utf-8")
            written.append(out_path)

        return written
