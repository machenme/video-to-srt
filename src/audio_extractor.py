"""
Stage 1: Audio extraction via ffmpeg subprocess.
Extracts audio stream from video files → 16kHz Mono 16-bit PCM WAV.
"""
from __future__ import annotations

import subprocess
import logging
from pathlib import Path

from src.utils import is_file_valid

logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a video."""


class AudioExtractor:
    """
    Extracts audio from video files using ffmpeg (subprocess).

    Output format: 16 kHz, mono, 16-bit PCM WAV (Whisper native format).
    """

    def __init__(self, temp_dir: Path, ffmpeg_bin: str = "ffmpeg"):
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg = ffmpeg_bin

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def extract(self, video_path: Path, output_dir: Path | None = None) -> Path:
        """
        Extract audio from a single video file.

        Args:
            video_path: Source video file path.
            output_dir: Directory for the output WAV (defaults to self._temp_dir).

        Returns:
            Path to the extracted WAV file.

        Raises:
            AudioExtractionError: ffmpeg call failed.
        """
        dest_dir = output_dir or self._temp_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        wav_name = video_path.stem + ".wav"
        wav_path = dest_dir / wav_name

        # Skip if already extracted and valid
        if is_file_valid(wav_path, min_bytes=1024):
            logger.info(f"Audio already extracted: {wav_path}")
            return wav_path

        logger.info(f"Extracting audio: {video_path.name} → {wav_name}")

        cmd = [
            self._ffmpeg,
            "-y",                          # overwrite
            "-threads", "1",               # single-threaded: better error resilience
            "-err_detect", "ignore_err",   # tolerate corrupt packets
            "-fflags", "+genpts+discardcorrupt",  # regenerate timestamps, skip broken frames
            "-i", str(video_path),
            "-vn",                         # no video
            "-c:a", "pcm_s16le",           # 16-bit PCM (re-encode, not stream copy)
            "-ar", "16000",                # 16 kHz
            "-ac", "1",                    # mono
            "-af", "aresample=async=1",    # fill gaps from corrupt frames with silence
            "-loglevel", "error",          # suppress ffmpeg output except errors
            str(wav_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                # ffmpeg exits non-zero on decode errors (corrupt source).
                # If output file exists and is usable, treat as partial success.
                if is_file_valid(wav_path, min_bytes=1024):
                    logger.warning(
                        f"Audio extracted with decode errors: {video_path.name} "
                        f"(output {wav_path.stat().st_size} bytes)"
                    )
                    # If the output duration is suspiciously short, try raw-AAC fallback.
                    actual_dur = self.get_duration(wav_path)
                    expected_dur = self.get_video_duration(video_path)
                    if expected_dur > 0 and actual_dur < expected_dur * 0.5:
                        logger.warning(
                            f"Extracted audio too short ({actual_dur:.0f}s vs {expected_dur:.0f}s). "
                            f"Trying raw-AAC fallback..."
                        )
                        return self._extract_via_raw_aac(video_path, wav_path)
                else:
                    # First attempt failed entirely; try raw-AAC fallback before giving up.
                    try:
                        return self._extract_via_raw_aac(video_path, wav_path)
                    except AudioExtractionError:
                        stderr = result.stderr.strip()
                        raise AudioExtractionError(
                            f"ffmpeg failed for {video_path.name}: {stderr}"
                        )
        except subprocess.TimeoutExpired:
            raise AudioExtractionError(f"ffmpeg timed out for {video_path.name}")
        except FileNotFoundError:
            raise AudioExtractionError(
                "ffmpeg not found. Please install ffmpeg and ensure it's on PATH."
            )

        if not is_file_valid(wav_path, min_bytes=1024):
            raise AudioExtractionError(f"Output WAV is missing or too small: {wav_path}")

        return wav_path

    # ------------------------------------------------------------------
    # Audio duration
    # ------------------------------------------------------------------

    def get_duration(self, path: Path) -> float:
        """Return media duration in seconds via ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
        except (ValueError, subprocess.TimeoutExpired):
            logger.warning(f"Could not determine duration for {path.name}, assuming 0")
            return 0.0

    def get_video_duration(self, video_path: Path) -> float:
        """Return the *container-level* video duration (may differ from decoded audio)."""
        return self.get_duration(video_path)

    # ------------------------------------------------------------------
    # Raw AAC fallback (for corrupt audio streams)
    # ------------------------------------------------------------------

    def _extract_via_raw_aac(self, video_path: Path, wav_path: Path) -> Path:
        """
        Two-step extraction for corrupt AAC streams:
        1. Stream-copy raw AAC out of the container (avoids decoder).
        2. Decode the raw AAC with error tolerance.

        Stripping the MP4/MKV container often lets the AAC decoder
        recover from errors that were fatal inside the container.
        """
        aac_path = wav_path.with_suffix(".aac")
        logger.info(f"Step 1/2: extracting raw AAC stream: {aac_path.name}")

        # Step 1: extract raw AAC
        cmd1 = [
            self._ffmpeg, "-y",
            "-err_detect", "ignore_err",
            "-i", str(video_path),
            "-vn", "-c:a", "copy",
            "-f", "adts",
            "-loglevel", "error",
            str(aac_path),
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
        if not is_file_valid(aac_path, min_bytes=1024):
            raise AudioExtractionError(
                f"Raw AAC extraction produced no usable output for {video_path.name}"
            )

        # Step 2: decode raw AAC → WAV with error tolerance
        logger.info(f"Step 2/2: decoding raw AAC → WAV")
        cmd2 = [
            self._ffmpeg, "-y",
            "-threads", "1",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts",
            "-i", str(aac_path),
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-af", "aresample=async=1",
            "-loglevel", "error",
            str(wav_path),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)

        # Clean up intermediate AAC
        try:
            aac_path.unlink()
        except OSError:
            pass

        if is_file_valid(wav_path, min_bytes=1024):
            dur = self.get_duration(wav_path)
            logger.info(f"Raw-AAC fallback produced {wav_path.stat().st_size} bytes ({dur:.0f}s)")
            return wav_path

        stderr = result2.stderr.strip()
        raise AudioExtractionError(
            f"Raw-AAC decode also failed for {video_path.name}: {stderr}"
        )

    # ------------------------------------------------------------------
    # Audio chunking (for parallel transcription of long videos)
    # ------------------------------------------------------------------

    def split_wav(self, wav_path: Path, chunk_duration: int) -> list[tuple[float, Path]]:
        """
        Split a WAV file into fixed-duration chunks using ffmpeg segment muxer.

        Args:
            wav_path: Path to the full 16kHz mono WAV.
            chunk_duration: Max seconds per chunk.

        Returns:
            List of (offset_seconds, chunk_wav_path) sorted by offset.
            Returns [(0.0, wav_path)] if the audio is shorter than chunk_duration.
        """
        duration = self.get_duration(wav_path)
        if duration <= chunk_duration:
            logger.info(f"Audio {wav_path.name} ({duration:.0f}s) within chunk limit, no split")
            return [(0.0, wav_path)]

        chunk_dir = wav_path.parent / f"{wav_path.stem}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(chunk_dir / f"{wav_path.stem}_%03d.wav")

        logger.info(f"Splitting {wav_path.name} ({duration:.0f}s) into {chunk_duration}s chunks")
        cmd = [
            self._ffmpeg, "-y",
            "-i", str(wav_path),
            "-f", "segment",
            "-segment_time", str(chunk_duration),
            "-c", "copy",
            "-loglevel", "error",
            pattern,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise AudioExtractionError(
                f"ffmpeg segment failed for {wav_path.name}: {result.stderr.strip()}"
            )

        # Collect chunks sorted by name (which encodes order)
        chunk_files = sorted(chunk_dir.glob(f"{wav_path.stem}_*.wav"))
        chunks: list[tuple[float, Path]] = []
        for i, cf in enumerate(chunk_files):
            offset = i * chunk_duration
            chunks.append((offset, cf))
            logger.debug(f"  Chunk {i}: offset={offset}s, file={cf.name}")

        logger.info(f"Split into {len(chunks)} chunk(s)")
        return chunks

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        video_paths: list[Path],
        output_dir: Path | None = None,
    ) -> list[Path]:
        """
        Batch extract audio from multiple video files.
        Individual failures are logged but do not stop the batch.

        Returns:
            List of successfully extracted WAV paths (in the same order as input,
            with failed entries omitted).
        """
        wav_paths: list[Path] = []
        for vp in video_paths:
            try:
                wav = self.extract(vp, output_dir)
                wav_paths.append(wav)
            except AudioExtractionError as exc:
                logger.error(str(exc))
        return wav_paths
