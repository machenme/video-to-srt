"""
Translation pipeline — parse SRT → batch → translate → merge → write.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ISO 639-1 → PotPlayer/VideoPlayer auto-load compatible suffix
_PLAYER_LANG_SUFFIX: dict[str, str] = {
    "zh": "chs",
    "zh-hant": "cht",
    "ja": "jpn",
    "en": "eng",
    "ko": "kor",
    "fr": "fre",
    "de": "ger",
    "es": "spa",
    "pt": "por",
    "it": "ita",
    "ru": "rus",
    "ar": "ara",
    "th": "tha",
    "vi": "vie",
}

from src.translator.types import (
    SrtCue,
    TranslateConfig,
    TranslationError,
    TranslationProvider,
)
from src.translator.parser import parse_srt
from src.translator.writer import write_bilingual_srt

logger = logging.getLogger(__name__)


def translate_srt(
    srt_path: str | Path,
    target_lang: str,
    *,
    provider: TranslationProvider,
    source_lang: str = "auto",
    config: TranslateConfig | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """
    Translate an SRT file end-to-end.

    1. Parse *srt_path* into :class:`SrtCue` list.
    2. Split cues into batches of *config.batch_size*.
    3. Concurrently translate each batch via *provider*.
    4. Merge results back in original order.
    5. Write bilingual SRT.

    Args:
        srt_path: Path to the source SRT.
        target_lang: ISO 639-1 target language code.
        provider: A :class:`TranslationProvider` instance.
        source_lang: ISO 639-1 source language (default ``"auto"``).
        config: Optional tuning overrides.
        output_path: Explicit output path.  Defaults to
            ``{srt_stem}.{target_lang}.srt`` alongside the source.

    Returns:
        Path to the written bilingual SRT file.

    Raises:
        TranslationError: All batches failed or provider is unreachable.
    """
    cfg = config or TranslateConfig()
    file_path = Path(srt_path)

    # --- 1. Parse ---
    cues = parse_srt(file_path)
    total = len(cues)
    logger.info("Parsed %d cues from %s", total, file_path.name)

    # --- 2. Batch ---
    texts = [cue.text for cue in cues]
    batches: list[tuple[int, list[int]]] = []  # [(batch_idx, [cue_indices])]
    for start in range(0, total, cfg.batch_size):
        end = min(start + cfg.batch_size, total)
        batches.append((start // cfg.batch_size, list(range(start, end))))

    logger.info(
        "%d batches (batch_size=%d, workers=%d)",
        len(batches), cfg.batch_size, cfg.max_workers,
    )

    # --- 3. Translate concurrently ---
    failed_batches: list[int] = []

    def _translate_batch(batch_idx: int, indices: list[int]) -> tuple[int, list[str]]:
        """Returns (batch_idx, [translated_lines])."""
        joined = "\n".join(texts[i] for i in indices)
        try:
            result = provider.translate(joined, source_lang, target_lang)
            lines = result.split("\n")
            # Defend against API merging/splitting differently
            if len(lines) != len(indices):
                logger.warning(
                    "Batch %d: expected %d lines, got %d — using best-effort alignment",
                    batch_idx, len(indices), len(lines),
                )
                # Pad or trim to match expected count
                while len(lines) < len(indices):
                    lines.append("")
                lines = lines[: len(indices)]
            return (batch_idx, lines)
        except TranslationError:
            logger.exception("Batch %d translation failed", batch_idx)
            raise

    start_time = time.time()
    results: dict[int, list[str]] = {}

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures: dict[object, int] = {}
        # Stagger submissions to avoid thundering-herd 429
        for idx, indices in batches:
            futures[executor.submit(_translate_batch, idx, indices)] = idx
            if len(futures) < len(batches):  # not the last one
                time.sleep(cfg.request_delay)

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                idx, lines = future.result()
                results[idx] = lines
            except TranslationError:
                failed_batches.append(batch_idx)
                _, indices = batches[batch_idx]
                results[batch_idx] = [texts[i] for i in indices]

    elapsed = time.time() - start_time
    logger.info(
        "Translation done in %.1fs (%d/%d batches ok)%s",
        elapsed,
        len(batches) - len(failed_batches),
        len(batches),
        f", {len(failed_batches)} failed" if failed_batches else "",
    )

    # --- 4. Merge ---
    for batch_idx, (_, indices) in enumerate(batches):
        lines = results.get(batch_idx, [texts[i] for i in indices])
        for j, cue_idx in enumerate(indices):
            cues[cue_idx].translation = lines[j] if j < len(lines) else cues[cue_idx].text

    # --- 5. Write ---
    suffix = _PLAYER_LANG_SUFFIX.get(target_lang, target_lang)
    out = Path(output_path) if output_path else file_path.with_stem(
        f"{file_path.stem}.{suffix}"
    )
    write_bilingual_srt(cues, out)
    logger.info("Bilingual SRT written: %s", out)

    if failed_batches and len(failed_batches) == len(batches):
        raise TranslationError(
            f"All {len(batches)} batches failed for {file_path.name}"
        )

    return out
