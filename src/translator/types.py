"""
Translation domain types — zero external dependencies.

Protocol + dataclasses shared across the translator sub-package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TranslationError(Exception):
    """Raised when a translation provider fails irrecoverably."""


class ParseError(Exception):
    """Raised when SRT parsing fails (malformed input)."""


# ---------------------------------------------------------------------------
# SRT cue
# ---------------------------------------------------------------------------

@dataclass
class SrtCue:
    """A single parsed SRT subtitle entry."""

    index: int            # 1-based cue number
    start: str            # "HH:MM:SS,mmm"
    end: str              # "HH:MM:SS,mmm"
    text: str             # original transcribed text
    translation: str = ""  # filled after translation; empty = not translated


# ---------------------------------------------------------------------------
# Translation provider protocol
# ---------------------------------------------------------------------------

class TranslationProvider(Protocol):
    """Protocol for pluggable translation backends."""

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate *text* (may contain \\n-separated lines).

        Args:
            text: Source text.  Newlines are preserved in the output.
            source_lang: ISO 639-1 code or ``"auto"``.
            target_lang: ISO 639-1 code.

        Returns:
            Translated text with the same newline structure.

        Raises:
            TranslationError: The provider could not complete the request.
        """
        ...


# ---------------------------------------------------------------------------
# Translation configuration
# ---------------------------------------------------------------------------

@dataclass
class TranslateConfig:
    """Tunables for the batch-translation pipeline."""

    target_lang: str = ""        # "" = skip translation
    source_lang: str = "auto"
    batch_size: int = 50
    max_workers: int = 2
    request_delay: float = 3.0   # seconds between batch submissions (Edge API rate-limit)
    token_ttl: int = 480         # seconds before token refresh
