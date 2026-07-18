"""
Microsoft Edge API translator — free, zero-registration translation backend.

Token lifecycle
    JWT obtained from ``edge.microsoft.com/translate/auth``, cached for
    *token_ttl* seconds (default 480 = 8 min).  A 401/403 response triggers
    a single transparent refresh + retry.

Rate limiting
    Edge API has a per-IP rate limit.  This implementation uses a **shared
    cooldown gate** (class-level ``threading.Lock``) — once any thread gets a
    429, all threads pause for a backoff window before retrying.  Up to 3
    retries with exponential backoff (1s → 2s → 4s).
"""
from __future__ import annotations

import logging
import threading
import time

import requests

from src.translator.types import TranslationError, TranslationProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUTH_URL = "https://edge.microsoft.com/translate/auth"
_TRANSLATE_URL = (
    "https://api-edge.cognitive.microsofttranslator.com/translate"
    "?api-version=3.0"
)
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds


class EdgeTranslator:
    """
    Microsoft Edge translation backend.

    Implements the :class:`TranslationProvider` protocol.

    Thread-safe: JWT cache and rate-limit cooldown are protected by locks.
    """

    # Shared rate-limit gate across all instances (class-level)
    _rate_lock = threading.Lock()
    _rate_limit_until: float = 0.0

    def __init__(self, token_ttl: int = 480) -> None:
        self._token_ttl = token_ttl
        self._token_value: str | None = None
        self._token_expires: float = 0.0
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # TranslationProvider protocol
    # ------------------------------------------------------------------

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "zh"
    ) -> str:
        """
        Translate *text* via the Edge API.

        Automatically retries on 429 (rate-limit) with exponential backoff.
        """
        url = f"{_TRANSLATE_URL}&to={target_lang}"
        if source_lang and source_lang != "auto":
            url += f"&from={source_lang}"

        body = [{"Text": text}]
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES + 1):
            # --- shared rate-limit gate ---
            with EdgeTranslator._rate_lock:
                wait = EdgeTranslator._rate_limit_until - time.time()
            if wait > 0:
                logger.debug("Rate-limit cooldown: waiting %.1fs", wait)
                time.sleep(wait)

            token = self._get_token()
            headers = {"Authorization": f"Bearer {token}"}

            try:
                resp = self._session.post(
                    url, json=body, headers=headers, timeout=30,
                )
            except requests.RequestException as exc:
                last_exc = TranslationError(f"Edge API request failed: {exc}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                continue

            # Auth refresh
            if resp.status_code in (401, 403):
                logger.info("Edge token rejected (401/403), refreshing")
                token = self._fetch_fresh_token()
                headers["Authorization"] = f"Bearer {token}"
                try:
                    resp = self._session.post(
                        url, json=body, headers=headers, timeout=30,
                    )
                except requests.RequestException as exc:
                    last_exc = TranslationError(f"Edge API retry failed: {exc}")
                    continue

            # Rate-limit — set shared cooldown and retry
            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                backoff = retry_after if retry_after else _RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Edge API 429 rate-limited, backoff %.1fs (attempt %d/%d)",
                    backoff, attempt + 1, _MAX_RETRIES,
                )
                with EdgeTranslator._rate_lock:
                    EdgeTranslator._rate_limit_until = time.time() + backoff
                last_exc = TranslationError(
                    f"Edge translate rate-limited (429)"
                )
                time.sleep(backoff)
                continue

            if not resp.ok:
                snippet = resp.text[:200]
                raise TranslationError(
                    f"Edge translate HTTP {resp.status_code}: {snippet}"
                )

            try:
                data = resp.json()
                translated = data[0]["translations"][0]["text"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise TranslationError(
                    f"Invalid response format from Edge Translate: {exc}"
                ) from exc

            return translated

        raise last_exc or TranslationError("Edge translate failed after retries")

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Return a valid cached token or fetch a fresh one."""
        with self._lock:
            if (self._token_value is not None
                    and time.time() < self._token_expires):
                return self._token_value
        return self._fetch_fresh_token()

    def _fetch_fresh_token(self) -> str:
        """Fetch a new JWT and update the cache."""
        try:
            resp = self._session.get(_AUTH_URL, timeout=10)
        except requests.RequestException as exc:
            raise TranslationError(
                f"Edge auth request failed: {exc}"
            ) from exc

        if not resp.ok:
            raise TranslationError(
                f"Edge auth HTTP {resp.status_code}: {resp.text[:200]}"
            )

        value = resp.text.strip()
        with self._lock:
            self._token_value = value
            self._token_expires = time.time() + self._token_ttl - 60
        logger.debug("Edge token refreshed (TTL=%ds)", self._token_ttl)
        return value


def _parse_retry_after(value: str | None) -> float | None:
    """Parse Retry-After header (seconds or HTTP-date)."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
