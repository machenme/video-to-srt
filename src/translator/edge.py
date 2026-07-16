"""
Microsoft Edge API translator — free, zero-registration translation backend.

Token lifecycle
    JWT obtained from ``edge.microsoft.com/translate/auth``, cached for
    *token_ttl* seconds (default 480 = 8 min).  A 401/403 response triggers
    a single transparent refresh + retry.
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


class EdgeTranslator:
    """
    Microsoft Edge translation backend.

    Implements the :class:`TranslationProvider` protocol.
    """

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

        Newlines in *text* are preserved; the API returns a single string
        with the same newline structure.
        """
        token = self._get_token()
        url = f"{_TRANSLATE_URL}&to={target_lang}"
        if source_lang and source_lang != "auto":
            url += f"&from={source_lang}"

        headers = {"Authorization": f"Bearer {token}"}
        body = [{"Text": text}]

        try:
            resp = self._session.post(
                url, json=body, headers=headers, timeout=30,
            )
        except requests.RequestException as exc:
            raise TranslationError(f"Edge API request failed: {exc}") from exc

        # Transparent token refresh on auth error
        if resp.status_code in (401, 403):
            logger.info("Edge token rejected (401/403), refreshing and retrying")
            token = self._fetch_fresh_token()
            headers["Authorization"] = f"Bearer {token}"
            try:
                resp = self._session.post(
                    url, json=body, headers=headers, timeout=30,
                )
            except requests.RequestException as exc:
                raise TranslationError(
                    f"Edge API retry failed: {exc}"
                ) from exc

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
        """Fetch a new JWT and update the cache (caller must NOT hold lock)."""
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
            self._token_expires = time.time() + self._token_ttl - 60  # 1-min safety margin
        logger.debug("Edge token refreshed (TTL=%ds)", self._token_ttl)
        return value
