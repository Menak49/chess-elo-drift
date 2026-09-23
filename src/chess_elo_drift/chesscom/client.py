"""HTTP client for the chess.com public API.

The public API is unauthenticated but rate limited in a specific way: serial
requests over a keep-alive connection are tolerated generously, while parallel
callers are throttled hard. The client therefore owns a single session and is
deliberately synchronous.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from chess_elo_drift import config

logger = logging.getLogger(__name__)


class ChessComError(RuntimeError):
    """Any unrecoverable failure while talking to the API."""


class NotFoundError(ChessComError):
    """The resource does not exist.

    This is an expected, routine outcome: a player may simply have no archive
    for a given month, or an account may have been closed since it played.
    """


class ChessComClient:
    """A rate-limit-aware, retrying JSON client.

    Instances are not thread safe by design; run one client per process.
    """

    def __init__(
        self,
        *,
        root: str = config.API_ROOT,
        user_agent: str = config.USER_AGENT,
        timeout: float = config.REQUEST_TIMEOUT_SECONDS,
        max_retries: int = config.MAX_RETRIES,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self._root = root.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._min_interval = min_interval_seconds
        self._last_request_at = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        self.request_count = 0

    def get(self, path: str) -> dict[str, Any]:
        """Fetch `path` relative to the API root and return the decoded body.

        Raises:
            NotFoundError: the endpoint returned 404 or 410.
            ChessComError: the request failed after exhausting the retries.
        """
        url = f"{self._root}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            self._throttle()
            try:
                response = self._session.get(url, timeout=self._timeout)
            except requests.RequestException as exc:
                last_error = exc
                self._sleep_before_retry(attempt, reason=str(exc))
                continue

            self.request_count += 1

            if response.status_code in (404, 410):
                raise NotFoundError(url)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = ChessComError(f"HTTP {response.status_code} for {url}")
                self._sleep_before_retry(attempt, reason=f"HTTP {response.status_code}")
                continue
            if not response.ok:
                raise ChessComError(f"HTTP {response.status_code} for {url}")

            try:
                return response.json()
            except ValueError as exc:  # truncated or non-JSON body
                last_error = exc
                self._sleep_before_retry(attempt, reason="invalid JSON")

        raise ChessComError(f"giving up on {url} after {self._max_retries} attempts: {last_error}")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "ChessComClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    def _throttle(self) -> None:
        """Keep a floor on the delay between two requests."""
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _sleep_before_retry(self, attempt: int, *, reason: str) -> None:
        delay = config.BACKOFF_BASE_SECONDS * (2**attempt)
        logger.debug("retrying in %.1fs (%s)", delay, reason)
        time.sleep(delay)
