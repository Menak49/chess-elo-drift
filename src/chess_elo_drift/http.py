"""The HTTP plumbing both sites' clients share.

Both public APIs are unauthenticated and rate limited in the same spirit:
serial requests over a keep-alive connection are tolerated, parallel callers are
throttled hard. A client therefore owns a single session and is deliberately
synchronous. What differs between the sites -- how long to back off after a 429,
how a body is decoded -- is left to the subclasses.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, TypeVar

import requests

from chess_elo_drift import config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ApiError(RuntimeError):
    """Any unrecoverable failure while talking to a site's API."""


class NotFoundError(ApiError):
    """The resource does not exist.

    This is an expected, routine outcome: a player may simply have no archive
    for a given month, or an account may have been closed since it played.
    """


class JsonHttpClient:
    """A rate-limit-aware, retrying client for one API root.

    Instances are not thread safe by design; run one client per process.
    """

    def __init__(
        self,
        root: str,
        *,
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
        self._session.headers.update({"User-Agent": user_agent})
        self.request_count = 0

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """Fetch `path` and decode its JSON body."""
        return self._request("GET", path, params=params, decode=_decode_json)

    def get_ndjson(self, path: str, *, params: dict[str, Any] | None = None) -> list[Any]:
        """Fetch `path` as newline-delimited JSON and decode every line."""
        return self._request(
            "GET", path, params=params, headers={"Accept": "application/x-ndjson"},
            decode=_decode_ndjson,
        )

    def post_text(self, path: str, body: str, *, params: dict[str, Any] | None = None) -> Any:
        """POST a plain-text body to `path` and decode the JSON answer."""
        return self._request(
            "POST", path, params=params, data=body.encode("utf-8"),
            headers={"Content-Type": "text/plain"}, decode=_decode_json,
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- hooks for subclasses -----------------------------------------------

    def _rate_limit_delay(self, attempt: int) -> float:
        """How long to wait after a 429."""
        return config.BACKOFF_BASE_SECONDS * (2**attempt)

    # -- internals ----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        decode: Callable[[requests.Response], T],
        params: dict[str, Any] | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> T:
        """Send one request, retrying transient failures.

        Raises:
            NotFoundError: the endpoint returned 404 or 410.
            ApiError: the request failed after exhausting the retries.
        """
        url = f"{self._root}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            self._throttle()
            try:
                response = self._session.request(
                    method, url, params=params, data=data, headers=headers, timeout=self._timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                self._sleep(self._backoff(attempt), reason=str(exc))
                continue

            self.request_count += 1

            if response.status_code in (404, 410):
                raise NotFoundError(url)
            if response.status_code == 429:
                last_error = ApiError(f"HTTP 429 for {url}")
                self._sleep(self._rate_limit_delay(attempt), reason="HTTP 429")
                continue
            if response.status_code >= 500:
                last_error = ApiError(f"HTTP {response.status_code} for {url}")
                self._sleep(self._backoff(attempt), reason=f"HTTP {response.status_code}")
                continue
            if not response.ok:
                raise ApiError(f"HTTP {response.status_code} for {url}")

            try:
                return decode(response)
            except ValueError as exc:  # truncated or non-JSON body
                last_error = exc
                self._sleep(self._backoff(attempt), reason="invalid JSON")

        raise ApiError(f"giving up on {url} after {self._max_retries} attempts: {last_error}")

    def _throttle(self) -> None:
        """Keep a floor on the delay between two requests."""
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> float:
        return config.BACKOFF_BASE_SECONDS * (2**attempt)

    @staticmethod
    def _sleep(delay: float, *, reason: str) -> None:
        logger.debug("retrying in %.1fs (%s)", delay, reason)
        time.sleep(delay)


def _decode_json(response: requests.Response) -> Any:
    return response.json()


def _decode_ndjson(response: requests.Response) -> list[Any]:
    # NDJSON is served without a charset, and `response.text` would then guess one.
    body = response.content.decode("utf-8")
    return [json.loads(line) for line in body.splitlines() if line.strip()]
