"""HTTP client for the Lichess public API.

Lichess asks API users for one request at a time, and for a full minute of
silence after any 429. The shared client already serialises requests; this
subclass only changes how long it backs off when told to slow down.
"""

from __future__ import annotations

from chess_elo_drift import config
from chess_elo_drift.http import JsonHttpClient


class LichessClient(JsonHttpClient):
    """A rate-limit-aware, retrying client for lichess.org."""

    def __init__(self, *, root: str = config.LICHESS_ROOT, **kwargs) -> None:
        super().__init__(root, **kwargs)
        self._session.headers.update({"Accept": "application/json"})

    def _rate_limit_delay(self, attempt: int) -> float:
        return config.LICHESS_RATE_LIMIT_PAUSE_SECONDS
