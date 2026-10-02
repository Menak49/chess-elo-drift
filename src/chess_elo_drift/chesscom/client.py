"""HTTP client for the chess.com public API.

The public API is unauthenticated but rate limited in a specific way: serial
requests over a keep-alive connection are tolerated generously, while parallel
callers are throttled hard. The retrying, throttled plumbing is shared with the
Lichess client and lives in `chess_elo_drift.http`.
"""

from __future__ import annotations

from chess_elo_drift import config
from chess_elo_drift.http import ApiError, JsonHttpClient, NotFoundError

#: Kept as the chess.com-facing name for the shared error type.
ChessComError = ApiError

__all__ = ["ChessComClient", "ChessComError", "NotFoundError"]


class ChessComClient(JsonHttpClient):
    """A rate-limit-aware, retrying JSON client for api.chess.com."""

    def __init__(self, *, root: str = config.API_ROOT, **kwargs) -> None:
        super().__init__(root, **kwargs)
        self._session.headers.update({"Accept": "application/json"})
