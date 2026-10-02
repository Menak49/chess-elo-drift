"""Thin, typed access layer over the Lichess public API."""

from chess_elo_drift.lichess.client import LichessClient

__all__ = ["LichessClient"]
