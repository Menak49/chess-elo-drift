"""Data collection for the conversion study.

The conversion study puts four rating pools (chess.com rapid and blitz, Lichess
rapid and blitz) on one scale for 2026. Nothing here computes a statistic; the
package only gathers the paired ratings the statistics need, in three files:

    chesscom_stats.jsonl   one chess.com player's rapid and blitz ratings
    lichess_users.jsonl    one Lichess player's rapid and blitz ratings
    linked_accounts.jsonl  one person's ratings on both sites

`survey.collect_survey` is the entry point.
"""
