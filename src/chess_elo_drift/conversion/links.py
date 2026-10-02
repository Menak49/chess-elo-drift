"""Finding the chess.com account a Lichess profile points to.

Only the person can say that a Lichess account and a chess.com account are the
same human, and some do: they paste their chess.com profile into the "links"
field of their Lichess profile or mention it in the bio. Reading those fields is
the only identity evidence available without asking anyone, so precision matters
more than recall here. A wrong link pairs two strangers' ratings and looks
exactly like a right one downstream, whereas a missed link only costs a row.

The rules are therefore strict:

* Only a chess.com URL that carries a username counts. Free text such as
  "chess.com: someone" is ignored because it cannot be told from prose.
* Profile pages of chess.com's titled players (`/players/magnus-carlsen`) are
  article slugs, not account names, and are ignored.
* A profile that points at two different chess.com names is ambiguous (a second
  account, or a friend) and yields no link at all.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: A chess.com username is 3 to 25 letters, digits, underscores or hyphens.
_NAME = r"([A-Za-z0-9][A-Za-z0-9_-]{1,24})"

#: The URL shapes that end in an account name. An optional two-letter locale
#: prefix (`chess.com/es/member/...`) precedes the path on the localised site.
_PATHS = (
    r"member/",  # the current profile page
    r"members/view/",  # legacy profile page
    r"profile/view/",  # legacy profile page
    r"stats/(?:live|daily|puzzles|puzzle-rush|tactics|custom)/(?:[a-z0-9]+/)?",  # stats page
    r"games/archive/",  # game archive page
    r"blog/",  # a member's blog lives under the member's username
)

_LINK = re.compile(
    r"(?<![A-Za-z0-9.-])"  # not the tail of another host such as fakechess.com
    r"(?:https?://)?(?:www\.|api\.)?chess\.com/"
    r"(?:[a-z]{2}(?:-[A-Za-z]{2})?/)?"
    r"(?:" + "|".join(_PATHS) + r")"
    + _NAME
    + r"(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)

#: Path words that the pattern could mistake for a username.
_NOT_A_NAME = frozenset(
    {
        "live", "daily", "puzzles", "tactics", "archive", "view", "member", "members",
        "stats", "blitz", "rapid", "bullet",
    }
)


def parse_chesscom_names(text: str | None) -> list[str]:
    """Every distinct chess.com username `text` links to, lowercase, in order."""
    if not text:
        return []
    names: list[str] = []
    for match in _LINK.finditer(text):
        name = match.group(1).lower()
        if len(name) < 3 or name in _NOT_A_NAME or name in names:
            continue
        names.append(name)
    return names


def chesscom_link(profile: Mapping[str, Any] | None) -> str | None:
    """The chess.com username a Lichess `profile` declares, or None.

    Both the links field and the bio are read. None is returned when there is
    no link and also when the profile names more than one account.
    """
    if not profile:
        return None
    names: list[str] = []
    for field in ("links", "bio"):
        value = profile.get(field)
        if isinstance(value, str):
            for name in parse_chesscom_names(value):
                if name not in names:
                    names.append(name)
    return names[0] if len(names) == 1 else None
