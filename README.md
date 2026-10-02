# chess-elo-drift

**What playing strength does a chess rating buy -- year by year from 2014 to 2026,
on chess.com and on Lichess -- and how do the four big online pools convert into
one another in 2026?**

Rating systems are relative: a rating says how you do against the pool, not how
well you play. If the pool's strength drifts, a number that stays put does not.
This study measures the playing strength behind the number directly, by
re-analysing real games with one fixed engine, and asks three questions:

1. **Year by year.** Does a 1500 in 2014 play like a 1500 in 2026? Asked
   separately for chess.com blitz, chess.com rapid, Lichess blitz and Lichess
   rapid.
2. **Site against site.** Which Lichess rating plays like a given chess.com
   rating, in each year and cadence?
3. **The 2026 conversion.** Formulas and tables between chess.com rapid,
   chess.com blitz, Lichess rapid and Lichess blitz.

The deliverable is [`reports/findings.md`](reports/findings.md), written entirely
from the data by the `report` command.

## How the questions are made answerable

**The metric is recomputed, not read off.** Neither site's own accuracy figure
can be compared across years or sites: chess.com's exists only for games
somebody reviewed and its formula changed over the years, and Lichess's is a
different formula again. Every game is therefore re-analysed locally with the
same engine at the same fixed depth (`config.ENGINE_DEPTH`), and accuracy,
average centipawn loss and error counts are derived from that single pass.

**Every year is sampled the same way.** Each (site, year) is crawled from the
same calendar months (March, June, September), with an equal quota per month,
and every model also controls for the calendar month, so seasonal differences
in who plays do not pass for a change over the years. Blitz and rapid are never pooled, nor are the two sites. Only
ratings in 600-2400 are studied, each side of a game at the rating it carried
*on that game*, and only games long enough to say something (`config.MIN_PLIES`).

**The sampling is stratified, because neither API offers a random player.** The
crawler walks the opponent graph: every archive it reads reveals more accounts
together with the rating each carried at that moment. The walk is steered toward
whichever 200-point band is furthest from its quota, so the whole window fills
instead of piling up in the middle. A band that cannot be filled -- Lichess had a
rating floor of 800 for years, chess.com rapid barely reached 2200 while it meant
fifteen-minute games -- stops the crawl after a run of requests that fill nothing,
rather than draining the budget.

**Pools are compared only while they existed as pools.** Two rating systems
changed shape inside the period, and the study follows each site's own record of
which pool a game was rated in:

- chess.com moved 10|0 from blitz to rapid on 2020-09-10. The API keeps the class
  a game was rated in at the time, so a 2016 10|0 game counts as blitz, as its
  rating did.
- Lichess split Rapid out of Classical on 2017-12-01, seeding it with copies of
  Classical ratings. Its API labels old games by today's speed rules, so a 2016
  10+0 game comes back as "rapid" while its rating is a Classical one; such games
  are excluded (`config.POOL_START`), and the Lichess rapid series starts in 2018.

**Clocks are held fixed.** The mix of time controls inside a cadence changes over
the years (above all at chess.com's 2020 reclassification), and longer games are
played more accurately. Every model controls for the estimated game length
(base + 40 x increment), and a robustness check keeps only chess.com time
controls whose class never changed (`config.STABLE_TIME_CONTROLS`).

**The year effect comes from a regression, not a difference of means.** Years
can sit at different points inside the same rating band, and a band-mean
comparison would read that as a change in skill. Rating enters continuously, one
model per site and cadence:

```
accuracy ~ year + rating + rating² + log(clock)      (errors clustered by player)
```

The report gives each year's accuracy at fixed ratings, the year effects against
2026, a linear trend, trends on each side of chess.com's 2020 break, and whether
the rating slope itself drifted. A band-by-year table is kept beside it as a
shape check.

**Sites are compared by equating accuracy.** For each year and cadence, each
site's accuracy-rating curve is fitted at the same clock, and a chess.com rating
is matched to the Lichess rating that plays as accurately, with a bootstrap over
players.

**The 2026 conversion pairs people, not moves.** Converting between pools needs
the same person rated in both:

- *rapid <-> blitz on one site*: current ratings of thousands of active
  accounts (`survey`), with same-month rating snapshots from the crawl as a
  cross-check. On chess.com, no single account's opponents may make up more than
  a dozen of the sample: opponents are matched on one cadence's rating, and a
  sample selected on one rating biases any symmetric conversion.
- *chess.com <-> Lichess*: Lichess profiles that link their owner's chess.com
  account -- a self-declared, self-selected, but direct pairing.

Only established ratings count (enough games, a low rating deviation, recent
play). The tables use equipercentile linking (a rating maps to the rating at the
same percentile among the paired people), which follows a curved relation and
is exactly invertible; the printed formula is its straight-line version, with the
range it is valid for. Outliers (mostly wrong links) are screened with a robust
rule and counted. The conversion is cross-checked against a second source, against
chained routes through a third pool, against the accuracy equating above, and
against the published ChessGoals survey.

## Setup

```bash
pip install -r requirements.txt
pip install -e .                  # makes the package importable from anywhere
```

Commands are then run as `python -m chess_elo_drift.cli ...`; without installing
the package, prefix them with `PYTHONPATH=src`. The install also declares a
`chess-elo-drift` console script, but writing it needs permission on the
interpreter's `Scripts` directory, which a system-wide Python install does not
grant; inside a virtualenv it lands normally.

The engine is not vendored. Download a Stockfish build and put the binary at
`tools/stockfish/stockfish.exe` (or point `--engine` elsewhere). Any recent
version works; what matters is that a single one is used for the whole run,
since it is the measuring instrument. This study's run used Stockfish 19.

## Running the study

Four commands, each resumable, each writing where the next one reads:

```bash
python -m chess_elo_drift.cli collect    # both APIs   -> data/raw/<site>/games_<year>.jsonl
python -m chess_elo_drift.cli survey     # both APIs   -> data/raw/conversion/
python -m chess_elo_drift.cli evaluate   # stored PGNs -> data/processed/evaluations.csv
python -m chess_elo_drift.cli report     # everything  -> reports/
```

**collect** fills a quota per (cadence x 200-point band) for every site and year.
`--platform chesscom|lichess|both` and `--years 2014-2026` (or `2018,2026`) pick
what to crawl, so the two sites can run as two processes at once. Re-running tops
the corpus up rather than doubling it. Lichess asks for one request at a time and
locks an address out of its streaming endpoints for about an hour after a stream
is abandoned, so never run two Lichess jobs at once or kill one mid-request.
Useful flags: `--per-cell`, `--max-requests`, `--min-interval`.

**survey** records current ratings for the rapid/blitz pairings and follows every
chess.com link found on a Lichess profile. It paces Lichess's bulk user endpoint
under its limit (8,000 accounts per 10 minutes).

**evaluate** scores games on every core, writing rows as they complete, and skips
games already present in the output, so an interrupted run resumes for free. A
worker whose engine dies starts a fresh one. `--workers` sets the parallelism
(lower it on a machine short of memory); `--limit N` spends a fixed engine budget
on a subsample spread evenly across the design.

**report** builds the estimation sample, runs every comparison and writes:

```
reports/findings.md                # the answers, with method and caveats
reports/tables/*.csv               # every table behind them, full precision
reports/figures/*.png              # accuracy by year, site offsets, conversions
```

## Layout

```
src/chess_elo_drift/
  config.py           every tunable of the protocol, in one immutable place
  records.py          GameRecord (one game, two sides) and RatingSnapshot
  http.py             the retrying, throttled client both sites share
  selection.py        spending a fixed engine budget evenly across the design
  chesscom/           chess.com client and endpoint wrappers
  lichess/            Lichess client, game export, extraction and seeding
  collection/         the stratified snowball crawler, its sources and stores
  conversion/         the 2026 rating survey and cross-site account links
  engine/             UCI analysis, accuracy/ACPL scoring, parallel runner
  analysis/           estimation sample, yearly models, site equating,
                      conversion fits, figures, findings note
tests/                unit tests; no network and no engine required
```

## Testing

```bash
python -m pytest
```

The suite is offline: the APIs are faked and no engine is started.

## Known limits

- **Snowball sampling is not a random sample of the population.** It is steered
  to cover the rating window evenly, which is what the questions need, but the
  accounts reached are those connected to the seeds by play, weighted toward the
  active. Results describe the opposition you would meet at a rating, not every
  account on a site.
- **Survivorship grows with age.** A 2014 game can only be read from an account
  that still exists; players who closed theirs are invisible.
- **Rating is the rating at the time of that game.** Provisional Lichess ratings
  are dropped; chess.com exposes no per-game equivalent.
- **Accuracy is one operationalisation of strength.** It is computed from a
  fixed-depth engine's evaluation swing per move; a deeper search would give
  different absolute numbers. Only comparisons are claimed, and they are valid
  because the instrument is identical everywhere.
- **Rating systems changed along the way** -- floors, starting ratings, Glicko
  parameters, cadence boundaries. A year effect combines pool composition with
  those administrative changes; the report lists the known ones.
- **The opening is not scored** (`config.OPENING_PLIES_SKIPPED`). Book moves are
  memorised rather than calculated, and the availability of preparation has
  itself changed over the years.
