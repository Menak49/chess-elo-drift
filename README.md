# chess-elo-drift

**Did a given chess.com rating mean the same playing strength in 2018-2019 as it does in 2024-2025?**

Rating systems are relative: a rating says how you do against the pool, not how
well you play. If the pool's strength drifts, a number that stays put does not.
This study measures the playing strength behind the number directly, by
re-analysing real games from both periods with one fixed engine, and asks
whether 1500 in 2018 buys the same quality of moves as 1500 today.

## How the question is made answerable

**The metric is recomputed, not read off.** chess.com publishes an accuracy
figure, but it is available for only a minority of older games, and the formula
behind it has changed over the years. A measuring instrument that drifts between
the two things being compared is useless here. Every game is therefore
re-analysed locally with the same engine at the same fixed depth
(`config.ENGINE_DEPTH`), and accuracy, average centipawn loss and error counts
are derived from that single pass. Where chess.com's own number does exist, it
is used as a cross-check on the recomputed one — `validate_against_reported` —
and its patchy coverage is reported rather than asserted.

**The two samples are matched by design.** Both eras are sampled from the same
calendar months (March, June, September), so seasonal differences in who is
playing — school terms, holidays, the January influx of new accounts — cancel
out. Blitz and rapid are never pooled: they are different scales on chess.com.
Only ratings in 600-2200 are studied, and only games long enough to say
something (`config.MIN_PLIES`).

**The sampling is stratified, because the API offers no random player.** The
crawler walks the opponent graph: every archive it reads reveals more accounts
together with the rating each carried at that moment. The walk is steered toward
whichever rating band is furthest from its quota, so the whole window fills
instead of piling up around 1000.

**The analysis decisions are made in the open, after collection.** The crawler
over-gathers by design, so a few very active accounts appear far more often than
anyone else; capping their weight is an analysis choice and lives in
`analysis/dataset.py`, not in the crawler. Standard errors are clustered by
player throughout, since the opponent-graph walk returns the same account more
than once and treating those rows as independent would understate the
uncertainty.

**The headline number comes from a regression, not a difference of means.** Two
eras can sit at different points inside the same rating band, and a band-mean
comparison would read that as a change in skill. Rating therefore enters as a
continuous control, one model per cadence:

```
accuracy ~ era + rating + rating²      (errors clustered by player)
```

The era coefficient is the study's answer. `rating_equivalent_of_gap` divides it
by the fitted rating slope to express it back in rating points, which is the
form the original question was asked in. The band-by-band table is kept beside
it as a shape check — sixteen small tests, not sixteen findings.

## Setup

```bash
pip install -r requirements.txt
pip install -e .                  # makes the package importable from anywhere
```

Commands are then run as `python -m chess_elo_drift.cli ...`. The install also
declares a `chess-elo-drift` console script, but writing it needs permission on
the interpreter's `Scripts` directory, which a system-wide Python install does
not grant; inside a virtualenv it lands normally.

The engine is not vendored. Download a Stockfish build and put the binary at
`tools/stockfish/stockfish.exe` (or point `--engine` elsewhere). Any recent
version works; what matters is that a single one is used for the whole run,
since it is the measuring instrument.

## Running the study

Three commands, each resumable, each writing where the next one reads:

```bash
python -m chess_elo_drift.cli collect    # chess.com API  -> data/raw/games_<era>.jsonl
python -m chess_elo_drift.cli evaluate   # stored PGNs    -> data/processed/evaluations.csv
python -m chess_elo_drift.cli report     # evaluations    -> reports/
```

Without installing the package, prefix those with `PYTHONPATH=src`.

**collect** fills a quota per (cadence × 200-point band) for both eras. Re-running
tops the corpus up rather than doubling it: already-collected games prime the
quotas and already-visited accounts are skipped. A single account the API fails
on is logged and stepped over; a run of failures stops the crawl, on the
assumption that the API is down rather than the account being bad. Useful flags:
`--per-cell`, `--max-requests`, `--min-interval` (raise it if throttled).

**evaluate** scores games on every core, writing rows as they complete, and skips
games already present in the output — an interrupted run resumes for free.
`--limit N` spends a fixed engine budget on a subsample spread evenly across the
design, which keeps the thin rating bands populated instead of spending
everything on the crowded middle. Games differ enormously in cost, so a batch's
wall time is dominated by its longest games, not its average one.

**report** builds the estimation sample, runs both comparisons and writes:

```
reports/findings.md                # the answer, with its method and caveats
reports/tables/*.csv               # era gaps, band comparison, coverage
reports/figures/*.png              # accuracy and ACPL by band, coverage
```

## Layout

```
src/chess_elo_drift/
  config.py           every tunable of the protocol, in one immutable place
  records.py          GameRecord: one game, stored once, two sides derived
  selection.py        spending a fixed engine budget evenly across the design
  chesscom/           rate-limit-aware API client and typed endpoint wrappers
  collection/         the stratified snowball crawler and its resumable stores
  engine/             UCI analysis, accuracy/ACPL scoring, parallel runner
  analysis/           estimation sample, statistics, figures, findings note
tests/                unit tests; no network and no engine required
```

## Testing

```bash
python -m pytest
```

The suite is offline: the API is faked and no engine is started, so it runs in
seconds.

## Known limits

- **Snowball sampling is not a random sample of the population.** It is steered
  to cover the rating window evenly, which is what the question needs, but the
  accounts reached are those connected to the seeds by play. Results describe
  the sampled pool, not every account on the site.
- **Rating is the rating at the time of that game**, as reported in the archive.
  Provisional ratings on young accounts are noisier than settled ones.
- **Accuracy is one operationalisation of strength.** It is computed from a
  fixed-depth engine's evaluation swing per move; a deeper search would give
  different absolute numbers. Only the comparison between eras is claimed, and
  it is valid because the instrument is identical on both sides.
- **The opening is not scored** (`config.OPENING_PLIES_SKIPPED`). Book moves are
  memorised rather than calculated, and the availability of preparation has
  itself changed over the years, so scoring them would measure something else.
