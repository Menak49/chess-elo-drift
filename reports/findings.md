# Does a chess.com rating still mean what it meant in 2018?

At equal rating, comparing 2018-2019 with 2024-2025 on 1,507 player-observations from 1,200 games: **blitz**: players of the same rating play worse now (-1.34 accuracy points, p = 0.0068); **rapid**: players of the same rating play worse now (-1.39 accuracy points, p = 0.0023).

## What was measured

- Rated blitz and rapid games, ratings 600-2200, sampled from the same calendar months on both sides (2018-03, 2018-06, 2018-09, 2019-03, 2019-06, 2019-09 versus 2024-03, 2024-06, 2024-09, 2025-03, 2025-06, 2025-09).
- A player's rating is the one recorded on that game, not their rating today.
- Accuracy recomputed with Stockfish at fixed depth 12, skipping the first 8 plies of book moves.
- Sample sizes:

| era       |   observations |   players |   games |   median_rating |
|:----------|---------------:|----------:|--------:|----------------:|
| 2018-2019 |            904 |       738 |     672 |          1459.5 |
| 2024-2025 |            603 |       493 |     528 |          1449   |

## Result

Each row is one cadence and one metric. The estimate is the coefficient on the era in a regression that also controls for rating, so it compares players of the *same* rating. Standard errors are clustered by player.

| time_class   | metric   |   modern_minus_legacy |   std_error |     t |   p_value |   ci_95_low |   ci_95_high |   observations |   players |
|:-------------|:---------|----------------------:|------------:|------:|----------:|------------:|-------------:|---------------:|----------:|
| blitz        | accuracy |                -1.338 |       0.492 | -2.72 |    0.0068 |      -2.304 |       -0.371 |            675 |       589 |
| blitz        | acpl     |                 5.203 |       3.196 |  1.63 |    0.1    |      -1.075 |       11.481 |            675 |       589 |
| rapid        | accuracy |                -1.393 |       0.455 | -3.06 |    0.0023 |      -2.286 |       -0.5   |            832 |       668 |
| rapid        | acpl     |                 6.929 |       2.829 |  2.45 |    0.015  |       1.374 |       12.485 |            832 |       668 |

- **blitz**: -1.34 accuracy points (95% CI -2.30 to -0.37, 589 players). Accuracy rises by only 0.38 points per 100 rating here (SE 0.05), and dividing the gap by that slope puts it at very roughly **-354 rating points** -- an order of magnitude, not a figure: the divisor is small, so the ratio is far less certain than the accuracy gap above it.
- **rapid**: -1.39 accuracy points (95% CI -2.29 to -0.50, 668 players). Accuracy rises by only 0.61 points per 100 rating here (SE 0.05), and dividing the gap by that slope puts it at very roughly **-228 rating points** -- an order of magnitude, not a figure: the divisor is small, so the ratio is far less certain than the accuracy gap above it.

## Band by band

A shape check on the regression, not sixteen separate findings: with this many small tests, individual p-values should not be read in isolation.

| time_class   |   band | band_label   |   n_legacy |   n_modern |   mean_legacy |   mean_modern |   difference |   p_value |
|:-------------|-------:|:-------------|-----------:|-----------:|--------------:|--------------:|-------------:|----------:|
| blitz        |    600 | 600-999      |         94 |         67 |         84.28 |         81.45 |        -2.83 |    0.02   |
| blitz        |   1000 | 1000-1399    |         93 |         69 |         84.77 |         83.87 |        -0.9  |    0.42   |
| blitz        |   1400 | 1400-1799    |         87 |         73 |         86.56 |         85.59 |        -0.96 |    0.27   |
| blitz        |   1800 | 1800-2199    |        117 |         75 |         87.81 |         86.89 |        -0.92 |    0.31   |
| rapid        |    600 | 600-999      |        113 |         81 |         82.18 |         83.19 |         1.01 |    0.39   |
| rapid        |   1000 | 1000-1399    |        121 |         74 |         87.07 |         84.97 |        -2.1  |    0.028  |
| rapid        |   1400 | 1400-1799    |        141 |         81 |         88.36 |         86.73 |        -1.64 |    0.022  |
| rapid        |   1800 | 1800-2199    |        138 |         83 |         90.87 |         88.4  |        -2.47 |    0.0012 |

## Why accuracy was recomputed

chess.com's own accuracy figure exists only for games somebody ran a review on, and that coverage collapsed backwards in time:

| era       | time_class   |   observations |   with_reported_accuracy |   coverage |
|:----------|:-------------|---------------:|-------------------------:|-----------:|
| 2018-2019 | blitz        |            686 |                       22 |     0.0321 |
| 2018-2019 | rapid        |            682 |                       44 |     0.0645 |
| 2024-2025 | blitz        |            536 |                       62 |     0.1157 |
| 2024-2025 | rapid        |            560 |                      104 |     0.1857 |

Using it directly would compare the 2024-2025 games people chose to review against almost nothing at all from 2018-2019, and across a formula that changed in between. The study therefore recomputes accuracy for every game with one engine at one depth.

Where both numbers exist (n = 148), the recomputed accuracy tracks chess.com's closely (Pearson r = 0.76, Spearman rho = 0.78), so the instrument measures the same thing on a different scale (mean 86.5 here against 73.9 reported).

## What this cannot tell you

- **Survivorship.** 2018-2019 games can only be read from accounts that still exist. Players who quit and deleted their account are invisible, and they are unlikely to be a random slice of the old population.
- **Activity weighting.** The crawler walks the opponent graph, so a player who played a hundred games that month is more likely to enter the sample than one who played three. The sample describes the opposition you would actually have met at a given rating, which is the relevant population for this question, but it is not a uniform draw over accounts.
- **Accuracy is not strength.** It rewards quiet, forcing and simplified positions. If the *style* of play at a given rating shifted -- more theory, sharper openings, more time trouble -- accuracy moves without strength moving.
- **One engine, one depth.** The scale is internally consistent, which is what the comparison needs, but the absolute numbers are not chess.com accuracies and should not be quoted as such.
- **Rating is a moving target by design.** Chess.com has adjusted its rating system over the period; any drift found here is the combined result of pool composition and administrative changes, which this design cannot separate.

## Files

- `accuracy_by_band`: `reports/tables/accuracy_by_band.csv`
- `acpl_by_band`: `reports/tables/acpl_by_band.csv`
- `band_comparison`: `reports/tables/band_comparison.csv`
- `era_gaps`: `reports/tables/era_gaps.csv`
- `reported_accuracy_coverage`: `reports/tables/reported_accuracy_coverage.csv`
- `accuracy`: `reports/figures/accuracy_by_band.png`
- `acpl`: `reports/figures/acpl_by_band.png`
- `coverage`: `reports/figures/reported_accuracy_coverage.png`
