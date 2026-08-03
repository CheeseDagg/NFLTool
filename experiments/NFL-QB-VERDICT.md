# NFL QB batch

```
==============================================================================
NFL QB BATCH — the blind spot the team rating cannot see
==============================================================================
panel: 4350 non-tie games 2010-2025   train 2662   holdout 1688
baseline reproduces nfl_model.run_elo on 4350 games, max |dp| 0
rows with a starter CHANGE on either side: 18.6%   with any 8-game discontinuity: 65.5%

Team Elo can represent any per-team quantity as R_t = beta*x_t +
skill_t, so QB QUALITY is absorbable and QBEXP is close to a control.
What it cannot represent is that the man who EARNED the rating is not
playing. QBCHG/QBNEW/QBEARN/QBRES are all ways of asking that.

------------------------------------------------------------------------------
--- GATE 0: AVAILABILITY — the five QB angles
    production predicts 272 unplayed games; a feature that reads a column
    null on those rows scored its backtest gain on data it will never have.
    columns read by this builder: away_qb_id, away_rest, away_team, div_game, home_qb_id, home_rest, home_team, location, season, week
      away_qb_id         unplayed-null  100.0%   UNAVAILABLE
      away_rest          unplayed-null    0.0%   OK
      away_team          unplayed-null    0.0%   OK
      div_game           unplayed-null    0.0%   OK
      home_qb_id         unplayed-null  100.0%   UNAVAILABLE
      home_rest          unplayed-null    0.0%   OK
      home_team          unplayed-null    0.0%   OK
      location           unplayed-null    0.0%   OK
      season             unplayed-null    0.0%   OK
      week               unplayed-null    0.0%   OK
    UNAVAILABLE COLUMNS: away_qb_id, home_qb_id
      qbchg      reads away_qb_id/home_qb_id, null on 100%/100% of the 272 unplayed games
      qbnew      reads away_qb_id/home_qb_id, null on 100%/100% of the 272 unplayed games
      qbexp      reads away_qb_id/home_qb_id, null on 100%/100% of the 272 unplayed games
      qbearn     reads away_qb_id/home_qb_id, null on 100%/100% of the 272 unplayed games
      qbres      reads away_qb_id/home_qb_id, null on 100%/100% of the 272 unplayed games

--- GATE 0: AVAILABILITY — CONTROL: rest/HFA/divisional, already live
    production predicts 272 unplayed games; a feature that reads a column
    null on those rows scored its backtest gain on data it will never have.
    columns read by this builder: away_rest, away_team, div_game, home_rest, home_team, location, season, week
      away_rest          unplayed-null    0.0%   OK
      away_team          unplayed-null    0.0%   OK
      div_game           unplayed-null    0.0%   OK
      home_rest          unplayed-null    0.0%   OK
      home_team          unplayed-null    0.0%   OK
      location           unplayed-null    0.0%   OK
      season             unplayed-null    0.0%   OK
      week               unplayed-null    0.0%   OK
    all columns available on the live slate — gate passed

--- AUDIT: every raw column nfl_model reads, vs the 272 unplayed rows it predicts
    away_moneyline     backtest only    unplayed-null   75.4%
    away_rest          PREDICTION       unplayed-null    0.0%
    away_score         outcome/selector unplayed-null  100.0%
    away_team          PREDICTION       unplayed-null    0.0%
    div_game           PREDICTION       unplayed-null    0.0%
    game_type          load/filter      unplayed-null    0.0%
    gameday            PREDICTION       unplayed-null    0.0%
    home_moneyline     backtest only    unplayed-null   75.4%
    home_rest          PREDICTION       unplayed-null    0.0%
    home_score         outcome/selector unplayed-null  100.0%
    home_team          PREDICTION       unplayed-null    0.0%
    location           PREDICTION       unplayed-null    0.0%
    season             PREDICTION       unplayed-null    0.0%
    spread_line        PREDICTION       unplayed-null   75.4%  <-- PARTIAL: confirm the read is guarded
    week               PREDICTION       unplayed-null    0.0%
    prediction-path columns not fully available: spread_line

------------------------------------------------------------------------------
--- PASS 1: baseline = recalibrated production model
QBCHG   starter changed since last game        holdout dLL +0.00377  seasons 5/6  -> BLOCKED (unavailable at prediction time)
QBNEW   share of last 8 starts by someone else holdout dLL +0.00685  seasons 5/6  -> BLOCKED (unavailable at prediction time)
QBEXP   career starts, log1p [PARTLY ABSORBABLE] holdout dLL +0.00287  seasons 5/6  -> BLOCKED (unavailable at prediction time)
QBEARN  new QB x how much rating is above 1500 holdout dLL +0.00435  seasons 6/6  -> BLOCKED (unavailable at prediction time)
QBRES   per-QB Elo minus his team's Elo        holdout dLL +0.00119  seasons 4/6  -> BLOCKED (unavailable at prediction time)
baseline holdout LL/game -0.63407

--- CEILINGS. ORACLE = what a model that knew the true
    coefficient would buy (a hard bound). n_rob/n_seed = how
    often a PLANTED effect survived the ship rule.
QBCHG   starter changed since last game        oracle(b=0.60) +0.00860 [+0.00678..+0.01142]  fitted +0.00827  plant 3/3  measured +0.00377   DEAD: a planted effect of this size was recovered 3/3, so a real one would have shown
QBNEW   share of last 8 starts by someone else oracle(b=0.60) +0.01015 [+0.00976..+0.01066]  fitted +0.01025  plant 3/3  measured +0.00685   DEAD: a planted effect of this size was recovered 3/3, so a real one would have shown
QBEXP   career starts, log1p [PARTLY ABSORBABLE] oracle(b=0.60) +0.08120 [+0.07730..+0.08639]  fitted +0.08346  plant 3/3  measured +0.00287   DEAD: a planted effect of this size was recovered 3/3, so a real one would have shown
QBEARN  new QB x how much rating is above 1500   (plant b=0.60 gave oracle +0.00172, UNDER the measured +0.00435 — probe too weak, stepping up)
QBEARN  new QB x how much rating is above 1500 oracle(b=0.90) +0.00522 [+0.00310..+0.00790]  fitted +0.00531  plant 3/3  measured +0.00435   DEAD: a planted effect of this size was recovered 3/3, so a real one would have shown
QBRES   per-QB Elo minus his team's Elo          (plant b=0.60 absorbed by the baseline, oracle -0.04400 — stepping down)
QBRES   per-QB Elo minus his team's Elo          (plant b=0.40 absorbed by the baseline, oracle -0.02064 — stepping down)
QBRES   per-QB Elo minus his team's Elo          (plant b=0.25 absorbed by the baseline, oracle -0.00875 — stepping down)
QBRES   per-QB Elo minus his team's Elo        oracle(b=0.15) -0.00376 [-0.00391..-0.00363]  fitted +0.00026  plant 1/3  measured +0.00119   PROBE UNINFORMATIVE: the oracle came out -0.00376, i.e. the refit baseline absorbed the planted effect. Do NOT read a verdict off this line.

------------------------------------------------------------------------------
--- PASS 2: QBEXP is now IN the baseline. A change term that only
    worked because it correlated with 'the backup is inexperienced'
    dies here, and should.
QBCHG   starter changed since last game        holdout dLL +0.00240  seasons 5/6  -> BLOCKED (unavailable at prediction time)
QBNEW   share of last 8 starts by someone else holdout dLL +0.00447  seasons 5/6  -> BLOCKED (unavailable at prediction time)
QBEARN  new QB x how much rating is above 1500 holdout dLL +0.00268  seasons 6/6  -> BLOCKED (unavailable at prediction time)
QBRES   per-QB Elo minus his team's Elo        holdout dLL +0.00089  seasons 3/6  -> BLOCKED (unavailable at prediction time)
baseline holdout LL/game -0.63120

--- CEILINGS. ORACLE = what a model that knew the true
    coefficient would buy (a hard bound). n_rob/n_seed = how
    often a PLANTED effect survived the ship rule.
QBCHG   starter changed since last game        oracle(b=0.60) +0.00728 [+0.00514..+0.01135]  fitted +0.00754  plant 3/3  measured +0.00240   DEAD: a planted effect of this size was recovered 3/3, so a real one would have shown
QBNEW   share of last 8 starts by someone else oracle(b=0.60) +0.00746 [+0.00697..+0.00813]  fitted +0.00883  plant 3/3  measured +0.00447   DEAD: a planted effect of this size was recovered 3/3, so a real one would have shown
QBEARN  new QB x how much rating is above 1500   (plant b=0.60 gave oracle +0.00103, UNDER the measured +0.00268 — probe too weak, stepping up)
QBEARN  new QB x how much rating is above 1500 oracle(b=0.90) +0.00378 [+0.00228..+0.00662]  fitted +0.00416  plant 1/3  measured +0.00268   WEAK PROBE: a planted effect was recovered only 1/3, so a real one this size could hide
QBRES   per-QB Elo minus his team's Elo          (plant b=0.60 absorbed by the baseline, oracle -0.04283 — stepping down)
QBRES   per-QB Elo minus his team's Elo          (plant b=0.40 absorbed by the baseline, oracle -0.01919 — stepping down)
QBRES   per-QB Elo minus his team's Elo          (plant b=0.25 absorbed by the baseline, oracle -0.00818 — stepping down)
QBRES   per-QB Elo minus his team's Elo        oracle(b=0.15) -0.00389 [-0.00435..-0.00330]  fitted +0.00047  plant 0/3  measured +0.00089   PROBE UNINFORMATIVE: the oracle came out -0.00389, i.e. the refit baseline absorbed the planted effect. Do NOT read a verdict off this line.

------------------------------------------------------------------------------
--- GATE 2: SHUFFLED PLACEBO (within season, strict).
    alpha = the SHIP RULE's own false-positive rate (a property of the
    gate, near-identical across angles). p_eff = how often a shuffled
    column matched the gain PASS 2 measured. Read p_eff.
QBCHG   starter changed since last game        alpha  11/200 (0.0597)   p_eff   0/200 (p=0.0050) vs measured +0.00240
QBNEW   share of last 8 starts by someone else alpha  20/200 (0.1045)   p_eff   0/200 (p=0.0050) vs measured +0.00447
QBEARN  new QB x how much rating is above 1500 alpha  16/200 (0.0846)   p_eff   0/200 (p=0.0050) vs measured +0.00268
QBRES   per-QB Elo minus his team's Elo        alpha  13/200 (0.0697)   p_eff   1/200 (p=0.0100) vs measured +0.00089
------------------------------------------------------------------------------
--- GATE 4: SHAPE. Does the effect live where the claim says?
games with a discontinuity on exactly one side: 2070
QBCHG   starter changed since last game        one-sided-change subset dLL +0.00570  4/6  -> BLOCKED (unavailable at prediction time)
QBNEW   share of last 8 starts by someone else one-sided-change subset dLL +0.01134  6/6  -> BLOCKED (unavailable at prediction time)
QBEARN  new QB x how much rating is above 1500 one-sided-change subset dLL +0.00381  5/6  -> BLOCKED (unavailable at prediction time)
QBRES   per-QB Elo minus his team's Elo        one-sided-change subset dLL +0.00065  5/6  -> BLOCKED (unavailable at prediction time)
==============================================================================
Ship rule: GATE 0 clear, then a ROBUST WIN in PASS 2, a measured gain
under the ORACLE bound, a placebo that does not fire, and a shape that
holds. Gate 0 is a veto and not a vote: gates 1-4 measure how much an
angle buys, and no size of gain makes a column that does not exist on
the live slate shippable.
==============================================================================
```
