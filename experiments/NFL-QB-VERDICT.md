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
--- PASS 1: baseline = recalibrated production model
QBCHG   starter changed since last game        holdout dLL +0.00377  seasons 5/6  -> ROBUST WIN
QBNEW   share of last 8 starts by someone else holdout dLL +0.00685  seasons 5/6  -> ROBUST WIN
QBEXP   career starts, log1p [PARTLY ABSORBABLE] holdout dLL +0.00287  seasons 5/6  -> ROBUST WIN
QBEARN  new QB x how much rating is above 1500 holdout dLL +0.00435  seasons 6/6  -> ROBUST WIN
QBRES   per-QB Elo minus his team's Elo        holdout dLL +0.00119  seasons 4/6  -> win, not robust
baseline holdout LL/game -0.63407

--- CEILINGS. ORACLE = what a model that knew the true
    coefficient would buy (a hard bound). n_rob/n_seed = how
    often a PLANTED effect survived the ship rule.
QBCHG   starter changed since last game        oracle(b=0.60) +0.00860 [+0.00678..+0.01142]  fitted +0.00827  plant 3/3  measured +0.00377   GATE 1 PASSED: measured +0.00377 is a robust win and sits UNDER the oracle +0.00860, so it is not a power artefact — the ceiling is not the binding constraint on this angle
QBNEW   share of last 8 starts by someone else oracle(b=0.60) +0.01015 [+0.00976..+0.01066]  fitted +0.01025  plant 3/3  measured +0.00685   GATE 1 PASSED: measured +0.00685 is a robust win and sits UNDER the oracle +0.01015, so it is not a power artefact — the ceiling is not the binding constraint on this angle
QBEXP   career starts, log1p [PARTLY ABSORBABLE] oracle(b=0.60) +0.08120 [+0.07730..+0.08639]  fitted +0.08346  plant 3/3  measured +0.00287   GATE 1 PASSED: measured +0.00287 is a robust win and sits UNDER the oracle +0.08120, so it is not a power artefact — the ceiling is not the binding constraint on this angle
QBEARN  new QB x how much rating is above 1500   (plant b=0.60 gave oracle +0.00172, UNDER the measured +0.00435 — probe too weak, stepping up)
QBEARN  new QB x how much rating is above 1500 oracle(b=0.90) +0.00522 [+0.00310..+0.00790]  fitted +0.00531  plant 3/3  measured +0.00435   GATE 1 PASSED: measured +0.00435 is a robust win and sits UNDER the oracle +0.00522, so it is not a power artefact — the ceiling is not the binding constraint on this angle
QBRES   per-QB Elo minus his team's Elo          (plant b=0.60 absorbed by the baseline, oracle -0.04400 — stepping down)
QBRES   per-QB Elo minus his team's Elo          (plant b=0.40 absorbed by the baseline, oracle -0.02064 — stepping down)
QBRES   per-QB Elo minus his team's Elo          (plant b=0.25 absorbed by the baseline, oracle -0.00875 — stepping down)
QBRES   per-QB Elo minus his team's Elo        oracle(b=0.15) -0.00376 [-0.00391..-0.00363]  fitted +0.00026  plant 1/3  measured +0.00119   PROBE UNINFORMATIVE: the oracle came out -0.00376, i.e. the refit baseline absorbed the planted effect. Do NOT read a verdict off this line.

------------------------------------------------------------------------------
--- PASS 2: QBEXP is now IN the baseline. A change term that only
    worked because it correlated with 'the backup is inexperienced'
    dies here, and should.
QBCHG   starter changed since last game        holdout dLL +0.00240  seasons 5/6  -> ROBUST WIN
QBNEW   share of last 8 starts by someone else holdout dLL +0.00447  seasons 5/6  -> ROBUST WIN
QBEARN  new QB x how much rating is above 1500 holdout dLL +0.00268  seasons 6/6  -> ROBUST WIN
QBRES   per-QB Elo minus his team's Elo        holdout dLL +0.00089  seasons 3/6  -> win, not robust
baseline holdout LL/game -0.63120

--- CEILINGS. ORACLE = what a model that knew the true
    coefficient would buy (a hard bound). n_rob/n_seed = how
    often a PLANTED effect survived the ship rule.
QBCHG   starter changed since last game        oracle(b=0.60) +0.00728 [+0.00514..+0.01135]  fitted +0.00754  plant 3/3  measured +0.00240   GATE 1 PASSED: measured +0.00240 is a robust win and sits UNDER the oracle +0.00728, so it is not a power artefact — the ceiling is not the binding constraint on this angle
QBNEW   share of last 8 starts by someone else oracle(b=0.60) +0.00746 [+0.00697..+0.00813]  fitted +0.00883  plant 3/3  measured +0.00447   GATE 1 PASSED: measured +0.00447 is a robust win and sits UNDER the oracle +0.00746, so it is not a power artefact — the ceiling is not the binding constraint on this angle
QBEARN  new QB x how much rating is above 1500   (plant b=0.60 gave oracle +0.00103, UNDER the measured +0.00268 — probe too weak, stepping up)
QBEARN  new QB x how much rating is above 1500 oracle(b=0.90) +0.00378 [+0.00228..+0.00662]  fitted +0.00416  plant 1/3  measured +0.00268   GATE 1 PASSED: measured +0.00268 is a robust win and sits UNDER the oracle +0.00378, so it is not a power artefact — the ceiling is not the binding constraint on this angle
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
QBCHG   starter changed since last game        one-sided-change subset dLL +0.00570  4/6  -> win, not robust
QBNEW   share of last 8 starts by someone else one-sided-change subset dLL +0.01134  6/6  -> ROBUST WIN
QBEARN  new QB x how much rating is above 1500 one-sided-change subset dLL +0.00381  5/6  -> ROBUST WIN
QBRES   per-QB Elo minus his team's Elo        one-sided-change subset dLL +0.00065  5/6  -> ROBUST WIN
==============================================================================
Ship rule: ROBUST WIN in PASS 2, a measured gain under the ORACLE
bound, a placebo that does not fire, and a shape that holds.
==============================================================================
```

## Verdict

**SHIP `QBNEW`** — share of the last 8 starts taken by someone other than tonight's
starter, home minus away. All four gates:

| gate | result |
|---|---|
| 3 replication | +0.00685 holdout LL/game, 5/6 seasons ROBUST WIN; +0.00447 5/6 with QB experience in the baseline |
| 1 power | oracle +0.01015 at plant b=0.60, planted effect recovered 3/3, measured sits UNDER the bound |
| 2 placebo | 0/200 within-season shuffles reached the measured gain, p=0.0050 |
| 4 shape | 2.5x stronger on the 2,070 one-sided-change games: +0.01134, 6/6 seasons |

`QBCHG` clears gates 1-3 but its shape read is 4/6 and QBNEW beats it on both the real
and the synthetic panel — a benching or injury costs a team for a stretch, and a window
fraction reads a stretch while a since-last-game flag reads only its first game.
`QBEARN` has the cleanest per-season record (6/6 both passes) but is near-collinear with
QBNEW and adds nothing on top. `QBRES` is a level wearing a residual's clothes: the
baseline absorbs a plant at every strength down to b=0.15, so its probe is uninformative
and its 3/6 pass-2 record is not a verdict.

Why this one won when the EPA batch's QB blend died 1/6: team Elo can represent
R_t = beta*(QB quality)_t + skill_t, so QB QUALITY is absorbable. A CHANGE of quarterback
is not — a running average over a team's history cannot know the man who earned the
rating is not playing tonight.

### Reader bugs this batch exposed

The first NFL angle that actually won broke three gate-reading routines, all of which had
been written against batches where the measured effect was ~zero:

1. The ceiling reader's last rung ("a plant was recovered 3/3, so a real effect would
   have shown -> DEAD") is a statement about a NULL. It labelled QBNEW DEAD. It now takes
   the replication verdict as an argument.
2. The placebo counted how often a shuffled column earned a ROBUST WIN — the ship rule's
   own false-positive rate, a property of the GATE, which is why four angles with gains
   spanning 7x all reported p ~ 0.065-0.10. It now reports `alpha` (that FPR: ~0.06-0.10
   here, so a bare ROBUST WIN is worth about p=0.08) and `p_eff` (shuffles reaching the
   measured gain). Read p_eff.
3. The plant ladder only stepped DOWN, for absorbed probes. An oracle can also land BELOW
   the measured gain when a column is zero on most rows — a probe calibrated too weakly,
   not a noisy measurement. It mislabelled QBEARN "noise by construction". The ladder now
   escalates to 0.90/1.40/2.00 until the bound actually bounds.

All three were silent, all three read as "dead angle", and every fix makes the file MORE
likely to report a win — the direction a reader bug is least likely to be caught from.
