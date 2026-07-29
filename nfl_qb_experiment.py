#!/usr/bin/env python3
"""
NFL QB BATCH — the model's own #1 listed blind spot, tested
===========================================================

WHY THIS EXISTS. NFLTool's knowledge file lists "who the quarterback actually
is (injuries/benchings — the team rating cannot know)" as the single biggest
blind spot in the NFL model. The data to test it has been sitting in games.csv
the whole time: nflverse ships `home_qb_id` / `away_qb_id` / names on every
row, 100% populated on every played game from 1999 on.

THE EPA BATCH ALREADY TESTED SOMETHING ELSE. nfl_epa_experiment.py blended a
per-QB EPA rating (`w_qb`) and it died — 1/6 holdout seasons. That is the
expected result and it is not this test. A QB-quality rating asks "who is the
quarterback", and a team Elo already knows the answer: the team's rating was
built out of games that man played. This is the NFL instance of the absorption
theorem from the UFC work — Elo scores a game on `R_h - R_a`, so if the truth
is `z = beta * (x_h - x_a)` for any per-team quantity x, then `R_t = beta*x_t +
skill_t` reproduces it exactly and the ratings converge there unaided. A
per-team QB quality term is exactly that shape, so it is absorbable, and a null
on it says "Elo already knows" rather than "the quarterback does not matter."

WHAT IS **NOT** ABSORBABLE IS A CHANGE. The rating is a running average over a
history. It cannot know that the man who earned it is not playing tonight. That
residual is what this file tests, five ways:

  QBCHG   tonight's starter differs from the one who started the LAST game.
          The crispest possible statement of the blind spot. Binary, so its
          power is limited, but there is nothing to misread.
  QBNEW   fraction of the team's last 8 starts made by someone OTHER than
          tonight's man. Continuous, and it separates "week-2 change that has
          since settled" from "he is genuinely new here."
  QBEXP   career starts by tonight's QB, log1p. A rookie is not a veteran.
          Partly absorbable — a team that starts rookies has a low rating —
          so this is closer to a control than to a candidate.
  QBEARN  THE ABSORPTION-AWARE ONE. A missing starter should cost a GOOD team
          and cost a bad team nothing, because a bad team's rating was not
          earned by him. QBNEW * max(R - 1500, 0)/100. This is the NFL
          analogue of the UFC's RCHNEW/SPNEW construction: key the term on
          where the rating is uninformative rather than on the raw trait.
  QBRES   per-QB Elo carried WITH the quarterback across teams, entered as a
          RESIDUAL against his current team's rating: (qb_elo - team_elo)
          differenced. The absorbable part is subtracted off by construction,
          so what is left is the part where a QB's own record disagrees with
          the team he is now on.

BASELINE. The production model, exactly — constants are IMPORTED from
nfl_model so they cannot drift, and the walk-forward here is asserted to
reproduce nfl_model.run_elo's p_home to 1e-12 on every game. A baseline that
silently diverges from production would make every number in this file a
verdict about a model nobody runs.

The baseline enters as logit(p_base) with its own fitted slope and intercept,
which RECALIBRATES the production model before any angle is asked to beat it.
Without that, an angle can win purely by fixing calibration drift and the win
has nothing to do with quarterbacks.

NO GRIDS. Coefficients here are joint MLE, not grid searches, so the
grid-edge / censored-fit problem that bit the UFC batches cannot arise: there
is no wall for a coefficient to pin against.

THE ONE HONEST CAVEAT. `home_qb_id` is the QB who actually started the game.
At Sunday-morning prediction time you know the ANNOUNCED starter, which is
usually but not always the same man. So these features are very slightly
better-informed than a live model would be, in the direction of making an
effect easier to find. That biases toward a FALSE POSITIVE, never toward a
false negative — so a null here is safe to believe, and a win would need the
announced-starter version re-run before it ships.

FOUR GATES, same rule as the UFC batches:
  1 POWER CEILING     plant an effect of known size, see whether the pipeline
                      recovers it. A non-positive oracle means the baseline
                      absorbed the plant: that is a BROKEN PROBE, not a dead
                      angle, and the plant strength walks down until one lands.
  2 SHUFFLED PLACEBO  permute the angle within season and count how often the
                      ship rule fires on noise. p = (hits + 1) / (N + 1).
  3 REPLICATION       per-season holdout deltas, 2020-2025.
  4 SHAPE             does the effect live where the claim says it lives?

RUN:
    python nfl_qb_experiment.py --selftest    # offline, synthetic, must pass
    python nfl_qb_experiment.py               # real verdict
"""

import os
import sys
import math

import numpy as np
import pandas as pd

import nfl_model as NM

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICT_OUT = os.environ.get("VERDICT_OUT",
                             os.path.join(HERE, "experiments",
                                          "NFL-QB-VERDICT.md"))

TRAIN_LO, TRAIN_HI = 2010, 2019      # tune here
HOLD_LO = 2020                       # decide here
QB_WINDOW = 8                        # starts of history for QBNEW
QB_K = 12.0                          # per-QB Elo K; smaller than team K (20)
                                     # because a QB plays ~17 games a year and
                                     # a per-QB rating that moves as fast as a
                                     # team's would be mostly noise.

# ---------------------------------------------------------------------------
# angles
# ---------------------------------------------------------------------------
# Every column is signed so that a POSITIVE value should favour the HOME team,
# which means every angle's expected coefficient is POSITIVE. Mixing sign
# conventions across five columns is how a real effect ends up read as a
# wrong-signed null.
ANGLES = [
    ("QBCHG   starter changed since last game", "qbchg"),
    ("QBNEW   share of last 8 starts by someone else", "qbnew"),
    ("QBEXP   career starts, log1p [PARTLY ABSORBABLE]", "qbexp"),
    ("QBEARN  new QB x how much rating is above 1500", "qbearn"),
    ("QBRES   per-QB Elo minus his team's Elo", "qbres"),
]
CANDIDATES = {"qbchg", "qbnew", "qbearn", "qbres"}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_logistic(X, y, l2=1e-4, iters=4000, lr=0.5):
    """Standardize on train, gradient descent, tiny L2. Returns (predict, cf).

    Standardization matters here for a specific reason: logit(p_base) has a
    spread of about +-1.5 while QBEARN can reach 5, and un-standardized
    gradient descent on columns of different scale converges at different
    rates, which shows up as an angle looking weak when it is only slow.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(iters):
        p = _sigmoid(Z @ w + b)
        gw = Z.T @ (p - y) / n + l2 * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb

    def predict(Xn):
        Zn = (np.asarray(Xn, dtype=float) - mu) / sd
        return _sigmoid(Zn @ w + b)

    return predict, dict(mu=mu, sd=sd, w=w, b=b)


def mean_ll(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# the panel: production Elo + QB state, one leak-free pass
# ---------------------------------------------------------------------------

def build_panel(g, start_season=TRAIN_LO):
    """One walk-forward. Team Elo mirrors nfl_model EXACTLY; QB state is read
    before the game and updated after it.

    Ties are dropped from the panel (as nfl_model.backtest does) because a
    binary-outcome log-likelihood has nowhere to put a 0.5.
    """
    R = {}
    H = NM.HFA_INIT
    cur = None
    # QB state, all of it strictly historical at read time
    last_start = {}                  # team -> qb_id who started its last game
    recent = {}                      # team -> list of last QB_WINDOW qb_ids
    starts = {}                      # qb_id -> games started so far
    qelo = {}                        # qb_id -> per-QB Elo
    rows = []

    for r in g.itertuples():
        if r.season != cur:
            cur = r.season
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - NM.REVERT)
            # QB ratings regress too, and toward the same mean, because the
            # alternative is a retired quarterback's rating sitting frozen at
            # 1700 forever and re-entering the panel if he ever backs up a
            # game. Teams and QBs use the SAME revert so a residual of the two
            # is not contaminated by one of them drifting faster.
            for q in qelo:
                qelo[q] = 1500 + (qelo[q] - 1500) * (1 - NM.REVERT)

        h, a = r.home_team, r.away_team
        R.setdefault(h, 1500)
        R.setdefault(a, 1500)
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = NM.REST_PER_DAY * ((r.home_rest - 7) - (r.away_rest - 7))
        neutral = str(r.location) == "Neutral"
        hfa = 0.0 if neutral else H
        dr = (R[h] + hfa + rest) - R[a]
        p_home = NM.expected(dr)
        div = bool(getattr(r, "div_game", 0) == 1)
        p_pred = NM.expected(dr * NM.DIV_TAU) if div else p_home

        qh, qa = r.home_qb_id, r.away_qb_id
        played = (pd.notna(r.home_score) and pd.notna(r.away_score))

        if played and r.season >= start_season and pd.notna(qh) and pd.notna(qa):
            margin = r.home_score - r.away_score

            def _new_frac(team, qb):
                hist = recent.get(team, [])
                if not hist:
                    # NO HISTORY IS NOT A CHANGE. A team's first game in the
                    # panel has nobody to be different from; scoring it as
                    # "brand new QB" would put a 1.0 on every team in week 1
                    # of the first season and make the column a season-opener
                    # indicator instead of a QB-change indicator.
                    return 0.0
                return sum(1 for x in hist if x != qb) / len(hist)

            nh, na = _new_frac(h, qh), _new_frac(a, qa)
            ch = 0.0 if last_start.get(h) in (None, qh) else 1.0
            ca = 0.0 if last_start.get(a) in (None, qa) else 1.0
            above_h = max(R[h] - 1500.0, 0.0) / 100.0
            above_a = max(R[a] - 1500.0, 0.0) / 100.0
            res_h = (qelo.get(qh, 1500.0) - R[h]) / 100.0
            res_a = (qelo.get(qa, 1500.0) - R[a]) / 100.0

            rows.append({
                "season": int(r.season), "week": int(r.week),
                "home": h, "away": a, "qh": qh, "qa": qa,
                "p_base": p_pred,
                "home_win": int(margin > 0), "tie": int(margin == 0),
                # signed so POSITIVE favours home in every column
                "qbchg": ca - ch,
                "qbnew": na - nh,
                "qbexp": (math.log1p(starts.get(qh, 0))
                          - math.log1p(starts.get(qa, 0))),
                "qbearn": na * above_a - nh * above_h,
                "qbres": res_h - res_a,
                # bookkeeping for the shape gate
                "home_new": nh, "away_new": na,
                "home_above": above_h, "away_above": above_a,
            })

        if played:
            margin = r.home_score - r.away_score
            mov = math.log(abs(margin) + 1) * (
                2.2 / ((0.001 * abs(dr) if margin * dr > 0
                        else -0.001 * abs(dr)) + 2.2))
            s_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            delta = NM.K * mov * (s_home - p_home)
            R[h] += delta
            R[a] -= delta
            if not neutral:
                H += NM.HFA_LR * (s_home - p_home)
            if pd.notna(qh) and pd.notna(qa):
                qelo.setdefault(qh, 1500.0)
                qelo.setdefault(qa, 1500.0)
                eq = NM.expected(qelo[qh] - qelo[qa])
                dq = QB_K * (s_home - eq)
                qelo[qh] += dq
                qelo[qa] -= dq
                starts[qh] = starts.get(qh, 0) + 1
                starts[qa] = starts.get(qa, 0) + 1
                last_start[h] = qh
                last_start[a] = qa
                recent[h] = (recent.get(h, []) + [qh])[-QB_WINDOW:]
                recent[a] = (recent.get(a, []) + [qa])[-QB_WINDOW:]

    P = pd.DataFrame(rows)
    return P[P["tie"] == 0].reset_index(drop=True), R, H


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _split(P):
    tr = (P["season"] >= TRAIN_LO) & (P["season"] <= TRAIN_HI)
    ho = P["season"] >= HOLD_LO
    return tr.values, ho.values


def score_angle(P, col, ys=None, extra=()):
    """Holdout mean-LL gain from adding `col` to the recalibrated baseline.

    Returns (d_ll, per_season dict, base_ll, treat_ll).
    """
    y = P["home_win"].values.astype(float) if ys is None else np.asarray(ys)
    base_cols = [_logit(P["p_base"].values)] + [P[c].values for c in extra]
    Xb = np.column_stack(base_cols)
    Xt = np.column_stack(base_cols + [P[col].values])
    tr, ho = _split(P)
    pb, _ = fit_logistic(Xb[tr], y[tr])
    pt, _ = fit_logistic(Xt[tr], y[tr])
    ph_b, ph_t = pb(Xb[ho]), pt(Xt[ho])
    yh = y[ho]
    per = {}
    seas = P["season"].values[ho]
    for s in np.unique(seas):
        m = seas == s
        per[int(s)] = mean_ll(ph_t[m], yh[m]) - mean_ll(ph_b[m], yh[m])
    return (mean_ll(ph_t, yh) - mean_ll(ph_b, yh), per,
            mean_ll(ph_b, yh), mean_ll(ph_t, yh))


def verdict_of(d_ll, per):
    good = sum(1 for v in per.values() if v > 0)
    n = len(per)
    if d_ll <= 0:
        return "NULL", good, n
    if good >= max(1, n - 1):
        return "ROBUST WIN", good, n
    return "win, not robust", good, n


# ---------------------------------------------------------------------------
# GATE 1 — power ceiling
# ---------------------------------------------------------------------------

PLANT_LADDER = (0.60, 0.40, 0.25, 0.15)
# Stepped UP only when the oracle lands BELOW the measured gain, i.e. when the
# probe was too weak to bound the thing it is supposed to bound.
PLANT_UP = (0.90, 1.40, 2.00)


def read_ceiling(oracle, fitted, n_rob, n_seed, measured, plant_b,
                 verdict="NULL"):
    """Read gate 1. `verdict` is gate 3's answer and it CHANGES the meaning.

    Every rung below except this first one is a statement about a NULL: given
    that the pipeline could see a planted effect, a measured nothing is a real
    nothing. That logic is invalid when the measured effect is itself a robust
    win — there the power question has been answered by the measurement itself,
    and calling it DEAD (as an earlier version of this reader did, because it
    was written for a batch in which every measured value was ~zero) inverts
    the finding. Note also that oracle magnitude scales with the COLUMN's
    variance, so measured/oracle is not comparable across angles; the ceiling
    answers detectability, not effect size.
    """
    if oracle <= 0.0:
        return ("PROBE UNINFORMATIVE: the oracle came out %+.5f, i.e. the "
                "refit baseline absorbed the planted effect. Do NOT read a "
                "verdict off this line." % oracle)
    if verdict == "ROBUST WIN" and measured < oracle and n_rob >= 1:
        return ("GATE 1 PASSED: measured %+.5f is a robust win and sits UNDER "
                "the oracle %+.5f, so it is not a power artefact — the ceiling "
                "is not the binding constraint on this angle"
                % (measured, oracle))
    if measured >= oracle:
        return "measured >= ORACLE: noise by construction"
    if n_rob == 0:
        return ("STILL CANNOT BE SEEN - do not bury (a planted effect of this "
                "size was never recovered in %d seeds)" % n_seed)
    if n_rob < n_seed:
        return ("WEAK PROBE: a planted effect was recovered only %d/%d, so a "
                "real one this size could hide" % (n_rob, n_seed))
    return ("DEAD: a planted effect of this size was recovered %d/%d, so a "
            "real one would have shown" % (n_rob, n_seed))


def probe(P, col, seeds=(7, 17, 29), extra=(), d_real=None):
    """Plant a KNOWN effect into the outcomes and see what comes back.

    The plant regenerates y from the baseline's own fitted probability nudged
    by b_true * col, so the planted effect is true BY CONSTRUCTION and the
    oracle is the gain a model that knew b_true would collect. That is a hard
    upper bound on any honest fit.

    The ladder moves in BOTH directions and for two different reasons:

      DOWN, when the oracle comes back non-positive. The refit baseline
      absorbed the plant, so the probe is broken at that strength and a
      smaller plant may land outside what the baseline can swallow.

      UP, when the oracle is positive but SMALLER than the gain actually
      measured (`d_real`). The oracle is meant to be an upper bound; if the
      measurement exceeds it, the honest conclusion is that the probe was
      calibrated too weakly for this column, not that the measurement is
      noise. Oracle magnitude scales with the column's variance, and a column
      that is zero on most rows (QBEARN: only live when a QB changed AND the
      rating is above 1500) has a small oracle at any given coefficient. An
      earlier version of this file only stepped down, and consequently
      labelled QBEARN 'noise by construction' — a wrong label produced by a
      probe that was never asked a hard enough question.
    """
    y = P["home_win"].values.astype(float)
    base = np.column_stack([_logit(P["p_base"].values)]
                           + [P[c].values for c in extra])
    tr, ho = _split(P)
    p0, _ = fit_logistic(base[tr], y[tr])
    # Score EVERY row with the train-fit baseline, holdout included. Refitting
    # per split would leak the holdout into the plant's own definition.
    lp = _logit(p0(base))
    x = P[col].values

    def _rung(plant_b):
        oracles, fits, robs = [], [], 0
        for sd in seeds:
            rng = np.random.default_rng(sd)
            p_true = _sigmoid(lp + plant_b * x)
            ys = (rng.random(len(x)) < p_true).astype(float)
            # ORACLE: refit the baseline on the synthetic outcomes, then add
            # the TRUE coefficient. If the refit baseline can absorb the plant
            # through inflated main effects, this comes out NEGATIVE — that is
            # the broken probe, and it is why the ladder exists.
            pb2, _ = fit_logistic(base[tr], ys[tr])
            lb = _logit(pb2(base[ho]))
            o = mean_ll(_sigmoid(lb + plant_b * x[ho]), ys[ho]) - \
                mean_ll(pb2(base[ho]), ys[ho])
            oracles.append(o)
            d, per, _, _ = score_angle(P, col, ys=ys, extra=extra)
            fits.append(d)
            v, _, _ = verdict_of(d, per)
            robs += int(v == "ROBUST WIN")
        return dict(plant_b=plant_b, oracle=float(np.mean(oracles)),
                    lo=float(np.min(oracles)), hi=float(np.max(oracles)),
                    fitted=float(np.mean(fits)), n_rob=robs,
                    n_seed=len(seeds))

    out, hit = [], None
    for plant_b in PLANT_LADDER:
        r = _rung(plant_b)
        if r["oracle"] > 0.0 or plant_b == PLANT_LADDER[-1]:
            # The weakest rung is reported with its REAL numbers even when the
            # oracle is negative, so the reader can print the actual value it
            # is refusing to interpret. Zeroing it would hide how badly the
            # baseline absorbed the plant.
            hit = r
            break
        out.append((plant_b, r["oracle"]))
    if hit["oracle"] <= 0.0:
        hit["stepped"] = out
        hit["raised"] = []
        return hit
    raised = []
    if d_real is not None:
        for plant_b in PLANT_UP:
            if hit["oracle"] >= d_real:
                break
            raised.append((hit["plant_b"], hit["oracle"]))
            r = _rung(plant_b)
            if r["oracle"] <= hit["oracle"]:
                # A bigger plant that buys LESS means the baseline is eating
                # the extra. Stop; escalating further is not informative.
                break
            hit = r
    hit["stepped"] = out
    hit["raised"] = raised
    return hit


# ---------------------------------------------------------------------------
# GATE 2 — shuffled placebo
# ---------------------------------------------------------------------------

def placebo(P, col, n=200, seed=5, within_season=True, extra=(), d_real=None):
    """Permute the column and measure TWO different things.

    These get conflated and they are not the same number:

      alpha  how often a MEANINGLESS column still earns a ROBUST WIN. This is
             the ship rule's own false-positive rate. It is a property of the
             GATE, not of the angle, so it comes back at roughly the same value
             for every column — which is exactly how the first version of this
             function was caught: four angles with measured gains spanning 7x
             all reported p in 0.065-0.10, because the criterion never looked
             at the measured gain at all.

      p_eff  how often a meaningless column reaches a gain >= the one actually
             measured. THIS is the angle's significance, and it is the number
             that belongs next to a verdict.

    WITHIN-SEASON is the strict version and the default: it preserves the
    season-to-season structure the replication gate reads, so a placebo that
    still replicates 5/6 is telling you the gate is loose. A GLOBAL shuffle
    destroys that structure and is therefore lenient — it is available for
    comparison, not for the decision.
    """
    rng = np.random.default_rng(seed)
    Q = P.copy()
    x = P[col].values.copy()
    seas = P["season"].values
    hits = 0
    ge = 0
    for _ in range(n):
        if within_season:
            xs = x.copy()
            for s in np.unique(seas):
                m = seas == s
                xs[m] = rng.permutation(x[m])
        else:
            xs = rng.permutation(x)
        Q[col] = xs
        d, per, _, _ = score_angle(Q, col, extra=extra)
        v, _, _ = verdict_of(d, per)
        hits += int(v == "ROBUST WIN")
        if d_real is not None and d >= d_real:
            ge += 1
    alpha = (hits + 1) / (n + 1)
    p_eff = None if d_real is None else (ge + 1) / (n + 1)
    return hits, alpha, ge, p_eff


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _assert_baseline_matches(g, P):
    """The panel's p_base MUST equal what production would have said.

    This is the guard that matters most in the whole file. If the walk-forward
    here drifts from nfl_model.run_elo by so much as a rest-day convention,
    every verdict below is a statement about a model nobody runs, and nothing
    about the output would look wrong.
    """
    _, PP, _ = NM.run_elo(g, start_season=TRAIN_LO)
    PP = PP[PP["tie"] == 0]
    key = ["season", "week", "home", "away"]
    m = P.merge(PP[key + ["p_home"]], on=key, how="inner",
                suffixes=("", "_prod"))
    assert len(m) >= 0.98 * len(P), (
        "only %d of %d panel games matched production's prediction set — the "
        "two walk-forwards are not scoring the same games" % (len(m), len(P)))
    worst = float(np.max(np.abs(m["p_base"].values - m["p_home"].values)))
    assert worst < 1e-12, (
        "panel p_base diverges from nfl_model.run_elo by %.3g — the baseline "
        "in this experiment is NOT the production model" % worst)
    return len(m), worst


def experiment(out=print):
    g = NM.load()
    P, R, H = build_panel(g)
    n_match, worst = _assert_baseline_matches(g, P)
    tr, ho = _split(P)

    out("=" * 78)
    out("NFL QB BATCH — the blind spot the team rating cannot see")
    out("=" * 78)
    out("panel: %d non-tie games %d-%d   train %d   holdout %d"
        % (len(P), int(P["season"].min()), int(P["season"].max()),
           int(tr.sum()), int(ho.sum())))
    out("baseline reproduces nfl_model.run_elo on %d games, max |dp| %.2g"
        % (n_match, worst))
    chg = float((P["qbchg"].values != 0).mean())
    new = float((P["qbnew"].values != 0).mean())
    out("rows with a starter CHANGE on either side: %.1f%%   with any "
        "8-game discontinuity: %.1f%%" % (100 * chg, 100 * new))
    out("")
    out("Team Elo can represent any per-team quantity as R_t = beta*x_t +")
    out("skill_t, so QB QUALITY is absorbable and QBEXP is close to a control.")
    out("What it cannot represent is that the man who EARNED the rating is not")
    out("playing. QBCHG/QBNEW/QBEARN/QBRES are all ways of asking that.")
    out("")

    def _pass(label, extra=()):
        out("-" * 78)
        out(label)
        base_ll = None
        results = {}
        for name, col in ANGLES:
            if extra and col in extra:
                continue
            d, per, bll, tll = score_angle(P, col, extra=extra)
            base_ll = bll
            v, good, nn = verdict_of(d, per)
            results[col] = (name, d, per, v)
            out("%-46s holdout dLL %+.5f  seasons %d/%d  -> %s"
                % (name, d, good, nn, v))
        out("baseline holdout LL/game %+.5f" % base_ll)
        out("")
        out("--- CEILINGS. ORACLE = what a model that knew the true")
        out("    coefficient would buy (a hard bound). n_rob/n_seed = how")
        out("    often a PLANTED effect survived the ship rule.")
        for name, col in ANGLES:
            if extra and col in extra:
                continue
            meas = results[col][1]
            pr = probe(P, col, extra=extra, d_real=meas)
            for b_, o_ in pr["stepped"]:
                out("%-46s   (plant b=%.2f absorbed by the baseline, oracle "
                    "%+.5f — stepping down)" % (name, b_, o_))
            for b_, o_ in pr["raised"]:
                out("%-46s   (plant b=%.2f gave oracle %+.5f, UNDER the "
                    "measured %+.5f — probe too weak, stepping up)"
                    % (name, b_, o_, meas))
            out("%-46s oracle(b=%.2f) %+.5f [%+.5f..%+.5f]  fitted %+.5f  "
                "plant %d/%d  measured %+.5f   %s"
                % (name, pr["plant_b"], pr["oracle"], pr["lo"], pr["hi"],
                   pr["fitted"], pr["n_rob"], pr["n_seed"], meas,
                   read_ceiling(pr["oracle"], pr["fitted"], pr["n_rob"],
                                pr["n_seed"], meas, pr["plant_b"],
                                verdict=results[col][3])))
            results[col] = results[col] + (pr,)
        out("")
        return results

    r1 = _pass("--- PASS 1: baseline = recalibrated production model")

    # SECOND PASS. QBEXP is the partly-absorbable control, and any term built
    # on the QB-change idea will impersonate raw QB experience if experience
    # is not already spoken for. Same reasoning that killed the UFC's RCHNEW.
    r2 = _pass("--- PASS 2: QBEXP is now IN the baseline. A change term that "
               "only\n    worked because it correlated with 'the backup is "
               "inexperienced'\n    dies here, and should.",
               extra=("qbexp",))

    out("-" * 78)
    out("--- GATE 2: SHUFFLED PLACEBO (within season, strict).")
    out("    alpha = the SHIP RULE's own false-positive rate (a property of the")
    out("    gate, near-identical across angles). p_eff = how often a shuffled")
    out("    column matched the gain PASS 2 measured. Read p_eff.")
    for name, col in ANGLES:
        if col not in CANDIDATES:
            continue
        d_real = r2[col][1]
        hits, alpha, ge, p_eff = placebo(P, col, n=200, extra=("qbexp",),
                                         d_real=d_real)
        out("%-46s alpha %3d/200 (%.4f)   p_eff %3d/200 (p=%.4f) vs measured "
            "%+.5f" % (name, hits, alpha, ge, p_eff, d_real))

    out("-" * 78)
    out("--- GATE 4: SHAPE. Does the effect live where the claim says?")
    # Restrict to games where a change ACTUALLY happened on exactly one side.
    m1 = ((P["home_new"].values > 0) ^ (P["away_new"].values > 0))
    out("games with a discontinuity on exactly one side: %d" % int(m1.sum()))
    for name, col in ANGLES:
        if col not in CANDIDATES:
            continue
        Ps = P[m1].reset_index(drop=True)
        if (Ps["season"] >= HOLD_LO).sum() < 100:
            out("%-46s subset too small for a shape read" % name)
            continue
        d, per, _, _ = score_angle(Ps, col)
        v, good, nn = verdict_of(d, per)
        out("%-46s one-sided-change subset dLL %+.5f  %d/%d  -> %s"
            % (name, d, good, nn, v))

    out("=" * 78)
    out("Ship rule: ROBUST WIN in PASS 2, a measured gain under the ORACLE")
    out("bound, a placebo that does not fire, and a shape that holds.")
    out("=" * 78)
    return P, r1, r2


# ---------------------------------------------------------------------------
# SELFTEST
# ---------------------------------------------------------------------------

def _synth(seed=3, n_seasons=14, n_teams=16, qb_effect=0.0):
    """Synthetic league with QB churn.

    Teams have a fixed skill. Each team has a starter and a clearly WORSE
    backup, and the starter misses random stretches. If qb_effect > 0 the
    outcome genuinely depends on who is playing, which is the thing a team
    rating cannot know — so a correct pipeline must find it.

    Seasons run 2010..2010+n_seasons-1 and n_seasons MUST carry the panel past
    HOLD_LO, or _split hands score_angle an empty holdout and every delta comes
    back nan. A nan is not a null: it fails the ROBUST WIN test for the wrong
    reason and would read as "the pipeline cannot see a planted effect" when
    the truth is "the pipeline was never given anything to score".
    """
    assert 2010 + n_seasons - 1 >= HOLD_LO + 1, (
        "synthetic league ends in %d but the holdout starts in %d — there "
        "would be nothing to score" % (2010 + n_seasons - 1, HOLD_LO))
    rng = np.random.default_rng(seed)
    teams = ["T%02d" % i for i in range(n_teams)]
    skill = {t: rng.normal(0, 0.45) for t in teams}
    starter = {t: "Q%s_A" % t for t in teams}
    backup = {t: "Q%s_B" % t for t in teams}
    out_until = {t: -1 for t in teams}
    rows = []
    gi = 0
    for s in range(2010, 2010 + n_seasons):
        for wk in range(1, 18):
            order = rng.permutation(teams)
            for i in range(0, n_teams - 1, 2):
                h, a = order[i], order[i + 1]
                gi += 1
                qbs = {}
                for t in (h, a):
                    if gi > out_until[t] and rng.random() < 0.04:
                        out_until[t] = gi + rng.integers(2, 7)
                    qbs[t] = (backup[t] if gi <= out_until[t]
                              else starter[t])
                pen = {t: (-1.0 if qbs[t] == backup[t] else 0.0)
                       for t in (h, a)}
                z = (skill[h] - skill[a]) + 0.20 \
                    + qb_effect * (pen[h] - pen[a])
                y = int(rng.random() < 1.0 / (1.0 + math.exp(-z)))
                marg = (3 if y else -3)
                rows.append(dict(
                    game_id="%d_%02d_%s_%s" % (s, wk, a, h),
                    season=s, game_type="REG", week=wk,
                    gameday="%d-09-%02d" % (s, 1 + (wk % 28)),
                    away_team=a, away_score=(0 if y else 3),
                    home_team=h, home_score=(3 if y else 0),
                    location="Home", result=marg, div_game=0,
                    home_rest=7, away_rest=7,
                    home_moneyline=-110, away_moneyline=-110,
                    spread_line=0.0,
                    home_qb_id=qbs[h], away_qb_id=qbs[a],
                    home_qb_name=qbs[h], away_qb_name=qbs[a]))
    g = pd.DataFrame(rows)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g.sort_values(["season", "week"]).reset_index(drop=True)


def selftest():
    buf = []

    def out(s=""):
        buf.append(str(s))

    # ---- signs and definitions, on hand-built state -----------------------
    g = _synth(qb_effect=0.0)
    P, _, _ = build_panel(g)

    # PRECONDITION, because the failure it catches is SILENT. If the QB id
    # plumbing breaks, every angle column reads 0.0 and the file reports NULL
    # on a panel built to contain an effect. Counting live rows is the only
    # thing that notices. This is exactly the bug that bit the UFC stance
    # batch, where a stance-encoding mismatch produced six columns of zeros
    # and a clean-looking NULL.
    for col in ("qbchg", "qbnew", "qbexp", "qbearn", "qbres"):
        nz = int((P[col].values != 0).sum())
        assert nz > 50, (
            "only %d of %d rows carry a non-zero %s — the synthetic panel is "
            "not exercising the column it is meant to test" % (nz, len(P), col))

    # THE HOLDOUT MUST EXIST. Every verdict in this file is a holdout number,
    # and an empty holdout does not raise — it returns nan, which then fails the
    # planted-effect assertion below with a message blaming the pipeline. Check
    # the panel before trusting anything scored off it.
    n_ho = int((P["season"].values >= HOLD_LO).sum())
    assert n_ho > 200, (
        "synthetic panel has only %d holdout rows (>= %d); deltas would be nan "
        "and every verdict meaningless" % (n_ho, HOLD_LO))

    # A team's FIRST game has no history, so it cannot be a change. Enforced in
    # build_panel's _new_frac branch; asserted here on the column itself.
    assert float(np.max(np.abs(P["qbnew"].values[:8]))) == 0.0, (
        "week 1 of the first season carries a non-zero qbnew — no-history is "
        "being scored as a QB change, making this a season-opener indicator")

    # QBNEW must be zero when a team's whole window is the same man, and it
    # must be bounded by 1 in magnitude.
    assert float(np.max(np.abs(P["qbnew"].values))) <= 1.0 + 1e-12
    assert float(np.max(np.abs(P["qbchg"].values))) <= 1.0 + 1e-12

    # QBEARN must be ZERO wherever the rating is at or below 1500, because the
    # claim is about a rating that was EARNED. If it were nonzero there, the
    # term would be charging bad teams for losing a starter they never
    # benefited from, which is the opposite of the hypothesis.
    bad = (P["home_above"].values == 0) & (P["away_above"].values == 0)
    if bad.sum() > 0:
        assert float(np.max(np.abs(P["qbearn"].values[bad]))) < 1e-12, (
            "QBEARN is non-zero on games where neither team is above 1500")

    # SIGN CONVENTION. Every column must be built so positive favours home.
    # Check it directly: rows where ONLY the home team has a discontinuity
    # must have qbnew <= 0 (home is the disadvantaged side).
    only_h = (P["home_new"].values > 0) & (P["away_new"].values == 0)
    if only_h.sum() > 0:
        assert float(np.max(P["qbnew"].values[only_h])) <= 0.0, (
            "qbnew is positive where only the HOME team changed QB — the sign "
            "convention is inverted and every verdict would read backwards")
    only_a = (P["away_new"].values > 0) & (P["home_new"].values == 0)
    if only_a.sum() > 0:
        assert float(np.min(P["qbnew"].values[only_a])) >= 0.0, (
            "qbnew is negative where only the AWAY team changed QB")

    # ---- leak proof -------------------------------------------------------
    # Tamper with a LATER game's result. Nothing about an earlier game's
    # features may move.
    g2 = g.copy()
    tgt = len(g2) - 40
    g2.loc[tgt, "home_score"] = 99
    g2.loc[tgt, "away_score"] = 0
    P2, _, _ = build_panel(g2)
    k = min(len(P), len(P2)) - 200
    cols = ["p_base", "qbchg", "qbnew", "qbexp", "qbearn", "qbres"]
    same = np.allclose(P[cols].values[:k], P2[cols].values[:k], atol=1e-12)
    assert same, "a later game's result leaked into earlier features"

    # ---- CLAIM: a REAL QB effect is recoverable ---------------------------
    # This is the whole point. If the pipeline cannot find an effect that was
    # planted into the data-generating process, then a null on the real panel
    # means nothing at all.
    gq = _synth(seed=11, qb_effect=0.85)
    Pq, _, _ = build_panel(gq)
    d_new, per_new, _, _ = score_angle(Pq, "qbnew")
    v_new, good, nn = verdict_of(d_new, per_new)
    d_chg, per_chg, _, _ = score_angle(Pq, "qbchg")
    out("planted QB effect: QBNEW %+.5f (%s, %d/%d)  QBCHG %+.5f"
        % (d_new, v_new, good, nn, d_chg))
    assert d_new > 0.0020 and v_new == "ROBUST WIN", (
        "a genuine 0.85-logit backup penalty was planted and QBNEW recovered "
        "%+.5f (%s). The pipeline cannot see an effect it was built to see, so "
        "a null on the real panel would be meaningless\n%s"
        % (d_new, v_new, "\n".join(buf)))

    # QBNEW must beat QBCHG on this panel: the plant makes a team worse for a
    # STRETCH of games, and a window fraction reads a stretch while a
    # since-last-game flag only reads its first game.
    assert d_new > d_chg, (
        "QBNEW %+.5f did not beat QBCHG %+.5f on a panel where the backup "
        "plays multi-game stretches. If the window buys nothing over a binary "
        "flag, QBNEW is QBCHG with extra arithmetic" % (d_new, d_chg))

    # ---- and a NULL panel must come back null -----------------------------
    d0, per0, _, _ = score_angle(P, "qbnew")
    v0, _, _ = verdict_of(d0, per0)
    assert v0 != "ROBUST WIN", (
        "QBNEW is a ROBUST WIN (%+.5f) on a panel where the QB genuinely does "
        "NOT matter — the test manufactures its own signal" % d0)

    # ---- ceiling reader rungs, pinned ------------------------------------
    assert "PROBE UNINFORMATIVE" in read_ceiling(-0.001, 0, 3, 3, 0.0, 0.6)
    assert "noise by construction" in read_ceiling(0.001, 0.001, 3, 3, 0.002, 0.6)
    assert "CANNOT BE SEEN" in read_ceiling(0.01, 0.01, 0, 3, 0.0001, 0.6)
    assert "WEAK PROBE" in read_ceiling(0.01, 0.01, 1, 3, 0.0001, 0.6)
    assert "DEAD" in read_ceiling(0.01, 0.01, 3, 3, 0.0001, 0.6)
    # and the rung that a robust win must NOT be read as dead
    assert "GATE 1 PASSED" in read_ceiling(
        0.01, 0.01, 3, 3, 0.006, 0.6, verdict="ROBUST WIN")
    assert "DEAD" in read_ceiling(
        0.01, 0.01, 3, 3, 0.0001, 0.6, verdict="win, not robust")

    # ---- the plant ladder must escalate, not just descend -----------------
    # Ask for a bound above an unreachable measured gain. The probe must step
    # UP rather than reporting a too-weak oracle and letting read_ceiling call
    # the measurement noise.
    pr_up = probe(Pq, "qbnew", seeds=(7,), d_real=0.05)
    assert pr_up["oracle"] > 0.0, (
        "the escalation check needs a probe whose FIRST rung is informative; "
        "this one came back %+.5f, so it exercised the step-DOWN early return "
        "instead" % pr_up["oracle"])
    assert pr_up["raised"], (
        "probe was asked to bound a measured gain of +0.05000 and never "
        "stepped the plant up — an angle whose oracle lands under its own "
        "measurement would be mislabelled 'noise by construction'")
    assert pr_up["plant_b"] > PLANT_LADDER[0], (
        "probe recorded raised rungs but returned plant_b=%.2f, no higher than "
        "the ladder's top" % pr_up["plant_b"])

    # ---- placebo must be reachable by noise, or it is not a test ---------
    hits, alpha, ge, p_eff = placebo(P, "qbnew", n=40, seed=2, d_real=d0)
    out("null-panel placebo: alpha %d/40 (%.3f)  p_eff %d/40 (%.3f)"
        % (hits, alpha, ge, p_eff))
    # On a panel where the QB does not matter, a shuffled column must match the
    # measured "effect" often. If p_eff were tiny here the comparison would be
    # rigged — the placebo would be unable to reach a value that is pure noise.
    assert p_eff > 0.10, (
        "shuffled columns beat a pure-noise measured gain only %.3f of the "
        "time — the p_eff comparison is not reachable and would rubber-stamp "
        "anything" % p_eff)

    print("NFL QB SELFTEST PASS — columns live and correctly signed, QBEARN "
          "silent below 1500, no leak from later games, a planted backup "
          "penalty is recovered (%+.5f ROBUST WIN) and beats the binary flag "
          "(%+.5f), and a null panel stays null (%+.5f)"
          % (d_new, d_chg, d0))
    return True


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    lines = []

    def out(s=""):
        print(s)
        lines.append(str(s))

    experiment(out=out)
    try:
        os.makedirs(os.path.dirname(VERDICT_OUT), exist_ok=True)
        with open(VERDICT_OUT, "w") as f:
            f.write("# NFL QB batch\n\n```\n" + "\n".join(lines) + "\n```\n")
        print("verdict -> %s" % VERDICT_OUT)
    except Exception as e:
        print("could not write verdict: %s" % e)
