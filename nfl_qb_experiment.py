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

THE CAVEAT THAT TURNED OUT TO BE THE VERDICT. `home_qb_id` is the QB who
actually started the game. An earlier version of this header called that a
slight optimism — "at Sunday-morning prediction time you know the ANNOUNCED
starter, usually but not always the same man" — and concluded it only biased
toward a false positive. That was wrong by a category. nflverse populates
home_qb_id on all 7,276 PLAYED rows and on 0 of the 272 UNPLAYED rows
nfl_model.state() predicts. It is not a slightly-optimistic version of a live
column; there is no live column. Every angle here is uncomputable in
production, and all five now come back BLOCKED at gate 0. The measured gains
below are real and they are unshippable, which is a different sentence from
either "win" or "null" and the file prints it as one.

FIVE GATES, same rule as the UFC batches:
  0 AVAILABILITY      are the raw columns the feature reads even POPULATED on
                      the unplayed games production predicts? This gate was
                      missing, QBNEW cleared the other four, and it took manual
                      inspection to notice. Unlike gates 1-4 it is not advisory:
                      it blocks inside verdict_of, so an unavailable feature
                      cannot print ROBUST WIN at any effect size.
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
import inspect

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
# GATE 0 — AVAILABILITY. Is the feature computable on the games we PREDICT?
# ---------------------------------------------------------------------------
# WHY THIS GATE EXISTS. The leak proof in selftest() tests TIME-ORDERING: that a
# later game's result cannot move an earlier game's features. That is a real
# check and QBNEW passes it — along with all four numbered gates — while still
# being uncomputable in production. QBNEW reads `home_qb_id`/`away_qb_id`, the
# quarterback who ACTUALLY STARTED. nflverse populates those on every played row
# and on NONE of the unplayed rows nfl_model.state() predicts. So QBNEW scored a
# real backtest gain on a column that does not exist at the moment of use.
# Time-ordering and availability are different questions and the harness was
# only asking one of them. This is the other one.
#
# WHY THE CHECK IS AT THE RAW-COLUMN LEVEL AND NOT ON THE FEATURE'S OUTPUT.
# The obvious version of this gate — compute the feature on the unplayed slate,
# look for NaN — does not work, and QBNEW is precisely the counterexample. Hand
# build_panel a slate whose home_qb_id is NaN and _new_frac compares NaN against
# the window's real ids, matches none of them, and returns a perfectly finite
# 1.0: "brand new quarterback" for all 32 teams. A NaN sweep over the output
# sees a clean, fully-populated, entirely fictional column and passes it. The
# missingness has to be caught upstream, before arithmetic launders it away.
#
# WHY A SPY AND NOT A DECLARATION. A hand-maintained "columns this feature
# reads" list per angle is only as good as the next author's memory, and its
# failure mode is silence — forget to register and the gate waves you through,
# which is the same way QBNEW got here. The spy wraps the games frame and
# records every column any builder touches, so a new column read is a new column
# checked with nothing to remember. The residual hole (a builder reaching for
# data through an access path the spy does not intercept, e.g. .loc or .values)
# is closed by verify_spy_coverage below, which poisons the columns the spy
# claims are UNREAD and fails if the panel moves anyway.

# Columns that DEFINE the unplayed slate. A builder must read the score to know
# a game has not been played yet, so reading them is not a violation — that they
# are 100% null on unplayed rows is the whole selector. What a builder may not
# do is let them into a feature VALUE, and the time-ordering leak proof is what
# polices that. Keeping this list tiny and explicit matters: every name here is
# a hole in the gate, so nothing goes in that is not literally the outcome.
OUTCOME_COLS = frozenset({"home_score", "away_score", "result", "total",
                          "overtime"})

# A column is UNAVAILABLE if it is null on any unplayed row at all. Zero, not
# "mostly populated": a feature that silently degrades on the 3 games a week it
# cannot see is still shipping a number nobody can explain, and the gate exists
# to make that a decision rather than an accident.
MAX_UNPLAYED_NULL = 0.0

# Populated by run_availability_gate(); consulted by verdict_of(). Module state
# rather than a parameter because the point is that no call site can forget it.
AVAILABILITY = {}


class _RowSpy:
    """One itertuples row that records which fields were read.

    getattr on the wrapped row runs FIRST so a missing column still raises
    AttributeError and `getattr(r, "div_game", 0)` keeps working — a column that
    does not exist was not read, and recording it would put phantom names in
    front of the null check.
    """

    __slots__ = ("_r", "_seen")

    def __init__(self, r, seen):
        object.__setattr__(self, "_r", r)
        object.__setattr__(self, "_seen", seen)

    def __getattr__(self, name):
        v = getattr(self._r, name)
        self._seen.add(name)
        return v


class ColumnSpy:
    """A games frame that records every raw column a feature builder reads.

    Supports the access patterns builders actually use — itertuples, column
    indexing, boolean masking — and forwards everything else to the real frame,
    RE-WRAPPING any DataFrame that comes back. That re-wrap is the part that
    matters: nfl_model.state() does `g[g["home_score"].isna()].copy()` before it
    iterates, and a spy that stopped at the first .copy() would report the
    prediction loop as reading nothing at all and pass every feature on earth.
    """

    def __init__(self, df, seen=None):
        self.__dict__["_df"] = df
        self.__dict__["_seen"] = set() if seen is None else seen

    @property
    def seen(self):
        return self.__dict__["_seen"]

    def _wrap(self, v):
        return ColumnSpy(v, self.seen) if isinstance(v, pd.DataFrame) else v

    def itertuples(self, *a, **kw):
        for r in self.__dict__["_df"].itertuples(*a, **kw):
            yield _RowSpy(r, self.seen)

    def __getitem__(self, key):
        if isinstance(key, str):
            self.seen.add(key)
        elif isinstance(key, (list, tuple)) and all(isinstance(k, str)
                                                    for k in key):
            self.seen.update(key)
        return self._wrap(self.__dict__["_df"][key])

    def __setitem__(self, key, val):
        self.__dict__["_df"][key] = val

    def merge(self, *a, **kw):
        """A JOIN KEY IS A READ. nfl_epa_experiment attaches its per-QB EPA with
        `g.merge(qfeat, on=["game_id", "home_qb_id"])` — the column name only
        ever appears inside a keyword argument, so an attribute/subscript trace
        never sees it and the feature looks like it reads nothing but game_id.
        Recording on/left_on is what makes the EPA harness auditable at all."""
        for k in ("on", "left_on"):
            v = kw.get(k)
            if isinstance(v, str):
                self.seen.add(v)
            elif isinstance(v, (list, tuple)):
                self.seen.update(x for x in v if isinstance(x, str))
        return self._wrap(self.__dict__["_df"].merge(*a, **kw))

    def __len__(self):
        return len(self.__dict__["_df"])

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        v = getattr(self.__dict__["_df"], name)
        # isroutine, not callable: pandas' .loc/.iloc accessors are themselves
        # callable objects, and wrapping them in a plain function breaks the
        # `gg.loc[...]` subscript. Only real methods get the re-wrap treatment;
        # accessors pass through untraced, which is precisely the blind spot
        # verify_spy_coverage is there to close.
        if inspect.isroutine(v):
            def _traced(*a, **kw):
                return self._wrap(v(*a, **kw))
            return _traced
        return self._wrap(v)


def columns_touched(builder, g):
    """Run `builder` against a spied copy of `g`; return the raw columns it read.

    The builder runs over the FULL frame, not just the unplayed slate, because a
    column read only inside a played-games branch is still a column the feature
    depends on. Restricting the trace to unplayed rows would let a builder hide
    its inputs behind `if played:` — which is exactly where build_panel reads
    home_qb_id.
    """
    spy = ColumnSpy(g)
    builder(spy)
    # itertuples exposes Index/count/index alongside the real fields; intersect
    # with the frame's columns so bookkeeping attributes are not audited.
    return set(spy.seen) & set(g.columns)


def unplayed_slate(g):
    """The rows production actually predicts — the same selector nfl_model.state
    uses. If this ever stops matching state(), the gate is auditing a different
    population than the one at risk, so it is written once and read from here."""
    return g[g["home_score"].isna()]


def null_rates(g, cols=None):
    """Fraction of the UNPLAYED slate on which each column is null."""
    up = unplayed_slate(g)
    assert len(up) > 0, (
        "no unplayed games in games.csv, so the availability gate has nothing "
        "to check. A gate that silently passes on an empty slate is worse than "
        "no gate — refresh games.csv before trusting any verdict")
    cols = list(g.columns) if cols is None else list(cols)
    return {c: float(up[c].isna().mean()) for c in cols if c in up.columns}


def _poisoned(builder_panel, g, col):
    """Rebuild the panel with `col` blanked. None means the build DIED.

    A builder that raises on a nulled column is the loudest possible statement
    that it depends on it, so the exception is caught and turned into a
    dependency rather than a crash — otherwise the gate falls over precisely on
    the features it exists to catch.
    """
    gp = g.copy()
    # .where(False) rather than `= np.nan`, because assigning a float NaN
    # RETYPES a string column to float64 and a downstream merge then dies with
    # "trying to merge on float64 and str". That crash is a dtype accident, not
    # a dependency, and reading it as one attributes every offending column to
    # every feature — which is how the EPA harness's prior-only team EPA got
    # blamed on the quarterback id. Blanking in place keeps the dtype.
    gp[col] = gp[col].where(pd.Series(False, index=gp.index))
    try:
        out = builder_panel(gp)
    except Exception:
        return None
    return out


def derived_columns(ref, g, feature_cols=()):
    """The panel columns a builder DERIVED, as opposed to carried through.

    nfl_epa_experiment's builder returns `g.merge(...)`, so its panel is the
    whole games table plus a few EPA columns. Comparing whole panels would then
    make blanking ANY raw column — referee, wind, the moneylines — count as a
    dependency, and the coverage check would fire on nineteen columns none of
    which touch a feature. A passthrough is not a read. Compare what the builder
    computed.
    """
    cols = [c for c in ref.columns if c not in set(g.columns)]
    cols += [c for c in feature_cols if c in ref.columns and c not in cols]
    return cols


def verify_spy_coverage(builder_panel, g, touched, suspect_cols,
                        feature_cols=()):
    """Belt and braces: poison the columns the spy says were NOT read.

    The spy sees itertuples, [] and merge keys. If some future builder reaches
    into the frame another way (.loc, .values, a groupby on the raw frame), the
    trace would come back short and the gate would pass a feature it never
    looked at. So take the columns the spy claims are unread AND are unavailable
    on the slate, blank each one, and rebuild. If the DERIVED columns move, the
    spy under-reported, and that is a hard error about the GATE rather than
    about the feature — silently trusting a trace that has been proven wrong is
    how the first version of this file got written.
    """
    ref = builder_panel(g)
    cmp_cols = derived_columns(ref, g, feature_cols)
    missed = []
    for c in suspect_cols:
        if c in touched or c in OUTCOME_COLS:
            continue
        alt = _poisoned(builder_panel, g, c)
        if alt is None or not _panels_identical(ref, alt, cmp_cols):
            missed.append(c)
    return missed


def _panels_identical(A, B, cols=None):
    if len(A) != len(B):
        return False
    cols = list(A.columns) if cols is None else list(cols)
    for c in cols:
        if c not in B.columns:
            return False
        a, b = A[c].values, B[c].values
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            if not np.allclose(a, b, atol=1e-12, equal_nan=True):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


# Row keys tried in order when lining a poisoned panel up against the reference.
# Different harnesses name the same game differently and a wrong key silently
# produces a many-to-many merge, which reads as "everything changed" and
# attributes every offending column to every feature.
PANEL_KEYS = (("game_id",),
              ("season", "week", "home", "away"),
              ("season", "week", "home_team", "away_team"))


def attribute_columns(builder_panel, g, feature_cols, bad_cols):
    """Which FEATURE depends on which unavailable RAW column.

    The spy answers at builder granularity ("something in build_panel reads
    home_qb_id"); a verdict is printed per angle, so the gate has to close the
    gap. Blank one offending raw column, rebuild, and see which feature columns
    move. Rows that vanish count as a dependency too — build_panel skips games
    with no qb_id, and "the panel cannot even be built without this column" is a
    stronger dependency than a changed value, not a weaker one.
    """
    ref = builder_panel(g)
    key = next((list(k) for k in PANEL_KEYS
                if all(c in ref.columns for c in k)), [])
    dep = {f: [] for f in feature_cols}
    for c in bad_cols:
        alt = _poisoned(builder_panel, g, c)
        if alt is None or len(alt) == 0:
            # Nulling the column destroyed the panel: every feature on it is
            # downstream of the column by definition.
            for f in feature_cols:
                dep[f].append(c)
            continue
        if key:
            m = ref.merge(alt, on=key, how="inner", suffixes=("", "_alt"))
            lost = len(ref) - len(m)
        elif len(alt) == len(ref):
            # No usable key but the same rows in the same order: compare
            # positionally rather than declaring a blanket dependency, which
            # would name every column in the reason string and tell nobody
            # anything.
            m = pd.concat([ref.reset_index(drop=True),
                           alt.reset_index(drop=True).add_suffix("_alt")],
                          axis=1)
            lost = 0
        else:
            for f in feature_cols:
                dep[f].append(c)
            continue
        for f in feature_cols:
            if f not in m.columns or f + "_alt" not in m.columns:
                dep[f].append(c)
                continue
            moved = not np.allclose(pd.to_numeric(m[f], errors="coerce").values,
                                    pd.to_numeric(m[f + "_alt"],
                                                  errors="coerce").values,
                                    atol=1e-12, equal_nan=True)
            if moved or lost > 0:
                dep[f].append(c)
    return dep


def run_availability_gate(builder_panel, g, feature_cols, label="",
                          out=print, register=True, verify=True):
    """THE GATE. Fails loudly and by name; returns {feature: reason or None}.

    `builder_panel` is any callable frame -> panel DataFrame, so this works for
    build_panel, for a single production feature, or for nfl_model's own
    prediction path — the whole point is that it is not specific to QBNEW.
    """
    up = unplayed_slate(g)
    rates = null_rates(g)
    touched = columns_touched(builder_panel, g)
    audited = sorted(touched - OUTCOME_COLS)
    bad = sorted(c for c in audited if rates.get(c, 0.0) > MAX_UNPLAYED_NULL)

    out("--- GATE 0: AVAILABILITY %s" % label)
    out("    production predicts %d unplayed games; a feature that reads a "
        "column" % len(up))
    out("    null on those rows scored its backtest gain on data it will never "
        "have.")
    out("    columns read by this builder: %s" % ", ".join(audited))
    for c in audited:
        r = rates.get(c, 0.0)
        out("      %-18s unplayed-null %6.1f%%   %s"
            % (c, 100 * r, "OK" if r <= MAX_UNPLAYED_NULL else "UNAVAILABLE"))

    if verify:
        suspect = [c for c, r in rates.items() if r > MAX_UNPLAYED_NULL]
        missed = verify_spy_coverage(builder_panel, g, touched, suspect,
                                     feature_cols)
        assert not missed, (
            "AVAILABILITY GATE IS BROKEN: blanking %s changed the panel even "
            "though the column spy never saw it read. The trace is incomplete, "
            "so every PASS this gate has printed is unverified." % missed)

    result = {f: None for f in feature_cols}
    if bad:
        dep = attribute_columns(builder_panel, g, feature_cols, bad)
        for f in feature_cols:
            if dep[f]:
                result[f] = ("reads %s, null on %s of the %d unplayed games"
                             % ("/".join(dep[f]),
                                "/".join("%.0f%%" % (100 * rates[c])
                                         for c in dep[f]), len(up)))
        out("    UNAVAILABLE COLUMNS: %s" % ", ".join(bad))
        for f in feature_cols:
            out("      %-10s %s" % (f, result[f] or "available"))
    else:
        out("    all columns available on the live slate — gate passed")
    out("")

    if register:
        AVAILABILITY.update(result)
    return result


def qb_panel_frame(g):
    """build_panel's panel alone — the shape run_availability_gate expects."""
    return build_panel(g)[0]


def production_rest_panel(g):
    """A CONTROL for the gate: a feature that IS legitimately available.

    The rest-day differential, the neutral-site flag and the divisional flag are
    what nfl_model already applies to every live prediction, so by construction
    they must clear a gate about live computability. A gate that has only ever
    been demonstrated failing is not evidence it can pass — without this control
    a gate that rejected everything would look exactly as convincing as a
    correct one.
    """
    rows = []
    for r in g.itertuples():
        if not (pd.notna(r.home_score) and pd.notna(r.away_score)):
            continue
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = NM.REST_PER_DAY * ((r.home_rest - 7) - (r.away_rest - 7))
        rows.append({"season": int(r.season), "week": int(r.week),
                     "home": r.home_team, "away": r.away_team,
                     "restdiff": rest,
                     "hfa": 0.0 if str(r.location) == "Neutral" else 1.0,
                     "div": float(getattr(r, "div_game", 0) == 1)})
    return pd.DataFrame(rows)


PRODUCTION_FEATURES = ["restdiff", "hfa", "div"]

# nfl_model.load() reads these off the CSV before any spy can be attached (it
# does the read itself), so they are declared rather than traced. Declared and
# audited beats invisible: they are the filter/sort keys, and a null game_type
# on the live slate would silently drop games from the prediction set.
LOADER_COLS = frozenset({"game_type", "gameday", "season", "week"})


def audit_production_model(g, out=print):
    """Null-rate audit of every raw column nfl_model actually reads, split by
    WHERE it reads it. Returns (backtest_cols, prediction_cols, rates).

    TWO traces, not one, because the split IS the finding. run_elo reads
    home_moneyline to score the backtest and moneylines are 75% null on the
    unplayed slate — which is fine, a backtest only ever runs on played games.
    The same column read inside state()'s unplayed loop would be a live bug.
    Merging the traces would either raise a false alarm on the moneylines or
    bury a real alarm underneath them.

    The prediction trace is taken by handing state() a spied frame and stubbing
    run_elo with its already-computed answer, so the only thing still touching
    the frame is the unplayed loop itself. Tracing state() rather than
    re-deriving its column list by hand is the point: a hand list goes stale the
    first time somebody adds a term to the live prediction.
    """
    spy_bt = ColumnSpy(g)
    NM.run_elo(spy_bt, start_season=TRAIN_LO)
    bt = set(spy_bt.seen) & set(g.columns)

    R, P, H = NM.run_elo(g)
    spy_pr = ColumnSpy(g)
    _load, _elo = NM.load, NM.run_elo
    try:
        NM.load = lambda: spy_pr
        NM.run_elo = lambda gg, **kw: (R, P, H)
        NM.state()
    finally:
        NM.load, NM.run_elo = _load, _elo
    pr = set(spy_pr.seen) & set(g.columns)

    rates = null_rates(g)
    up = unplayed_slate(g)
    out("--- AUDIT: every raw column nfl_model reads, vs the %d unplayed rows "
        "it predicts" % len(up))
    for c in sorted((bt | pr | LOADER_COLS) & set(g.columns)):
        where = ("PREDICTION" if c in pr and c not in OUTCOME_COLS
                 else ("outcome/selector" if c in OUTCOME_COLS
                       else ("backtest only" if c in bt else "load/filter")))
        r = rates.get(c, 0.0)
        flag = ""
        if c in pr and c not in OUTCOME_COLS and r > MAX_UNPLAYED_NULL:
            # 100% null = the term is uncomputable on every live game. Partial
            # = computable on some of the slate, which is a DECISION (guard and
            # degrade, or drop the term) and the gate's job is to force it to be
            # made on purpose rather than discovered in a dashboard.
            flag = ("  <-- UNCOMPUTABLE LIVE" if r >= 1.0
                    else "  <-- PARTIAL: confirm the read is guarded")
        out("    %-18s %-16s unplayed-null %6.1f%%%s" % (c, where, 100 * r,
                                                         flag))
    live_bad = sorted(c for c in pr
                      if c not in OUTCOME_COLS
                      and rates.get(c, 0.0) > MAX_UNPLAYED_NULL)
    out("    prediction-path columns not fully available: %s"
        % (", ".join(live_bad) if live_bad else "none"))
    out("")
    return bt, pr, rates


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


def verdict_of(d_ll, per, col=None, enforce=True):
    """The single place the string "ROBUST WIN" is produced, which is why the
    availability gate is enforced HERE rather than printed alongside gates 1-4.

    Gates 1, 2 and 4 are advisory: they print a line and a human decides. That
    is how QBNEW got to the edge of production with a clean sheet. Availability
    is not a judgement call — a feature whose inputs do not exist at prediction
    time cannot be shipped at any effect size — so it blocks at the chokepoint
    and no caller can forget to consult it. `col` is REQUIRED once the gate has
    run: an angle that was never audited raises rather than passing quietly,
    because the failure mode being fixed is a feature slipping through unasked.

    enforce=False is for the two callers that are NOT judging a feature. The
    probe's plant counter and the placebo's alpha both ask "how often does the
    ship rule fire on outcomes I generated", which is a property of the
    STATISTICS, and blocking those makes gate 1 report "a planted effect was
    never recovered" when the truth is that the counter was gagged. The reported
    verdict for an angle always enforces; the calibration counters never do.
    """
    good = sum(1 for v in per.values() if v > 0)
    n = len(per)
    if enforce and AVAILABILITY:
        assert col is not None and col in AVAILABILITY, (
            "verdict_of was asked to judge %r but the availability gate never "
            "audited it. Add it to the gate's feature list or the gate is "
            "decorative." % (col,))
        if AVAILABILITY[col]:
            return "BLOCKED (unavailable at prediction time)", good, n
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
            # enforce=False: this counts how often a PLANTED effect survives
            # the ship rule, i.e. the pipeline's power. Availability is a fact
            # about the column, not about the synthetic outcomes being scored.
            v, _, _ = verdict_of(d, per, col, enforce=False)
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
        # enforce=False for the same reason as in probe(): alpha is the ship
        # rule's false-positive rate on shuffled noise, and a blocked column
        # would report alpha 0 and read as a gate that never fires.
        v, _, _ = verdict_of(d, per, col, enforce=False)
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

    # GATE 0 RUNS FIRST AND ON PURPOSE. Gates 1-4 are all measurements of how
    # much an angle buys; there is no point paying for them before establishing
    # that the angle can be computed at all, and running the cheap disqualifier
    # last is how a disqualified feature ends up with four green lines above it.
    out("-" * 78)
    AVAILABILITY.clear()
    run_availability_gate(qb_panel_frame, g, [c for _, c in ANGLES],
                          label="— the five QB angles", out=out)
    # The control. Same gate, same slate, a feature production already ships.
    run_availability_gate(production_rest_panel, g, PRODUCTION_FEATURES,
                          label="— CONTROL: rest/HFA/divisional, already live",
                          out=out, register=False)
    audit_production_model(g, out=out)

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
            v, good, nn = verdict_of(d, per, col)
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
        v, good, nn = verdict_of(d, per, col)
        out("%-46s one-sided-change subset dLL %+.5f  %d/%d  -> %s"
            % (name, d, good, nn, v))

    out("=" * 78)
    out("Ship rule: GATE 0 clear, then a ROBUST WIN in PASS 2, a measured gain")
    out("under the ORACLE bound, a placebo that does not fire, and a shape that")
    out("holds. Gate 0 is a veto and not a vote: gates 1-4 measure how much an")
    out("angle buys, and no size of gain makes a column that does not exist on")
    out("the live slate shippable.")
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


def _with_unplayed_slate(g, n_games=16):
    """Append an UNPLAYED week shaped like the real one nflverse ships.

    _synth only makes finished games, so the availability gate would find an
    empty slate and its own assertion would fire — which proves nothing about
    the gate. The appended rows carry the fields that ARE known before kickoff
    (teams, rest, location, divisional flag) and leave null exactly what nflverse
    leaves null on a future game: the scores and BOTH quarterback ids. The whole
    defect being tested is that a schedule row does not know who will start.
    """
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    last = g.iloc[-1]
    rows = []
    for i in range(n_games):
        h, a = teams[(2 * i) % len(teams)], teams[(2 * i + 1) % len(teams)]
        rows.append(dict(
            game_id="future_%02d" % i, season=int(last["season"]),
            game_type="REG", week=int(last["week"]) + 1,
            gameday=last["gameday"] + pd.Timedelta(days=7), weekday="Sunday",
            away_team=a, away_score=np.nan, home_team=h, home_score=np.nan,
            location="Home", result=np.nan, total=np.nan, overtime=np.nan,
            home_rest=7, away_rest=7, div_game=0,
            home_moneyline=np.nan, away_moneyline=np.nan, spread_line=np.nan,
            home_qb_id=np.nan, away_qb_id=np.nan,
            home_qb_name=np.nan, away_qb_name=np.nan))
    up = pd.DataFrame(rows)
    return pd.concat([g, up.reindex(columns=g.columns)],
                     ignore_index=True)


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

    # ---- GATE 0: availability, BOTH directions ----------------------------
    # The leak proof above is a TIME-ORDERING proof and QBNEW passes it. It has
    # to be paired with an AVAILABILITY proof or the pair of them still lets an
    # uncomputable feature through, which is the exact history of this file.
    gs = _with_unplayed_slate(g)
    assert len(unplayed_slate(gs)) == 16

    # DIRECTION 1 — the real defect must FAIL, by name.
    bad = run_availability_gate(qb_panel_frame, gs, [c for _, c in ANGLES],
                                out=lambda s="": None, register=False)
    for col in ("qbchg", "qbnew", "qbexp", "qbearn", "qbres"):
        assert bad[col], (
            "%s reads the started-QB id and the availability gate passed it — "
            "the gate cannot see the bug it was written for" % col)
        assert "home_qb_id" in bad[col] and "away_qb_id" in bad[col], (
            "%s was blocked but the reason does not name the offending "
            "columns: %r. A gate that fails without naming the column is a "
            "gate nobody can act on" % (col, bad[col]))

    # DIRECTION 2 — a feature production already ships must PASS. Without this
    # a gate that rejected every input on earth would look identical to a
    # correct one from the QBNEW output alone.
    ok = run_availability_gate(production_rest_panel, gs, PRODUCTION_FEATURES,
                               out=lambda s="": None, register=False)
    assert all(v is None for v in ok.values()), (
        "the rest/HFA/divisional terms are computed on every live prediction "
        "today and the gate blocked them: %r" % ok)

    # THE BLOCK MUST BITE. A blocked column may not print ROBUST WIN no matter
    # how good its holdout numbers are — that is the entire point of putting the
    # check inside verdict_of instead of printing it next to gates 1-4.
    winner = {2020: 0.01, 2021: 0.01, 2022: 0.01, 2023: 0.01,
              2024: 0.01, 2025: 0.01}
    _saved = dict(AVAILABILITY)
    try:
        AVAILABILITY.clear()
        AVAILABILITY.update({"qbnew": "reads home_qb_id", "restdiff": None})
        v_block, _, _ = verdict_of(0.0100, winner, "qbnew")
        assert v_block.startswith("BLOCKED"), (
            "an unavailable column scored +0.01 across 6/6 holdout seasons and "
            "verdict_of returned %r — the gate does not block a WIN" % v_block)
        v_ok, _, _ = verdict_of(0.0100, winner, "restdiff")
        assert v_ok == "ROBUST WIN", (
            "an AVAILABLE column with the same numbers came back %r — the gate "
            "is blocking everything, not just the unavailable" % v_ok)
        # An angle nobody audited must RAISE, not sail through. Silence is how
        # the next feature repeats this bug.
        try:
            verdict_of(0.0100, winner, "qbnever_registered")
            raise AssertionError(
                "verdict_of judged a column the gate never audited instead of "
                "raising — a new angle could skip the gate by not being in it")
        except AssertionError as e:
            assert "availability gate never audited" in str(e), e
    finally:
        AVAILABILITY.clear()
        AVAILABILITY.update(_saved)

    # THE SPY'S BLIND SPOT MUST BE COVERED. A builder that reaches for a column
    # through .loc is invisible to the trace; verify_spy_coverage exists to
    # catch exactly that, and if it ever stops working the gate degrades to
    # "whatever the spy happened to see" without saying so.
    def _sneaky(gg):
        pnl = production_rest_panel(gg)
        # read spread_line (null on the live slate) WITHOUT the spy seeing it.
        # Counting non-nulls rather than summing, because _synth's spread_line
        # is a constant 0.0 and a sum of it survives blanking unchanged — the
        # poison check can only see a column the builder genuinely depends on.
        pnl["sneak"] = float(gg.loc[:, "spread_line"].notna().sum())
        return pnl

    touched_sneaky = columns_touched(_sneaky, gs)
    assert "spread_line" not in touched_sneaky, (
        "the .loc read was traced after all, so this case no longer exercises "
        "the spy's blind spot and verify_spy_coverage is untested")
    missed = verify_spy_coverage(
        _sneaky, gs, touched_sneaky,
        [c for c, r in null_rates(gs).items() if r > MAX_UNPLAYED_NULL])
    assert "spread_line" in missed, (
        "a builder read an unavailable column behind the spy's back and "
        "verify_spy_coverage did not notice — the gate silently degrades to "
        "trusting an incomplete trace")

    # A JOIN KEY IS A READ. nfl_epa_experiment attaches its per-QB feature with
    # merge(on=[..., "home_qb_id"]) and nothing else in that function mentions
    # the column, so a trace that only watches attribute and subscript access
    # reports the QB-EPA term as depending on game_id alone — a clean pass on
    # the same defect this gate exists to catch, in a different harness.
    def _joiner(gg):
        side = pd.DataFrame({"home_qb_id": ["x"], "junk": [1.0]})
        gg.merge(side, on=["home_qb_id"], how="left")
        return production_rest_panel(gg)

    assert "home_qb_id" in columns_touched(_joiner, gs), (
        "a merge key was not recorded as a column read — any feature that "
        "joins on the started-QB id would pass this gate")

    # ---- CLAIM: a REAL QB effect is recoverable ---------------------------
    # This is the whole point. If the pipeline cannot find an effect that was
    # planted into the data-generating process, then a null on the real panel
    # means nothing at all.
    gq = _synth(seed=11, qb_effect=0.85)
    Pq, _, _ = build_panel(gq)
    d_new, per_new, _, _ = score_angle(Pq, "qbnew")
    v_new, good, nn = verdict_of(d_new, per_new, "qbnew")
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
    v0, _, _ = verdict_of(d0, per0, "qbnew")
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
          "silent below 1500, no leak from later games, the availability gate "
          "blocks the started-QB columns by name and passes the live "
          "rest/HFA/divisional terms, a planted backup penalty is recovered "
          "(%+.5f ROBUST WIN) and beats the binary flag (%+.5f), and a null "
          "panel stays null (%+.5f)" % (d_new, d_chg, d0))
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
