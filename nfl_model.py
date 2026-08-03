"""
nfl_model.py  —  NFL rating model + honest backtest vs the CLOSING MARKET
=========================================================================
RECONSTRUCTED 2026-07-07 from the recovered original (conversation archive).
Data: nflverse games.csv (auto-downloaded; 1999-present, includes closing
moneylines/spreads/totals -- so we measure the model against the actual
market, not a proxy).

Model: Elo-family team ratings updated per game (margin-aware, K controlled,
preseason regression toward the mean), home field, rest days. Win prob from
rating diff via logistic. Deliberately simple -- the lesson from three sports.

Original validated benchmarks to reproduce (16 seasons, 2010+):
  model 64.6% | closing market 66.4% | 693 disagreements, model right 44.3%

RUN:  python nfl_model.py          (downloads data if missing, backtests, prints verdict)
"""
import os, sys, math, urllib.request
import numpy as np, pandas as pd

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
K, REVERT, SCALE = 20.0, 0.33, 400.0                  # Elo params
# ADAPTIVE home-field advantage. The old fixed HFA=48 was implicitly tuned on history
# that includes recent seasons (a look-ahead constant); the honest 2019-vintage pick
# (55) scores much worse 2020-2025. This EWMA tracks the realized home edge with zero
# look-ahead (HFA_LR=1.0 is an a-priori default, not train-tuned — the drift is a
# post-2020 phenomenon so no in-sample tuning is possible; 4/6 holdout seasons better).
HFA_INIT, HFA_LR = 50.0, 1.0
REST_PER_DAY = 4.0                                     # Elo per rest-day differential vs 7
DIV_TAU = 0.90   # divisional games: shrink the Elo edge toward 50% at PREDICTION time only
                 # (ratings/updates untouched). Walk-forward validated: holdout Brier -0.0003,
                 # improved 5/6 holdout seasons; division rivals upset favorites more often.

def load(refresh=True):
    """Load nflverse games.csv, REFRESHING it every run.

    This used to be `if not os.path.exists(path): download`. games.csv is a TRACKED
    file, so in CI actions/checkout always puts it there and the download branch never
    ran: every scheduled build re-read the snapshot frozen into the last commit. The
    consequences compound quietly --

      * final scores for the current season never arrive, so nfl_grade.grade_all()
        finds nothing to settle, the Calibration tab stays at n=0 all year, and the
        live market-disagreement study never gets a single game;
      * newly scheduled games and reschedules never appear;
      * the walk-forward "verified this build" strip re-reports last season's number
        while claiming it was recomputed from raw data this run.

    Measured 2026-08-03 on the committed copy: last game carrying a score was
    2026-02-08 (the 2025 Super Bowl). Nothing after it could ever be graded.

    So: pull every run, and fail SOFT to the cached copy when the pull fails -- a
    network blip must not take the whole build down -- but record the failure on
    load.source / load.error instead of pretending the cache is current. A truncated
    or unparseable download is rejected rather than written over a good cache.
    """
    path = os.path.join(HERE, "games.csv")
    load.source, load.error, load.fetched_rows = "cache", None, None
    if refresh or not os.path.exists(path):
        tmp = path + ".tmp"
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (NFLTool)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            fresh = pd.read_csv(tmp)                       # must parse
            need = {"season", "week", "gameday", "home_team", "away_team", "home_score"}
            if not need <= set(fresh.columns):
                raise ValueError(f"missing columns: {sorted(need - set(fresh.columns))}")
            # never let a truncated response clobber a good cache
            if os.path.exists(path):
                old_n = sum(1 for _ in open(path)) - 1
                if len(fresh) < old_n * 0.95:
                    raise ValueError(f"download has {len(fresh)} rows vs cached {old_n} — refusing")
            os.replace(tmp, path)
            load.source, load.fetched_rows = "nflverse", len(fresh)
        except Exception as e:
            try: os.remove(tmp)
            except OSError: pass
            load.error = f"{type(e).__name__}: {e}"
            if not os.path.exists(path):
                raise                                      # no cache to fall back to
            print(f"   ! games.csv refresh FAILED ({load.error}) — using the cached copy. "
                  f"Scores and schedule may be stale.")
    g = pd.read_csv(path)
    g = g[g["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    return g.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

def expected(dr): return 1.0 / (1.0 + 10 ** (-dr / SCALE))

def run_elo(g, start_season=2010):
    """Walk forward through every game; store pregame prediction, update after."""
    R = {}
    cur_season = None
    preds = []
    H = HFA_INIT                     # adaptive home-field advantage (leak-free EWMA)
    for r in g.itertuples():
        if r.season != cur_season:                     # preseason regression
            cur_season = r.season
            for t in R: R[t] = 1500 + (R[t] - 1500) * (1 - REVERT)
        h, a = r.home_team, r.away_team
        R.setdefault(h, 1500); R.setdefault(a, 1500)
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = REST_PER_DAY * ((r.home_rest - 7) - (r.away_rest - 7))
        neutral = str(r.location) == "Neutral"
        hfa = 0.0 if neutral else H
        dr = (R[h] + hfa + rest) - R[a]
        p_home = expected(dr)
        # divisional shrink applies to the REPORTED prediction only; Elo updates below
        # keep using the unshrunk p_home so ratings are byte-identical to before.
        _div = bool(getattr(r, "div_game", 0) == 1)
        p_pred = expected(dr * DIV_TAU) if _div else p_home
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin = r.home_score - r.away_score
            if r.season >= start_season:
                preds.append({"season": r.season, "week": r.week, "home": h, "away": a,
                              "p_home": p_pred, "home_win": int(margin > 0),
                              "tie": int(margin == 0),
                              "home_ml": r.home_moneyline, "away_ml": r.away_moneyline,
                              "spread": r.spread_line})
            # margin-aware K (538-style multiplier): dampens blowouts by favorites,
            # amplifies upsets — the autocorrelation correction
            mov = math.log(abs(margin) + 1) * (2.2 / ((0.001 * abs(dr) if margin * dr > 0 else -0.001 * abs(dr)) + 2.2))
            s_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            delta = K * mov * (s_home - p_home)
            R[h] += delta; R[a] -= delta
            if not neutral:
                # adaptive HFA: one gradient step on the home win residual. Tracks the
                # documented post-2020 HFA decline (~57 -> ~40 Elo) with zero look-ahead.
                H += HFA_LR * (s_home - p_home)
    return R, pd.DataFrame(preds), H

def _dec(ml):
    return ml / 100 + 1 if ml > 0 else 100 / (-ml) + 1

def market_p_home(row):
    """Devig the closing moneylines -> market P(home)."""
    if pd.isna(row.home_ml) or pd.isna(row.away_ml): return np.nan
    ih, ia = 1 / _dec(row.home_ml), 1 / _dec(row.away_ml)
    return ih / (ih + ia)

def backtest(P):
    P = P[P["tie"] == 0].copy()
    P["mkt"] = P.apply(market_p_home, axis=1)
    n = len(P)
    acc = ((P["p_home"] > 0.5).astype(int) == P["home_win"]).mean()
    brier = ((P["p_home"] - P["home_win"]) ** 2).mean()
    M = P.dropna(subset=["mkt"])
    macc = ((M["mkt"] > 0.5).astype(int) == M["home_win"]).mean()
    # LIKE FOR LIKE. `acc` above is over EVERY backtested game; `market_acc` can only
    # be over the games that carried closing moneylines. Printing those two side by
    # side compares the model on one game set to the market on another. acc_mkt is the
    # model's rate on the market's own subset, which is the only fair comparison.
    pacc = ((M["p_home"] > 0.5).astype(int) == M["home_win"]).mean() if len(M) else np.nan
    dis = M[((M["p_home"] > 0.5) != (M["mkt"] > 0.5))]
    dacc = ((dis["p_home"] > 0.5).astype(int) == dis["home_win"]).mean() if len(dis) else np.nan
    # THE HALF THIS PANEL WAS MISSING. It reported how often the MODEL was right when
    # it disagreed with the price and never how often the MARKET was on those same
    # games. Ties are dropped above, so here the two DO sum to 100 and 44.3% implies
    # 55.7% -- but a reader has to do that subtraction to see it, and 44.3 reads as
    # "a bit under half" until 55.7 is sitting next to it. It is counted, not derived,
    # so the number survives if the tie handling above ever changes.
    dmk = ((dis["mkt"] > 0.5).astype(int) == dis["home_win"]).mean() if len(dis) else np.nan
    return {"n": n, "acc": round(100 * acc, 1), "brier": round(brier, 4),
            "n_mkt": len(M), "market_acc": round(100 * macc, 1),
            "acc_mkt": (round(100 * pacc, 1) if len(M) else None),
            "n_disagree": len(dis), "model_right_in_disagree": round(100 * dacc, 1),
            "market_right_in_disagree": (round(100 * dmk, 1) if len(dis) else None)}

def state():
    """Live API for the publisher: current ratings + every unplayed scheduled
    game with a pregame p_home from today's Elo (HFA + rest applied)."""
    g = load()
    # Read these immediately — they are function attributes and the next load() wipes them.
    _scored = g[g["home_score"].notna()]
    state.data = {"source": load.source, "error": load.error,
                  "rows": int(len(g)),
                  "last_scored": (_scored["gameday"].max().strftime("%Y-%m-%d")
                                  if len(_scored) and pd.notna(_scored["gameday"].max()) else None),
                  "last_scheduled": (g["gameday"].max().strftime("%Y-%m-%d")
                                     if pd.notna(g["gameday"].max()) else None)}
    R, P, H = run_elo(g)
    bt = backtest(P)
    up = g[g["home_score"].isna()].copy()
    rows = []
    for r in up.itertuples():
        h, a = r.home_team, r.away_team
        if h not in R or a not in R: continue
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = REST_PER_DAY * ((r.home_rest - 7) - (r.away_rest - 7))
        hfa = 0.0 if str(r.location) == "Neutral" else H
        _dr = (R[h] + hfa + rest) - R[a]
        _tau = DIV_TAU if getattr(r, "div_game", 0) == 1 else 1.0
        rows.append({"season": int(r.season), "week": int(r.week),
                     "gameday": r.gameday, "home": h, "away": a,
                     "elo_home": round(R[h], 1), "elo_away": round(R[a], 1),
                     "p_home": round(100 * expected(_dr * _tau), 1),
                     "neutral": str(r.location) == "Neutral",
                     "spread_line": None if pd.isna(r.spread_line) else float(r.spread_line)})
    # R accumulates EVERY franchise code seen since 1999, so it carries relocation
    # ghosts: STL, SD and OAK are still in there. They stop being updated the year
    # the team moves but the preseason REVERT step keeps pulling them toward 1500,
    # so they converge on the middle of the table rather than falling off the bottom.
    # Measured 2026-08-03: 35 keys, with STL 1500 / SD 1498 / OAK 1494 sitting at
    # ranks 17-19. The publisher then took ratings[:32], which does not drop the
    # ghosts -- it drops the three genuinely WORST teams, LV 1352, NYJ 1360 and
    # TEN 1358. The dashboard's power ranking was showing three franchises that do
    # not exist and hiding three that do.
    #
    # Filter to the franchises on the newest season's schedule (which includes
    # unplayed games, so this is populated from schedule release onward) and return
    # the whole list -- the publisher must not truncate.
    cur = int(g["season"].max())
    _cs = g[g["season"] == cur]
    active = set(_cs["home_team"]) | set(_cs["away_team"])
    if len(active) < 20:
        # newest season barely populated (schedule not out yet): fall back a year
        # rather than publishing a handful of teams.
        _pv = g[g["season"] == int(g.loc[g["season"] < cur, "season"].max())]
        active |= set(_pv["home_team"]) | set(_pv["away_team"])
    ratings = sorted(({"team": t, "elo": round(v, 1)} for t, v in R.items() if t in active),
                     key=lambda x: -x["elo"])
    # hand the already-loaded frame to the publisher so it does not pull games.csv a
    # second time in the same build (and cannot end up grading a DIFFERENT snapshot
    # than the one the ratings were built from).
    state.games = g
    state.n_teams_dropped = len(R) - len(ratings)
    state.dropped = sorted(set(R) - active)
    assert len(ratings) >= 28, f"only {len(ratings)} active teams — franchise filter is wrong"
    return ratings, pd.DataFrame(rows), bt

def _selftest():
    """Pure checks on the pieces that do not need games.csv or a network.

    This module had NO selftest, and it is the one that computes the number the whole
    tool is judged on. The daily workflow ran no gate at all, so a change here would
    have published silently.
    """
    # devig: a symmetric pair must land on 0.5, and the favourite must exceed it.
    assert abs(market_p_home(type("R", (), {"home_ml": -110, "away_ml": -110})()) - 0.5) < 1e-9
    assert market_p_home(type("R", (), {"home_ml": -300, "away_ml": +250})()) > 0.5
    assert np.isnan(market_p_home(type("R", (), {"home_ml": np.nan, "away_ml": -110})()))
    # expected() is monotone and centred
    assert abs(expected(0) - 0.5) < 1e-12 and expected(100) > 0.5 > expected(-100)

    # BACKTEST SYMMETRY. Four games, all with closing lines, plus one with no line so
    # that `acc` (all games) and `acc_mkt` (priced subset) are forced apart -- the exact
    # apples-to-oranges this panel used to print next to market_acc.
    #   g1 model home .7 / mkt home .6 -> agree, home won      -> both right
    #   g2 model home .8 / mkt away .4 -> DISAGREE, home won   -> model right
    #   g3 model home .6 / mkt away .3 -> DISAGREE, away won   -> market right
    #   g4 model away .2 / mkt away .1 -> agree, away won      -> both right
    #   g5 model home .9, NO LINE, away won -> drags acc down, invisible to acc_mkt
    P = pd.DataFrame([
        {"tie": 0, "p_home": .7, "home_win": 1, "home_ml": -150, "away_ml": +130},
        {"tie": 0, "p_home": .8, "home_win": 1, "home_ml": +150, "away_ml": -170},
        {"tie": 0, "p_home": .6, "home_win": 0, "home_ml": +240, "away_ml": -280},
        {"tie": 0, "p_home": .2, "home_win": 0, "home_ml": +600, "away_ml": -800},
        {"tie": 0, "p_home": .9, "home_win": 0, "home_ml": np.nan, "away_ml": np.nan},
        {"tie": 1, "p_home": .5, "home_win": 0, "home_ml": -110, "away_ml": -110},
    ])
    b = backtest(P)
    assert b["n"] == 5 and b["n_mkt"] == 4, b          # the tie is dropped, the no-line game is not
    assert b["acc"] == 60.0, b                          # 3 of 5 over every game
    assert b["acc_mkt"] == 75.0, b                      # 3 of 4 over the priced subset
    assert b["acc"] != b["acc_mkt"], "the fixture must force these apart or it tests nothing"
    assert b["n_disagree"] == 2, b
    # THE HALF THAT WAS MISSING. One disagreement each way.
    assert b["model_right_in_disagree"] == 50.0, b
    assert b["market_right_in_disagree"] == 50.0, b
    # ties are dropped, so on this two-way market the two sum to 100. Asserted so that
    # a future change to tie handling has to come back through this test.
    assert b["model_right_in_disagree"] + b["market_right_in_disagree"] == 100.0, b
    # and with no priced games at all the panel must not divide by zero
    b0 = backtest(pd.DataFrame([{"tie": 0, "p_home": .7, "home_win": 1,
                                 "home_ml": np.nan, "away_ml": np.nan}]))
    assert b0["n_mkt"] == 0 and b0["acc_mkt"] is None and b0["market_right_in_disagree"] is None, b0
    print("NFL MODEL SELFTEST PASS — devig, Elo expectation, and a backtest that reports "
          "the model and the market over the SAME games, both halves of the disagreement")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest(); sys.exit(0)
    g = load()
    R, P, H = run_elo(g)
    bt = backtest(P)
    print(f"walk-forward {P['season'].min()}-{P['season'].max()}: {bt['n']} games")
    print(f"  model accuracy:  {bt['acc']}%   (benchmark: 64.6)")
    print(f"  model Brier:     {bt['brier']}")
    print(f"  market accuracy: {bt['market_acc']}%  on {bt['n_mkt']} games  (benchmark: 66.4)")
    print(f"  model on those same {bt['n_mkt']}: {bt['acc_mkt']}%")
    print(f"  disagreements:   {bt['n_disagree']}  — model right {bt['model_right_in_disagree']}%, "
          f"market right {bt['market_right_in_disagree']}%  (benchmark: 693 @ 44.3)")
    top = sorted(R.items(), key=lambda x: -x[1])[:5]
    print("  current top-5 Elo:", ", ".join(f"{t} {r:.0f}" for t, r in top))
