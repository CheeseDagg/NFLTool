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
K, HFA_PTS, REVERT, SCALE = 20.0, 48.0, 0.33, 400.0   # Elo params (48 Elo ~ 1.7 pts HFA)
REST_PER_DAY = 4.0                                     # Elo per rest-day differential vs 7
DIV_TAU = 0.90   # divisional games: shrink the Elo edge toward 50% at PREDICTION time only
                 # (ratings/updates untouched). Walk-forward validated: holdout Brier -0.0003,
                 # improved 5/6 holdout seasons; division rivals upset favorites more often.

def load():
    path = os.path.join(HERE, "games.csv")
    if not os.path.exists(path):
        print("downloading nflverse games.csv ...")
        urllib.request.urlretrieve(URL, path)
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
    for r in g.itertuples():
        if r.season != cur_season:                     # preseason regression
            cur_season = r.season
            for t in R: R[t] = 1500 + (R[t] - 1500) * (1 - REVERT)
        h, a = r.home_team, r.away_team
        R.setdefault(h, 1500); R.setdefault(a, 1500)
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = REST_PER_DAY * ((r.home_rest - 7) - (r.away_rest - 7))
        hfa = 0.0 if str(r.location) == "Neutral" else HFA_PTS
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
    return R, pd.DataFrame(preds)

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
    dis = M[((M["p_home"] > 0.5) != (M["mkt"] > 0.5))]
    dacc = ((dis["p_home"] > 0.5).astype(int) == dis["home_win"]).mean() if len(dis) else np.nan
    return {"n": n, "acc": round(100 * acc, 1), "brier": round(brier, 4),
            "n_mkt": len(M), "market_acc": round(100 * macc, 1),
            "n_disagree": len(dis), "model_right_in_disagree": round(100 * dacc, 1)}

def state():
    """Live API for the publisher: current ratings + every unplayed scheduled
    game with a pregame p_home from today's Elo (HFA + rest applied)."""
    g = load()
    R, P = run_elo(g)
    bt = backtest(P)
    up = g[g["home_score"].isna()].copy()
    rows = []
    for r in up.itertuples():
        h, a = r.home_team, r.away_team
        if h not in R or a not in R: continue
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = REST_PER_DAY * ((r.home_rest - 7) - (r.away_rest - 7))
        hfa = 0.0 if str(r.location) == "Neutral" else HFA_PTS
        _dr = (R[h] + hfa + rest) - R[a]
        _tau = DIV_TAU if getattr(r, "div_game", 0) == 1 else 1.0
        rows.append({"season": int(r.season), "week": int(r.week),
                     "gameday": r.gameday, "home": h, "away": a,
                     "elo_home": round(R[h], 1), "elo_away": round(R[a], 1),
                     "p_home": round(100 * expected(_dr * _tau), 1),
                     "neutral": str(r.location) == "Neutral",
                     "spread_line": None if pd.isna(r.spread_line) else float(r.spread_line)})
    ratings = sorted(({"team": t, "elo": round(v, 1)} for t, v in R.items()),
                     key=lambda x: -x["elo"])
    return ratings, pd.DataFrame(rows), bt

if __name__ == "__main__":
    g = load()
    R, P = run_elo(g)
    bt = backtest(P)
    print(f"walk-forward {P['season'].min()}-{P['season'].max()}: {bt['n']} games")
    print(f"  model accuracy:  {bt['acc']}%   (benchmark: 64.6)")
    print(f"  model Brier:     {bt['brier']}")
    print(f"  market accuracy: {bt['market_acc']}%  on {bt['n_mkt']} games  (benchmark: 66.4)")
    print(f"  disagreements:   {bt['n_disagree']}  — model right {bt['model_right_in_disagree']}%  (benchmark: 693 @ 44.3)")
    top = sorted(R.items(), key=lambda x: -x[1])[:5]
    print("  current top-5 Elo:", ", ".join(f"{t} {r:.0f}" for t, r in top))
