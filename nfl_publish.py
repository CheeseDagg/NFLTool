"""
nfl_publish.py — run the whole pipeline, emit data/slate.json for the dashboard.
Games window: next 30 days; OFFSEASON FALLBACK shows the next scheduled week
(recovered original behavior), so the page is never empty. Every write is
NaN-scrubbed with allow_nan=False (browsers reject bare NaN — learned the
hard way on MLB).
"""
import os, json, math, csv, datetime as dt
import pandas as pd
import nfl_model, nfl_edge, nfl_grade

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def _scrub(o):
    if isinstance(o, float) and not math.isfinite(o): return None
    if isinstance(o, dict):  return {k: _scrub(v) for k, v in o.items()}
    if isinstance(o, list):  return [_scrub(v) for v in o]
    return o

def main():
    os.makedirs(DATA, exist_ok=True)
    print("1) model + walk-forward…")
    ratings, up, bt = nfl_model.state()
    print(f"   {len(ratings)} teams | backtest: {bt['acc']}% vs market {bt['market_acc']}% "
          f"({bt['n_disagree']} disagreements @ {bt['model_right_in_disagree']}%)")

    print("2) upcoming window…")
    up["gameday"] = pd.to_datetime(up["gameday"], errors="coerce")
    now = pd.Timestamp(dt.date.today())
    win = up[(up["gameday"] >= now) & (up["gameday"] <= now + pd.Timedelta(days=30))]
    if not len(win):                                  # offseason: next scheduled week
        fut = up[up["gameday"] >= now]
        if len(fut):
            first = fut.sort_values("gameday").iloc[0]
            win = fut[(fut["season"] == first["season"]) & (fut["week"] == first["week"])]
    win = win.sort_values("gameday")
    print(f"   {len(win)} games (season {win['season'].iloc[0] if len(win) else '—'}, "
          f"week {win['week'].iloc[0] if len(win) else '—'})")

    print("3) edges (line-shop only — model is not a betting signal)…")
    odds_rows = []
    op = os.path.join(DATA, "nfl_odds.csv")
    if os.path.exists(op):
        with open(op) as f: odds_rows = list(csv.DictReader(f))
    edges = nfl_edge.find_edges(odds_rows) if odds_rows else []
    edge_note = "" if edges else ("no priced games yet (offseason)" if not odds_rows
                                  else "no side clears the 1% line-shop bar today")
    # market consensus per game for the prediction log
    mkt = {}
    if odds_rows:
        import statistics
        by = {}
        for r in odds_rows: by.setdefault((r["home"], r["away"]), []).append(r)
        for k, bks in by.items():
            fh = []
            for b in bks:
                ih, ia = 1/nfl_edge.dec(b["home_ml"]), 1/nfl_edge.dec(b["away_ml"])
                fh.append(ih/(ih+ia))
            mkt[k] = round(100*statistics.median(fh), 1)

    print("4) grade past predictions + log this slate…")
    g = nfl_model.load()
    done = g[g["home_score"].notna()]
    results = {(int(r.season), int(r.week), r.home_team, r.away_team):
               (r.home_score - r.away_score) for r in done.itertuples()}
    n_new, cal = nfl_grade.grade_all(results)
    n_log = nfl_grade.log_predictions(win, mkt) if len(win) else 0
    print(f"   graded {n_new} new | logged {n_log} new predictions | panel n={cal.get('n',0)}")

    games = []
    for r in win.itertuples():
        games.append({"season": r.season, "week": r.week,
                      "date": r.gameday.strftime("%a %b %-d") if pd.notna(r.gameday) else "",
                      "away": r.away, "home": r.home,
                      "elo_away": r.elo_away, "elo_home": r.elo_home,
                      "p_home": r.p_home, "neutral": bool(r.neutral),
                      "spread_line": r.spread_line,
                      "mkt_p": mkt.get((r.home, r.away))})
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "games": games, "ratings": ratings[:32],
           "edges": edges, "edge_note": edge_note,
           "backtest": bt, "cal": cal}
    with open(os.path.join(DATA, "slate.json"), "w") as f:
        json.dump(_scrub(out), f, indent=1, allow_nan=False)
    print(f"slate.json written: {len(games)} games, {len(edges)} edges, "
          f"cal n={cal.get('n',0)}")

if __name__ == "__main__":
    main()
