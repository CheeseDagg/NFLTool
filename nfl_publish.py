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
    # function attributes — read now, before anything else calls state()/load()
    dstat = dict(getattr(nfl_model.state, "data", {}))
    dstat["dropped_franchises"] = getattr(nfl_model.state, "dropped", [])
    print(f"   {len(ratings)} teams | backtest: {bt['acc']}% vs market {bt['market_acc']}% "
          f"({bt['n_disagree']} disagreements @ {bt['model_right_in_disagree']}%)")
    print(f"   games.csv from {dstat.get('source')} | {dstat.get('rows')} rows | "
          f"last scored {dstat.get('last_scored')}"
          + (f" | REFRESH FAILED: {dstat['error']}" if dstat.get("error") else ""))

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
    # An empty odds file used to be reported as "no priced games yet (offseason)"
    # unconditionally, but nfl_odds.py writes an empty file only when the API
    # ANSWERED with zero events; when the pull fails it now keeps the old file and
    # records why in nfl_odds_status.json. Read that instead of guessing from the
    # CSV, or a dead API key reads as a quiet week forever.
    ostat = {}
    sp = os.path.join(DATA, "nfl_odds_status.json")
    if os.path.exists(sp):
        try:
            with open(sp) as f: ostat = json.load(f)
        except Exception:
            ostat = {}
    edges = nfl_edge.find_edges(odds_rows) if odds_rows else []
    _unb = getattr(nfl_edge.find_edges, "_no_bettable", 0)
    if edges:
        edge_note = ""
    elif ostat.get("ok") is False:
        edge_note = (f"odds pull FAILED ({ostat.get('error', 'unknown')}) at "
                     f"{ostat.get('checked', '?')} — prices below, if any, are stale. "
                     f"This is not an offseason signal.")
    elif not odds_rows:
        edge_note = "the book feed answered with zero priced games (offseason)"
    elif _unb and not any(nfl_edge.is_bettable(r["book"]) for r in odds_rows):
        edge_note = (f"{_unb} game(s) priced, none at "
                     f"{'/'.join(sorted(nfl_edge.BETTABLE))} — nothing quotable")
    else:
        edge_note = "no side clears the 1% line-shop bar today"
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
    # the SAME frame state() rated off — not a second download (see nfl_model.state)
    g = getattr(nfl_model.state, "games", None)
    if g is None: g = nfl_model.load()
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
                      "date_iso": r.gameday.strftime("%Y-%m-%d") if pd.notna(r.gameday) else "",
                      "away": r.away, "home": r.home,
                      "elo_away": r.elo_away, "elo_home": r.elo_home,
                      "p_home": r.p_home, "neutral": bool(r.neutral),
                      "spread_line": r.spread_line,
                      "mkt_p": mkt.get((r.home, r.away))})
    # Real freshness signal = the games' own dates, not wall-clock `generated` (which is
    # stamped every run). If the pipeline stops publishing, slate_end falls into the past
    # and the dashboard can warn instead of showing a frozen week as current.
    _gd = sorted(g["date_iso"] for g in games if g["date_iso"])
    out = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "slate_date": _gd[0] if _gd else None,
           "slate_end": _gd[-1] if _gd else None,
           # NOT ratings[:32]. nfl_model.state() already filters to the franchises on
           # the current schedule; slicing the top 32 off a 35-key dict kept the
           # relocation ghosts (STL/SD/OAK, parked near 1500 by the preseason revert)
           # and dropped the three worst real teams instead.
           "games": games, "ratings": ratings,
           "edges": edges, "edge_note": edge_note,
           "odds_status": ostat, "bettable": sorted(nfl_edge.BETTABLE),
           # where games.csv actually came from this run. A frozen tracked snapshot
           # (the old behaviour) is invisible from the outside — this makes it not.
           "data_status": dstat,
           "backtest": bt, "cal": cal}
    with open(os.path.join(DATA, "slate.json"), "w") as f:
        json.dump(_scrub(out), f, indent=1, allow_nan=False)
    print(f"slate.json written: {len(games)} games, {len(edges)} edges, "
          f"cal n={cal.get('n',0)}")

if __name__ == "__main__":
    main()
