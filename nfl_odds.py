"""
nfl_odds.py — closing-ish NFL moneylines from The Odds API -> data/nfl_odds.csv
Zero events from a HEALTHY call is a normal state (offseason). A failed call is
not, and is never allowed to look like one — see main().
"""
import os, json, csv, urllib.request, urllib.parse, datetime as dt

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Odds API full names <-> nflverse abbreviations
TEAMS = {
 "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
 "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
 "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
 "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
 "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LA","Miami Dolphins":"MIA",
 "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
 "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF",
 "Seattle Seahawks":"SEA","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS",
}

def fetch():
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("ODDS_API_KEY secret not set")
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us",
                                "markets": "h2h", "oddsFormat": "american"})
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NFLTool)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def rows_from_events(events):
    out = []
    for ev in events or []:
        h, a = TEAMS.get(ev.get("home_team", "")), TEAMS.get(ev.get("away_team", ""))
        if not h or not a: continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h": continue
                ph = pa = None
                for o in mk.get("outcomes", []):
                    if TEAMS.get(o.get("name", "")) == h: ph = o.get("price")
                    if TEAMS.get(o.get("name", "")) == a: pa = o.get("price")
                if ph is not None and pa is not None:
                    out.append({"commence": ev.get("commence_time", ""),
                                "home": h, "away": a, "book": bk.get("key", "?"),
                                "home_ml": int(ph), "away_ml": int(pa)})
    return out

def main():
    """Two zero-row states exist and they are NOT the same thing.

      A. the API answered and returned no events  -> nothing is priced. Real, normal
         in the offseason, and worth writing: it retires yesterday's stale lines.
      B. the API did not answer                   -> we know nothing.

    The old code collapsed both into "writing empty file (fail-soft)" and then
    printed "(offseason: none priced yet is normal)". That did two bad things at
    once: it OVERWROTE the last good nfl_odds.csv with an empty one, destroying data
    the publisher could still have used, and it labelled an auth failure, a quota
    exhaustion or a timeout as a season fact. A quiet board and a broken key looked
    identical from the dashboard, so a broken key could sit there for weeks.

    Now B keeps the existing file and says so. Both states are recorded in
    data/nfl_odds_status.json so downstream (and the dashboard) can tell them apart
    instead of inferring from an empty CSV.
    """
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "nfl_odds.csv")
    spath = os.path.join(DATA, "nfl_odds_status.json")
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        rows = rows_from_events(fetch())
    except Exception as e:
        prev = 0
        if os.path.exists(path):
            with open(path) as f:
                prev = max(sum(1 for _ in f) - 1, 0)
        st = {"checked": ts, "ok": False, "error": f"{type(e).__name__}: {e}",
              "rows": None, "games": None, "kept_previous_rows": prev,
              "note": "odds pull FAILED — the previous nfl_odds.csv was kept, not "
                      "overwritten. This is NOT an offseason signal."}
        json.dump(st, open(spath, "w"), indent=1)
        print(f"odds pull FAILED ({type(e).__name__}) — kept the existing "
              f"nfl_odds.csv ({prev} rows). This is not 'offseason'.")
        return 1
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["commence", "home", "away", "book", "home_ml", "away_ml"])
        w.writeheader()
        for r in rows: w.writerow(r)
    games = len({(r['home'], r['away'], r['commence']) for r in rows})
    json.dump({"checked": ts, "ok": True, "error": None,
               "rows": len(rows), "games": games, "kept_previous_rows": None,
               "note": ("API answered with zero priced events — nothing is on the "
                        "board right now" if not rows else "")},
              open(spath, "w"), indent=1)
    print(f"nfl_odds.csv: {len(rows)} book-lines across {games} games"
          + (" (API answered, zero events priced — genuinely nothing up)" if not rows else ""))
    return 0

if __name__ == "__main__":
    main()
