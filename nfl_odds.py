"""
nfl_odds.py — closing-ish NFL moneylines from The Odds API -> data/nfl_odds.csv
Fail-soft by design: offseason returns zero events and that is a normal state.
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
    key = os.environ.get("ODDS_API_KEY") or "2aa2e57832d4c9ca4bd66b20b05ba448"
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
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "nfl_odds.csv")
    try:
        rows = rows_from_events(fetch())
    except Exception as e:
        print(f"odds pull failed ({type(e).__name__}) — writing empty file (fail-soft)")
        rows = []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["commence", "home", "away", "book", "home_ml", "away_ml"])
        w.writeheader()
        for r in rows: w.writerow(r)
    games = len({(r['home'], r['away'], r['commence']) for r in rows})
    print(f"nfl_odds.csv: {len(rows)} book-lines across {games} games"
          + (" (offseason: none priced yet is normal)" if not rows else ""))

if __name__ == "__main__":
    main()
