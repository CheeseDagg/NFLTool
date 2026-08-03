"""
nfl_edge.py — line-shopping edges: the ONLY validated betting signal here.
Per-book devig -> median no-vig consensus per game -> best available price per
side -> quarter-Kelly on the gap. The model's win% is DELIBERATELY not used:
in 693 measured disagreements with the closing market it was right 44.3%.
"""
import os, sys, csv, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nfl_books import BETTABLE, is_bettable

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MIN_EDGE = 1.0          # percent
BANKROLL_FRAC_CAP = 0.02

def dec(ml):
    ml = int(ml)
    return ml / 100 + 1 if ml > 0 else 100 / (-ml) + 1

def am(p):
    p = min(max(p, 1e-6), 1 - 1e-6); d = 1 / p
    return f"+{round((d-1)*100)}" if d >= 2 else f"-{round(100/(d-1))}"

def find_edges(rows):
    """rows: dicts with home/away/book/home_ml/away_ml. -> edge rows."""
    games = {}
    for r in rows:
        games.setdefault((r["home"], r["away"]), []).append(r)
    out = []
    find_edges._no_bettable = 0
    for (h, a), bks in games.items():
        fair_h = []
        for b in bks:
            ih, ia = 1 / dec(b["home_ml"]), 1 / dec(b["away_ml"])
            fair_h.append(ih / (ih + ia))
        # Consensus pools EVERY book — it is an estimate of the true probability and
        # gets better the more books feed it. The QUOTE may only come from a book the
        # user can actually bet: EV is computed against the quoted decimal, so pricing
        # the edge off a book he has no account at produces an EV, a Kelly stake and a
        # board rank belonging to a bet that cannot be placed.
        cons_h = statistics.median(fair_h)
        bet_bks = [b for b in bks if is_bettable(b["book"])]
        if not bet_bks:
            find_edges._no_bettable += 1
            continue
        best = {"home": max(bet_bks, key=lambda b: dec(b["home_ml"])),
                "away": max(bet_bks, key=lambda b: dec(b["away_ml"]))}
        # the field's top number, reported alongside so the cost of the restriction is
        # visible — never used to price anything.
        any_best = {"home": max(bks, key=lambda b: dec(b["home_ml"])),
                    "away": max(bks, key=lambda b: dec(b["away_ml"]))}
        for side, p in (("home", cons_h), ("away", 1 - cons_h)):
            b = best[side]; price = b[f"{side}_ml"]
            ev = p * dec(price) - 1
            if ev * 100 >= MIN_EDGE and len(bks) >= 3:
                kelly = max(ev / (dec(price) - 1), 0) * 0.25
                ab = any_best[side]
                out.append({"matchup": f"{a} @ {h}",
                            "bet": (h if side == "home" else a) + " ML",
                            "price": int(price), "book": b["book"],
                            "fair": am(p), "edge_pct": round(ev * 100, 1),
                            "stake_frac": round(min(kelly, BANKROLL_FRAC_CAP), 4),
                            "books_n": len(bks), "bettable_n": len(bet_bks),
                            "any_price": (int(ab[f"{side}_ml"])
                                          if dec(ab[f"{side}_ml"]) > dec(price) else None),
                            "any_book": (ab["book"]
                                         if dec(ab[f"{side}_ml"]) > dec(price) else None)})
    out.sort(key=lambda r: -r["edge_pct"])
    return out

def main():
    path = os.path.join(DATA, "nfl_odds.csv")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    edges = find_edges(rows) if rows else []
    print(f"edges: {len(edges)} sides clear {MIN_EDGE}%+ across {len(rows)} book-lines "
          f"(quoted at {'/'.join(sorted(BETTABLE))} only; "
          f"{getattr(find_edges, '_no_bettable', 0)} games dropped for no bettable book)")
    return edges

def selftest():
    assert BETTABLE <= {"".join(c for c in b.lower() if c.isalnum()) for b in BETTABLE}
    rows = [
        {"home":"KC","away":"BUF","book":"draftkings","home_ml":"-120","away_ml":"+100"},
        {"home":"KC","away":"BUF","book":"betmgm","home_ml":"-125","away_ml":"+105"},
        {"home":"KC","away":"BUF","book":"fanduel","home_ml":"-105","away_ml":"+102"},  # soft home price
    ]
    e = find_edges(rows)
    assert e and e[0]["bet"] == "KC ML" and e[0]["book"] == "fanduel", e
    assert e[0]["price"] == -105 and e[0]["edge_pct"] >= 1.0
    assert 0 < e[0]["stake_frac"] <= BANKROLL_FRAC_CAP
    assert e[0]["any_price"] is None, "nothing beats the quote here, so nothing to disclose"
    assert find_edges(rows[:2]) == []          # <3 books -> no edge calls

    # THE RESTRICTION. Swap the books so the soft -105 lives at DraftKings and FanDuel
    # holds the -120: the edge must be priced off the -120, which kills it, and the
    # -105 must NOT be quoted just because it is the best number in the feed.
    rows2 = [dict(r) for r in rows]
    rows2[0]["book"], rows2[2]["book"] = "fanduel", "draftkings"
    e2 = find_edges(rows2)
    assert e2 == [], "the -105 lives at DraftKings now; there is no edge left to book"

    # ...and where a bettable edge DOES exist, the field's top number is disclosed
    # beside it without ever being priced off. FanDuel +100 against a 50.65%
    # consensus is +1.3%; DraftKings' outlier +120 would read +11.4%.
    rows3 = [
        {"home":"KC","away":"BUF","book":"betmgm",     "home_ml":"-115","away_ml":"-105"},
        {"home":"KC","away":"BUF","book":"caesars",    "home_ml":"-118","away_ml":"-102"},
        {"home":"KC","away":"BUF","book":"pointsbet",  "home_ml":"-113","away_ml":"-107"},
        {"home":"KC","away":"BUF","book":"fanduel",    "home_ml":"+100","away_ml":"-115"},
        {"home":"KC","away":"BUF","book":"draftkings", "home_ml":"+120","away_ml":"-160"},
    ]
    e3 = find_edges(rows3)
    kc = [x for x in e3 if x["bet"] == "KC ML"]
    assert kc and kc[0]["book"] == "fanduel" and kc[0]["price"] == 100, kc
    assert kc[0]["any_price"] == 120 and kc[0]["any_book"] == "draftkings", kc
    assert kc[0]["books_n"] == 5 and kc[0]["bettable_n"] == 1, kc
    assert abs(kc[0]["edge_pct"] - 1.3) < 0.3, \
        f"edge must be off +100 (~1.3%), not off the +120 nobody can take (~11.4%): {kc[0]}"

    # a game with no bettable book at all is DROPPED and COUNTED, not silently
    # repriced off someone else's number
    rows4 = [dict(r, book=b) for r, b in zip(rows, ("draftkings", "betmgm", "caesars"))]
    assert find_edges(rows4) == []
    assert find_edges._no_bettable == 1, find_edges._no_bettable
    print(f"EDGE SELFTEST PASS — consensus pools every book, quotes restricted to "
          f"{'/'.join(sorted(BETTABLE))}, kelly/min-books exact")
    return 0


if __name__ == "__main__":
    # defined last on purpose: selftest() must exist before this runs
    sys.exit(selftest()) if "--selftest" in sys.argv else main()
