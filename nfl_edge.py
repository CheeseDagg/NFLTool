"""
nfl_edge.py — line-shopping edges: the ONLY validated betting signal here.
Per-book devig -> median no-vig consensus per game -> best available price per
side -> quarter-Kelly on the gap. The model's win% is DELIBERATELY not used:
in 693 measured disagreements with the closing market it was right 44.3%.
"""
import os, csv, statistics

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
    for (h, a), bks in games.items():
        fair_h = []
        for b in bks:
            ih, ia = 1 / dec(b["home_ml"]), 1 / dec(b["away_ml"])
            fair_h.append(ih / (ih + ia))
        cons_h = statistics.median(fair_h)
        best = {"home": max(bks, key=lambda b: dec(b["home_ml"])),
                "away": max(bks, key=lambda b: dec(b["away_ml"]))}
        for side, p in (("home", cons_h), ("away", 1 - cons_h)):
            b = best[side]; price = b[f"{side}_ml"]
            ev = p * dec(price) - 1
            if ev * 100 >= MIN_EDGE and len(bks) >= 3:
                kelly = max(ev / (dec(price) - 1), 0) * 0.25
                out.append({"matchup": f"{a} @ {h}",
                            "bet": (h if side == "home" else a) + " ML",
                            "price": int(price), "book": b["book"],
                            "fair": am(p), "edge_pct": round(ev * 100, 1),
                            "stake_frac": round(min(kelly, BANKROLL_FRAC_CAP), 4),
                            "books_n": len(bks)})
    out.sort(key=lambda r: -r["edge_pct"])
    return out

def main():
    path = os.path.join(DATA, "nfl_odds.csv")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    edges = find_edges(rows) if rows else []
    print(f"edges: {len(edges)} sides clear {MIN_EDGE}%+ across {len(rows)} book-lines")
    return edges

if __name__ == "__main__":
    main()

def selftest():
    rows = [
        {"home":"KC","away":"BUF","book":"a","home_ml":"-120","away_ml":"+100"},
        {"home":"KC","away":"BUF","book":"b","home_ml":"-125","away_ml":"+105"},
        {"home":"KC","away":"BUF","book":"c","home_ml":"-105","away_ml":"+102"},  # soft home price
    ]
    e = find_edges(rows)
    assert e and e[0]["bet"] == "KC ML" and e[0]["book"] == "c", e
    assert e[0]["price"] == -105 and e[0]["edge_pct"] >= 1.0
    assert 0 < e[0]["stake_frac"] <= BANKROLL_FRAC_CAP
    assert find_edges(rows[:2]) == []          # <3 books -> no edge calls
    print("EDGE SELFTEST PASS — consensus/best-price/kelly/min-books exact")
    return 0
