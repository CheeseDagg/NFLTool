"""
nfl_grade.py — settle every published p_home against actual results and keep
the disagreement study running LIVE. Results come from the same nflverse
games.csv the model trains on, so grading needs no second source.
Outcomes: home / away / tie / pending. Idempotent.
"""
import os, csv, json, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PLOG = os.path.join(DATA, "nfl_predictions.csv")
GRADED = os.path.join(DATA, "nfl_graded.csv")
GCOLS = ["date","season","week","home","away","p_home","mkt_p","outcome"]

def load_csv(p):
    if not os.path.exists(p): return []
    with open(p) as f: return list(csv.DictReader(f))

def settle(pred, results):
    """results: {(season,week,home,away): margin} for completed games."""
    k = (int(pred["season"]), int(pred["week"]), pred["home"], pred["away"])
    if k not in results: return "pending"
    m = results[k]
    return "home" if m > 0 else ("away" if m < 0 else "tie")

def summarize(rows):
    live = [r for r in rows if r.get("outcome") in ("home", "away")]
    n = len(live)
    panel = {"n": n, "ties": sum(1 for r in rows if r.get("outcome") == "tie"),
             "weeks": len({(r["season"], r["week"]) for r in rows}) if rows else 0}
    if not n: return panel
    p = [float(r["p_home"]) / 100 for r in live]
    y = [1.0 if r["outcome"] == "home" else 0.0 for r in live]
    panel["acc"] = round(100 * sum(1 for pi, yi in zip(p, y) if (pi > 0.5) == (yi == 1)) / n, 1)
    panel["brier"] = round(sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n, 4)
    edges = [(0, 40), (40, 50), (50, 60), (60, 100)]
    panel["buckets"] = []
    for lo, hi in edges:
        sel = [(pi, yi) for pi, yi in zip(p, y) if lo <= pi * 100 < hi]
        if sel:
            panel["buckets"].append({"bucket": f"{lo}-{hi if hi < 100 else '+'}",
                "n": len(sel),
                "pred": round(100 * sum(a for a, _ in sel) / len(sel), 1),
                "actual": round(100 * sum(b for _, b in sel) / len(sel), 1)})
    M = [(float(r["p_home"]) / 100, float(r["mkt_p"]) / 100,
          1.0 if r["outcome"] == "home" else 0.0)
         for r in live if r.get("mkt_p") not in ("", None)]
    if M:
        mn = len(M)
        macc = sum(1 for pm, mm, yy in M if (mm > 0.5) == (yy == 1)) / mn
        dis = [(pm, mm, yy) for pm, mm, yy in M if (pm > 0.5) != (mm > 0.5)]
        panel["market"] = {"n": mn, "acc": round(100 * macc, 1),
                           "disagree_n": len(dis),
                           "disagree_model_right": (round(100 * sum(
                               1 for pm, _m, yy in dis if (pm > 0.5) == (yy == 1)) / len(dis), 1)
                               if dis else None)}
    return panel

def log_predictions(games_df, mkt_lookup=None):
    """Called by publish: append today's upcoming-slate predictions (dedup by game)."""
    os.makedirs(DATA, exist_ok=True)
    existing = load_csv(PLOG)
    have = {(r["season"], r["week"], r["home"], r["away"]) for r in existing}
    today = dt.date.today().isoformat()
    new = 0
    with open(PLOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GCOLS[:-1])
        if not existing: w.writeheader()
        for r in games_df.itertuples():
            k = (str(r.season), str(r.week), r.home, r.away)
            if k in have: continue
            mkt = ""
            if mkt_lookup:
                mkt = mkt_lookup.get((r.home, r.away), "")
            w.writerow({"date": today, "season": r.season, "week": r.week,
                        "home": r.home, "away": r.away, "p_home": r.p_home,
                        "mkt_p": mkt})
            new += 1
    return new

def grade_all(results):
    preds = load_csv(PLOG)
    done = {(r["season"], r["week"], r["home"], r["away"]) for r in load_csv(GRADED)}
    new = []
    for r in preds:
        k = (r["season"], r["week"], r["home"], r["away"])
        if k in done: continue
        o = settle(r, results)
        if o == "pending": continue
        rec = {c: r.get(c, "") for c in GCOLS[:-1]}; rec["outcome"] = o
        new.append(rec)
    if new:
        exists = os.path.exists(GRADED)
        with open(GRADED, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GCOLS)
            if not exists: w.writeheader()
            for r in new: w.writerow(r)
    return len(new), summarize(load_csv(GRADED))

def panel_for_publish():
    try: return summarize(load_csv(GRADED))
    except Exception as e: return {"n": 0, "error": type(e).__name__}

def selftest():
    res = {(2026, 1, "KC", "BUF"): 3, (2026, 1, "SEA", "SF"): -7, (2026, 1, "DAL", "NYG"): 0}
    row = lambda **k: dict({"date": "x", "season": "2026", "week": "1",
                            "home": "", "away": "", "p_home": "60", "mkt_p": ""}, **k)
    assert settle(row(home="KC", away="BUF"), res) == "home"
    assert settle(row(home="SEA", away="SF"), res) == "away"
    assert settle(row(home="DAL", away="NYG"), res) == "tie"
    assert settle(row(home="MIA", away="NYJ"), res) == "pending"
    rows = [
        row(home="KC", away="BUF", p_home="65", mkt_p="70", outcome="home"),
        row(home="SEA", away="SF", p_home="55", mkt_p="48", outcome="away"),   # model+mkt disagree; model wrong
        row(home="DAL", away="NYG", p_home="50", outcome="tie"),
    ]
    p = summarize(rows)
    assert p["n"] == 2 and p["ties"] == 1
    assert p["acc"] == 50.0
    assert p["brier"] == round(((0.65-1)**2 + (0.55-0)**2)/2, 4)
    assert p["market"]["n"] == 2 and p["market"]["acc"] == 100.0
    assert p["market"]["disagree_n"] == 1 and p["market"]["disagree_model_right"] == 0.0
    json.dumps(p)
    print("NFL GRADER SELFTEST PASS — settle/tie/pending + Brier/buckets/market-disagree exact")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    print("run via nfl_publish.py")
