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
# date       = first time this game was published
# last_date  = last time it was republished (i.e. the number being graded)
# p_home     = LATEST published model prob;  first_p_home = the T-30 one
# mkt_p      = LATEST market consensus;      first_mkt_p  = the T-30 one (usually blank)
PCOLS = ["date","last_date","season","week","home","away",
         "p_home","first_p_home","mkt_p","first_mkt_p"]
GCOLS = PCOLS + ["outcome"]

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
        # Like for like: panel["acc"] above is over EVERY graded game, market["acc"]
        # can only be over the ones that carried a closing line. model_acc is the
        # model's rate on that same subset, so the two figures describe one game set.
        pacc = sum(1 for pm, _m, yy in M if (pm > 0.5) == (yy == 1)) / mn
        dis = [(pm, mm, yy) for pm, mm, yy in M if (pm > 0.5) != (mm > 0.5)]
        panel["market"] = {"n": mn, "acc": round(100 * macc, 1),
                           "model_acc": round(100 * pacc, 1),
                           "disagree_n": len(dis),
                           "disagree_model_right": (round(100 * sum(
                               1 for pm, _m, yy in dis if (pm > 0.5) == (yy == 1)) / len(dis), 1)
                               if dis else None),
                           # The number this panel never carried: how often the MARKET
                           # was right on exactly the games where the model claimed to
                           # know better. Ties are excluded from `live` above, so on
                           # this two-way market it is 100 minus the model's rate --
                           # but leaving a reader to do that subtraction is why the
                           # backtest's 44.3% read as mediocre for a season instead of
                           # as "the price wins 55.7% of the head-to-heads". Counted,
                           # not derived, so it stays correct if ties ever come back in.
                           "disagree_market_right": (round(100 * sum(
                               1 for _p, mm, yy in dis if (mm > 0.5) == (yy == 1)) / len(dis), 1)
                               if dis else None)}
    # FIRST SIGHT vs FINAL. The ledger used to freeze p_home at T-30 and grade that,
    # which is a different model from the one on the board. Publishing both numbers
    # keeps the claim falsifiable instead of resting on a comment.
    F = [(float(r["first_p_home"]) / 100, float(r["p_home"]) / 100,
          1.0 if r["outcome"] == "home" else 0.0)
         for r in live if str(r.get("first_p_home", "")).strip() not in ("", "None")]
    if F and any(abs(a - b) > 1e-9 for a, b, _ in F):
        fn = len(F)
        panel["first_vs_final"] = {
            "n": fn,
            "first_acc": round(100 * sum(1 for a, _b, yy in F if (a > 0.5) == (yy == 1)) / fn, 2),
            "final_acc": round(100 * sum(1 for _a, b, yy in F if (b > 0.5) == (yy == 1)) / fn, 2),
            "first_brier": round(sum((a - yy) ** 2 for a, _b, yy in F) / fn, 4),
            "final_brier": round(sum((b - yy) ** 2 for _a, b, yy in F) / fn, 4),
            "n_moved": sum(1 for a, b, _ in F if abs(a - b) > 1e-9)}
    return panel

def log_predictions(games_df, mkt_lookup=None):
    """One row per scheduled game, REFRESHED on every run until it is graded.

    This used to append-and-dedup: the first time a game entered the publisher's
    30-day window its p_home was written and never touched again. Two things
    followed, both bad.

      1. The ledger graded a T-30 forecast while the dashboard showed today's. A
         month of Elo updates -- every result since, the adaptive HFA step, the rest
         differential, which is not even knowable at T-30 -- separates the two. On
         the logged history the game-day number scores 64.90% and the frozen T-30
         number 63.20%, so the site was reporting itself 1.7 points worse than the
         thing it actually publishes. Grading a number you did not show is not
         conservatism, it is measuring a different model.

      2. mkt_p was captured at first sight too, and at T-30 there are no moneylines,
         so it was written as "" and never backfilled. summarize()'s market block
         needs mkt_p, so the entire market-disagreement study -- the one piece of
         evidence for the standing rule that the model is NOT a betting signal --
         rendered empty forever.

    Both are the same fix: rewrite the pending rows each run with the currently
    published number, and keep the first-sight number beside it as first_p_home /
    first_mkt_p so the T-30-vs-final question stays answerable (summarize reports
    it). Graded rows are never revisited -- grade_all skips anything already in
    GRADED, and games_df only ever contains unplayed games.

    Returns the number of games seen for the first time (the old return value).
    """
    os.makedirs(DATA, exist_ok=True)
    rows = load_csv(PLOG)
    idx = {(r["season"], r["week"], r["home"], r["away"]): r for r in rows}
    today = dt.date.today().isoformat()
    new = 0
    for r in games_df.itertuples():
        k = (str(r.season), str(r.week), str(r.home), str(r.away))
        mkt = ""
        if mkt_lookup:
            v = mkt_lookup.get((r.home, r.away))
            mkt = "" if v is None else str(v)
        cur = idx.get(k)
        if cur is None:
            rec = {"date": today, "last_date": today,
                   "season": str(r.season), "week": str(r.week),
                   "home": str(r.home), "away": str(r.away),
                   "p_home": str(r.p_home), "first_p_home": str(r.p_home),
                   "mkt_p": mkt, "first_mkt_p": mkt}
            idx[k] = rec; rows.append(rec); new += 1
        else:
            # rows written before this change have no first_* columns; the value
            # they carry IS the first-sight one, so seed from it rather than
            # backdating today's number into the historical slot.
            if not cur.get("first_p_home"):
                cur["first_p_home"] = cur.get("p_home", "")
            if not cur.get("first_mkt_p"):
                cur["first_mkt_p"] = cur.get("mkt_p", "")
            cur["p_home"] = str(r.p_home)
            cur["last_date"] = today
            if mkt != "":
                cur["mkt_p"] = mkt
            cur.setdefault("date", today)
    tmp = PLOG + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PCOLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in PCOLS})
    os.replace(tmp, PLOG)      # atomic: a crash mid-write must not truncate the log
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
        rec = {c: r.get(c, "") for c in PCOLS}; rec["outcome"] = o
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
    # BOTH HALVES OF THE DISAGREEMENT, AND ONE GAME SET. The panel used to publish only
    # the model's rate on its disagreements with the price, and to sit market["acc"]
    # (priced subset) next to panel["acc"] (every graded game). On the one disagreement
    # here the model said home, the market said away, and away won: model 0%, market 100%.
    assert p["market"]["disagree_market_right"] == 100.0, p["market"]
    # ties are dropped from `live`, so on this two-way market the two DO sum to 100 --
    # asserted so that a future change to tie handling has to come back through here.
    assert (p["market"]["disagree_model_right"]
            + p["market"]["disagree_market_right"]) == 100.0, p["market"]
    # model_acc is over the 2 priced games, same as market["acc"]; here it equals the
    # overall acc only because every graded game happened to carry a line.
    assert p["market"]["model_acc"] == 50.0, p["market"]
    json.dumps(p)

    # ---- the refresh: T-30 number must not be what gets graded ----
    global DATA, PLOG, GRADED
    import tempfile, types
    _D, _P, _G = DATA, PLOG, GRADED
    try:
        DATA = tempfile.mkdtemp(prefix="nflgrade_")
        PLOG = os.path.join(DATA, "nfl_predictions.csv")
        GRADED = os.path.join(DATA, "nfl_graded.csv")

        class _DF:                       # minimal stand-in for a games DataFrame
            def __init__(self, recs): self._r = recs
            def itertuples(self):
                return [types.SimpleNamespace(**r) for r in self._r]

        g1 = _DF([{"season": 2026, "week": 1, "home": "KC", "away": "BUF", "p_home": 58.0}])
        assert log_predictions(g1, mkt_lookup={}) == 1
        r0 = load_csv(PLOG)[0]
        assert r0["p_home"] == "58.0" and r0["first_p_home"] == "58.0"
        assert r0["mkt_p"] == "" and r0["first_mkt_p"] == ""

        # a week later: Elo has moved and the book has finally priced it
        g2 = _DF([{"season": 2026, "week": 1, "home": "KC", "away": "BUF", "p_home": 66.0}])
        assert log_predictions(g2, mkt_lookup={("KC", "BUF"): 63.5}) == 0, "must not duplicate"
        rows_ = load_csv(PLOG)
        assert len(rows_) == 1, rows_
        r1 = rows_[0]
        assert r1["p_home"] == "66.0", "the LATEST published number is what gets graded"
        assert r1["first_p_home"] == "58.0", "the T-30 number must survive for comparison"
        assert r1["mkt_p"] == "63.5", "mkt_p must backfill once odds exist — it never did"
        assert r1["first_mkt_p"] == "", "and the blank first sighting is preserved as blank"
        assert r1["date"] != "" and r1["last_date"] != ""

        # grading picks up the refreshed number, and first_vs_final reports both
        n_new, panel = grade_all({(2026, 1, "KC", "BUF"): 10})
        assert n_new == 1, n_new
        gr = load_csv(GRADED)[0]
        assert gr["p_home"] == "66.0" and gr["first_p_home"] == "58.0" and gr["outcome"] == "home"
        assert panel["first_vs_final"]["n"] == 1
        assert panel["first_vs_final"]["first_brier"] > panel["first_vs_final"]["final_brier"], \
            "KC won; the 66% call must beat the 58% one"
        assert panel["market"]["n"] == 1, "the disagreement study is live again"
        # already graded -> a later run must not regrade or re-append
        assert grade_all({(2026, 1, "KC", "BUF"): 10})[0] == 0
        json.dumps(panel)
    finally:
        DATA, PLOG, GRADED = _D, _P, _G

    print("NFL GRADER SELFTEST PASS — settle/tie/pending, Brier/buckets/market-disagree, "
          "refresh-until-graded + mkt backfill + first-vs-final")
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv: sys.exit(selftest())
    print("run via nfl_publish.py")
