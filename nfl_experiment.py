"""NFL accuracy experiment: does a QB-adjusted Elo beat the team-only baseline
OUT OF SAMPLE? Baseline team Elo is kept byte-identical (R updates on the no-QB
expectation), and a per-QB residual rating is layered on top of the PREDICTION only.
Leak-free: every prediction is made before that game's update. Train seasons tune the
weight; holdout seasons are the honest verdict.

AVAILABILITY. "Leak-free" here means time-ordered, and that is NOT the same as
computable. This harness reads home_qb_id/away_qb_id — the QB who ACTUALLY
STARTED — which nflverse populates on every played game and on none of the
unplayed ones nfl_model.state() predicts. So the holdout numbers below are
honest about the past and unusable for the future. The gate imported from
nfl_qb_experiment prints that in the header instead of leaving it for a reader
to notice; see the GATE 0 section of that file for why the check is on raw
columns rather than on the feature's output.
"""
import math, sys
import pandas as pd, numpy as np

import nfl_qb_experiment as QBX

HERE = "/root/NFLTool"
K, HFA_PTS, REVERT, SCALE = 20.0, 48.0, 0.33, 400.0
REST_PER_DAY = 4.0
def expected(dr): return 1.0 / (1.0 + 10 ** (-dr / SCALE))

def load():
    g = pd.read_csv(f"{HERE}/games.csv")
    g = g[g["game_type"].isin(["REG","WC","DIV","CON","SB"])].copy()
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    return g.sort_values(["season","week","gameday"]).reset_index(drop=True)

def run(g, qb_w=0.0, qb_k=0.0, qb_decay=0.0, new_qb_prior=0.0, start_season=2010):
    """qb_w=0 -> pure baseline. qb_w>0 -> QB residual rating added to the prediction.
    Team Elo R updates on the NO-QB expectation, so the baseline is preserved exactly and
    any accuracy delta is attributable to the QB layer alone."""
    R, Q = {}, {}
    cur = None; preds = []
    for r in g.itertuples():
        if r.season != cur:
            cur = r.season
            for t in R: R[t] = 1500 + (R[t]-1500)*(1-REVERT)
        h, a = r.home_team, r.away_team
        R.setdefault(h,1500); R.setdefault(a,1500)
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = REST_PER_DAY*((r.home_rest-7)-(r.away_rest-7))
        hfa = 0.0 if str(r.location)=="Neutral" else HFA_PTS
        base_dr = (R[h]+hfa+rest) - R[a]
        p_base = expected(base_dr)
        hq = r.home_qb_id if isinstance(r.home_qb_id,str) else None
        aq = r.away_qb_id if isinstance(r.away_qb_id,str) else None
        qh = Q.get(hq, new_qb_prior) if hq else 0.0
        qa = Q.get(aq, new_qb_prior) if aq else 0.0
        full_dr = base_dr + qb_w*(qh - qa)
        p_full = expected(full_dr)
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin = r.home_score - r.away_score
            if r.season >= start_season:
                preds.append({"season": r.season, "p": p_full, "y": int(margin>0),
                              "tie": int(margin==0)})
            # team Elo update — identical to production (uses p_base, no QB)
            mov = math.log(abs(margin)+1) * (2.2/((0.001*abs(base_dr) if margin*base_dr>0 else -0.001*abs(base_dr))+2.2))
            s = 1.0 if margin>0 else (0.5 if margin==0 else 0.0)
            d = K*mov*(s - p_base)
            R[h]+=d; R[a]-=d
            # QB residual update: attribute the team-Elo surprise to the starters, symmetric,
            # with slow reversion so an established starter's residual stays near 0.
            if qb_k and hq and aq:
                resid = s - p_base
                Q[hq] = Q.get(hq,new_qb_prior)*(1-qb_decay) + qb_k*resid
                Q[aq] = Q.get(aq,new_qb_prior)*(1-qb_decay) - qb_k*resid
    return pd.DataFrame(preds)

def score(P):
    P = P[P["tie"]==0]
    if not len(P): return {}
    acc = ((P["p"]>0.5).astype(int)==P["y"]).mean()
    brier = ((P["p"]-P["y"])**2).mean()
    return {"n": len(P), "acc": round(100*acc,2), "brier": round(brier,5)}

def availability_gate(g):
    """GATE 0 for this harness. Runs the QB layer under the column spy and
    checks every raw column it touches against the unplayed slate.

    Traced rather than declared: a hand-written list of "columns this run()
    reads" would have to be updated by whoever next adds a term, and the failure
    mode of forgetting is a silent pass — which is exactly how a QB feature got
    to the edge of production once already.
    """
    QBX.run_availability_gate(
        lambda gg: run(gg, qb_w=1.0, qb_k=20.0, qb_decay=0.03),
        g, ["qb_layer"], label="— the QB-adjusted Elo layer", register=False)


if __name__ == "__main__":
    g = load()
    availability_gate(g)
    base = score(run(g))
    print(f"BASELINE (team-only Elo):  {base}")
    TRAIN, HOLD = range(2010,2020), range(2020,2026)
    def split_score(P):
        tr = score(P[P["season"].isin(TRAIN)]); ho = score(P[P["season"].isin(HOLD)])
        return tr, ho
    b_tr, b_ho = split_score(run(g))
    print(f"  baseline train {b_tr}  holdout {b_ho}")
    print()
    print("QB-adjusted Elo — grid on TRAIN, verdict on HOLDOUT (Brier; lower=better):")
    best = None
    for qb_w in (0.6, 1.0):
        for qb_k in (20.0, 40.0):
            for prior in (0.0, -40.0, -80.0):
                P = run(g, qb_w=qb_w, qb_k=qb_k, qb_decay=0.03, new_qb_prior=prior)
                tr, ho = split_score(P)
                tag = f"qb_w={qb_w} qb_k={qb_k} prior={prior}"
                print(f"  {tag:34} train brier {tr['brier']}  holdout brier {ho['brier']}  (base ho {b_ho['brier']})")
                if best is None or tr["brier"] < best[0]:
                    best = (tr["brier"], tag, tr, ho, qb_w, qb_k, prior)
    print()
    print(f"BEST-ON-TRAIN: {best[1]}")
    print(f"  train  {best[2]}")
    print(f"  HOLDOUT {best[3]}   vs baseline holdout {b_ho}")
    print(f"  holdout Brier delta: {round(best[3]['brier']-b_ho['brier'],5)} (negative = improvement)")
    print(f"  holdout acc delta:   {round(best[3]['acc']-b_ho['acc'],2)} pts")
