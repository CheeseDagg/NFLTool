"""Robustness check for the QB-adjusted Elo before shipping: per-season holdout, and the
market-disagreement accuracy (the metric the production backtest flags at 44.3%)."""
import math
import pandas as pd, numpy as np
HERE="/root/NFLTool"
K,HFA_PTS,REVERT,SCALE=20.0,48.0,0.33,400.0; REST_PER_DAY=4.0
def expected(dr): return 1.0/(1.0+10**(-dr/SCALE))
def _dec(ml): return ml/100+1 if ml>0 else 100/(-ml)+1
def load():
    g=pd.read_csv(f"{HERE}/games.csv"); g=g[g["game_type"].isin(["REG","WC","DIV","CON","SB"])].copy()
    g["gameday"]=pd.to_datetime(g["gameday"],errors="coerce")
    return g.sort_values(["season","week","gameday"]).reset_index(drop=True)
def run(g, qb_w=0.0, qb_k=0.0, qb_decay=0.03, prior=0.0, start=2010):
    R,Q={},{}; cur=None; out=[]
    for r in g.itertuples():
        if r.season!=cur:
            cur=r.season
            for t in R: R[t]=1500+(R[t]-1500)*(1-REVERT)
        h,a=r.home_team,r.away_team; R.setdefault(h,1500); R.setdefault(a,1500)
        rest=REST_PER_DAY*((r.home_rest-7)-(r.away_rest-7)) if pd.notna(r.home_rest) and pd.notna(r.away_rest) else 0.0
        hfa=0.0 if str(r.location)=="Neutral" else HFA_PTS
        base_dr=(R[h]+hfa+rest)-R[a]; p_base=expected(base_dr)
        hq=r.home_qb_id if isinstance(r.home_qb_id,str) else None
        aq=r.away_qb_id if isinstance(r.away_qb_id,str) else None
        qh=Q.get(hq,prior) if hq else 0.0; qa=Q.get(aq,prior) if aq else 0.0
        p=expected(base_dr+qb_w*(qh-qa))
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin=r.home_score-r.away_score
            if r.season>=start:
                mp=np.nan
                if pd.notna(r.home_moneyline) and pd.notna(r.away_moneyline):
                    ih,ia=1/_dec(r.home_moneyline),1/_dec(r.away_moneyline); mp=ih/(ih+ia)
                out.append({"season":r.season,"p":p,"y":int(margin>0),"tie":int(margin==0),"mkt":mp})
            mov=math.log(abs(margin)+1)*(2.2/((0.001*abs(base_dr) if margin*base_dr>0 else -0.001*abs(base_dr))+2.2))
            s=1.0 if margin>0 else (0.5 if margin==0 else 0.0)
            d=K*mov*(s-p_base); R[h]+=d; R[a]-=d
            if qb_k and hq and aq:
                resid=s-p_base
                Q[hq]=Q.get(hq,prior)*(1-qb_decay)+qb_k*resid
                Q[aq]=Q.get(aq,prior)*(1-qb_decay)-qb_k*resid
    return pd.DataFrame(out)
def metrics(P):
    P=P[P["tie"]==0]
    acc=((P["p"]>0.5).astype(int)==P["y"]).mean(); brier=((P["p"]-P["y"])**2).mean()
    M=P.dropna(subset=["mkt"]); dis=M[((M["p"]>0.5)!=(M["mkt"]>0.5))]
    dacc=((dis["p"]>0.5).astype(int)==dis["y"]).mean() if len(dis) else np.nan
    return len(P),round(100*acc,2),round(brier,5),len(dis),round(100*dacc,2) if len(dis) else None

g=load()
base=run(g); qb=run(g,qb_w=0.6,qb_k=20.0,prior=-80.0)
print("            n     acc    brier   disagree  model-right-on-disagree")
for name,P in (("baseline",base),("qb-elo ",qb)):
    n,a,b,dn,da=metrics(P); print(f"{name}  {n:5}  {a:6}  {b:.5f}   {dn:5}    {da}%")
print()
print("Per-season HOLDOUT Brier (2020-2025), baseline vs qb-elo (neg delta = qb better):")
wins=0; tot=0
for yr in range(2020,2026):
    _,_,bb,_,_=metrics(base[base["season"]==yr]); _,_,qbb,_,_=metrics(qb[qb["season"]==yr])
    d=round(qbb-bb,5); tot+=1; wins+= (d<0)
    print(f"  {yr}: baseline {bb:.5f}  qb {qbb:.5f}  delta {d:+.5f}  {'QB better' if d<0 else 'baseline better'}")
print(f"  -> QB better in {wins}/{tot} holdout seasons")
