"""
nfl_epa_experiment.py — standalone "new data" experiment harness
================================================================
QUESTION: does nflverse play-by-play EPA (offense/defense efficiency, plus a
QB-specific EPA adjustment) make game win-probability predictions MORE ACCURATE
than the current team-Elo model (nfl_model.py)?

This is a STANDALONE experiment. It never imports/writes production state and it
does NOT modify nfl_model.py or any prediction file. It only reads games.csv and
(on a networked host) the nflverse pbp releases, then prints a VERDICT.

DESIGN (leak-free walk-forward, exactly the discipline of nfl_model / nfl_experiment):
  * BASELINE  = team-only Elo p_home, reimplemented byte-identically to nfl_model
                (verified against nfl_model.run_elo in --selftest).
  * TREATMENT = the SAME baseline Elo, with a prediction-only adjustment built
                from rolling EPA:  full_dr = base_dr + w_epa*epa_adj + w_qb*qb_adj.
                Team Elo R still updates on p_base (no EPA), so the baseline is
                preserved exactly and any delta is attributable to the EPA layer.
  * All EPA features use ONLY prior games: per team/QB, sort by date, then
                shift(1).rolling(window).mean(). The current game is never in its
                own feature (asserted in --selftest).
  * Blend weights are tuned on TRAIN seasons only; the honest verdict is on the
                HOLDOUT seasons. Per-season deltas are printed too.

DATA: nflverse pbp releases (EPA per play). Tries parquet then csv.gz per year.
  This cloud session's egress is BLOCKED to GitHub — the live pull returns 403
  'Tunnel connection failed'. That is EXPECTED here; the script degrades with a
  clear "PBP UNREACHABLE from here — run on Actions" message and exits 0. On
  GitHub Actions (where nflverse IS reachable) the full experiment runs.

RUN:
  python3 nfl_epa_experiment.py            # full experiment (needs network + pyarrow)
  python3 nfl_epa_experiment.py --selftest # OFFLINE, synthetic, no network — CI gate
"""
import os, sys, math, io, urllib.request, urllib.error, warnings
import numpy as np, pandas as pd

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---- Elo params: copied verbatim from nfl_model.py so BASELINE is identical ----
K, HFA_PTS, REVERT, SCALE = 20.0, 48.0, 0.33, 400.0
REST_PER_DAY = 4.0

# ---- experiment split ----
TRAIN_SEASONS = range(2010, 2020)   # tune blend weights here
HOLD_SEASONS  = range(2020, 2026)   # honest verdict here
START_SEASON  = 2010

# nflverse pbp release URLs (parquet preferred, csv.gz fallback)
PBP_PARQUET = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
PBP_CSVGZ   = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.csv.gz"

def expected(dr):
    return 1.0 / (1.0 + 10 ** (-dr / SCALE))

# =====================================================================
#  DATA LOADING
# =====================================================================
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"

def load_games():
    """The same games table the production model uses. Download it if absent (the
    Actions runner starts with a clean checkout and games.csv is not committed)."""
    path = os.path.join(HERE, "games.csv")
    if not os.path.exists(path):
        import urllib.request
        print(f"games.csv not present — downloading from nflverse ...")
        urllib.request.urlretrieve(GAMES_URL, path)
    g = pd.read_csv(path)
    g = g[g["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    g["gameday"] = pd.to_datetime(g["gameday"], errors="coerce")
    return g.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def _read_parquet_bytes(b):
    return pd.read_parquet(io.BytesIO(b))   # needs pyarrow


def _read_csvgz_bytes(b):
    return pd.read_csv(io.BytesIO(b), compression="gzip", low_memory=False)


# Only the columns we actually need — keeps memory sane if we do read pbp.
PBP_COLS = ["game_id", "season", "posteam", "defteam", "epa", "qb_epa",
            "passer_player_id", "pass", "rush", "play_type"]


def _fetch_one_year(year, cache=True):
    """Return a pbp DataFrame for one season, or None if unreachable.
    Tries parquet (needs pyarrow) then csv.gz. Caches raw bytes under data/."""
    os.makedirs(DATA, exist_ok=True)
    # cached parquet?
    for ext, reader in ((".parquet", _read_parquet_bytes), (".csv.gz", _read_csvgz_bytes)):
        cpath = os.path.join(DATA, f"pbp_{year}{ext}")
        if cache and os.path.exists(cpath):
            try:
                with open(cpath, "rb") as f:
                    return reader(f.read())
            except Exception as e:
                print(f"  [warn] cached {cpath} unreadable ({e}); refetching")
    # live pull
    attempts = [(PBP_PARQUET.format(y=year), ".parquet", _read_parquet_bytes),
                (PBP_CSVGZ.format(y=year),   ".csv.gz",  _read_csvgz_bytes)]
    last_err = None
    for url, ext, reader in attempts:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nfl-epa-experiment"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            df = reader(raw)                 # may raise if pyarrow missing for parquet
            if cache:
                try:
                    with open(os.path.join(DATA, f"pbp_{year}{ext}"), "wb") as f:
                        f.write(raw)
                except Exception:
                    pass
            return df
        except Exception as e:
            last_err = e
            continue
    print(f"  [year {year}] unreachable: {last_err}")
    return None


def fetch_pbp(years):
    """Fetch all seasons. Returns (concat_df_or_None, reachable_bool)."""
    frames, reachable = [], False
    for y in years:
        df = _fetch_one_year(y)
        if df is None:
            continue
        reachable = True
        keep = [c for c in PBP_COLS if c in df.columns]
        frames.append(df[keep].copy())
    if not frames:
        return None, reachable
    return pd.concat(frames, ignore_index=True), True


# =====================================================================
#  EPA AGGREGATION  (pbp -> per-team-per-game and per-QB-per-game)
# =====================================================================
def aggregate_epa(pbp):
    """From raw pbp build a long per-team-per-game table with offensive and
    defensive EPA/play, plus a per-QB-per-game EPA table.

    offensive EPA/play(team, game) = mean epa of plays where posteam==team
    defensive  EPA/play(team, game) = mean epa ALLOWED, i.e. plays where defteam==team
    qb epa/play(qb, game)           = mean qb_epa of that passer's dropbacks
    """
    p = pbp.copy()
    # restrict to real offensive plays with a valid epa
    if "play_type" in p.columns:
        p = p[p["play_type"].isin(["pass", "run"])] if p["play_type"].notna().any() else p
    elif {"pass", "rush"}.issubset(p.columns):
        p = p[(p["pass"] == 1) | (p["rush"] == 1)]
    p = p[p["epa"].notna()]

    off = (p[p["posteam"].notna()]
           .groupby(["game_id", "posteam"], as_index=False)["epa"].mean()
           .rename(columns={"posteam": "team", "epa": "off_epa"}))
    dfn = (p[p["defteam"].notna()]
           .groupby(["game_id", "defteam"], as_index=False)["epa"].mean()
           .rename(columns={"defteam": "team", "epa": "def_epa"}))
    team_game = off.merge(dfn, on=["game_id", "team"], how="outer")

    qb = pd.DataFrame(columns=["game_id", "qb_id", "qb_epa"])
    if "qb_epa" in p.columns and "passer_player_id" in p.columns:
        q = p[p["passer_player_id"].notna() & p["qb_epa"].notna()]
        qb = (q.groupby(["game_id", "passer_player_id"], as_index=False)["qb_epa"].mean()
                .rename(columns={"passer_player_id": "qb_id", "qb_epa": "qb_epa"}))
    return team_game, qb


# =====================================================================
#  LEAK-FREE ROLLING FEATURES  (shift(1) — current game never included)
# =====================================================================
def rolling_shift_mean(df, key, date_col, value_cols, window):
    """Per `key`, sort by date, and compute prior-only rolling means:
    shift(1) removes the current row, then .rolling(window).mean() averages the
    PRIOR games. Row 0 for each key is NaN (no history). This is the single
    leak-free primitive the whole experiment relies on (asserted in --selftest).
    """
    out = df.sort_values([key, date_col]).copy()
    for c in value_cols:
        out[c + "_roll"] = (out.groupby(key)[c]
                              .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean()))
    return out


def build_game_features(g, team_game, qb, window=10):
    """Attach pre-game rolling EPA features to each game as home_/away_ columns.
    Everything here is prior-only, so merging back onto the game is leak-free."""
    # need a date per (game_id, team) to order the rolling window
    long = team_game.merge(g[["game_id", "gameday", "season"]], on="game_id", how="left")
    long = rolling_shift_mean(long, "team", "gameday", ["off_epa", "def_epa"], window)
    # a team's pre-game net efficiency: its offense minus the epa it allows
    long["net_epa_roll"] = long["off_epa_roll"] - long["def_epa_roll"]
    feat = long[["game_id", "team", "net_epa_roll"]]

    home = feat.rename(columns={"team": "home_team", "net_epa_roll": "home_net_epa"})
    away = feat.rename(columns={"team": "away_team", "net_epa_roll": "away_net_epa"})
    out = g.merge(home, on=["game_id", "home_team"], how="left") \
           .merge(away, on=["game_id", "away_team"], how="left")

    # per-QB rolling epa (pre-game)
    out["home_qb_epa"] = np.nan
    out["away_qb_epa"] = np.nan
    if qb is not None and len(qb):
        ql = qb.merge(g[["game_id", "gameday"]], on="game_id", how="left")
        ql = rolling_shift_mean(ql, "qb_id", "gameday", ["qb_epa"], window)
        qfeat = ql[["game_id", "qb_id", "qb_epa_roll"]]
        hq = qfeat.rename(columns={"qb_id": "home_qb_id", "qb_epa_roll": "home_qb_epa_r"})
        aq = qfeat.rename(columns={"qb_id": "away_qb_id", "qb_epa_roll": "away_qb_epa_r"})
        out = out.merge(hq, on=["game_id", "home_qb_id"], how="left") \
                 .merge(aq, on=["game_id", "away_qb_id"], how="left")
        out["home_qb_epa"] = out["home_qb_epa_r"]
        out["away_qb_epa"] = out["away_qb_epa_r"]
    return out


# =====================================================================
#  WALK-FORWARD BACKTEST  (baseline Elo + optional EPA treatment)
# =====================================================================
# EPA/play differences are ~±0.2; convert to Elo points with this constant so a
# 0.1 EPA/play net edge ~ EPA_TO_ELO*0.1 Elo pts. The tunable weight scales it.
EPA_TO_ELO = 1000.0
QB_TO_ELO  = 1000.0


def run_backtest(g, w_epa=0.0, w_qb=0.0):
    """Walk forward. Team Elo R updates on p_base (baseline, no EPA) so BASELINE
    is byte-identical to nfl_model regardless of w_epa/w_qb. The treatment
    p_full adds the EPA/QB adjustment to the PREDICTION only.
    Returns a preds DataFrame with both p_base and p_full per game."""
    R = {}
    cur_season = None
    preds = []
    has_epa = "home_net_epa" in g.columns
    for r in g.itertuples():
        if r.season != cur_season:
            cur_season = r.season
            for t in R:
                R[t] = 1500 + (R[t] - 1500) * (1 - REVERT)
        h, a = r.home_team, r.away_team
        R.setdefault(h, 1500); R.setdefault(a, 1500)
        rest = 0.0
        if pd.notna(r.home_rest) and pd.notna(r.away_rest):
            rest = REST_PER_DAY * ((r.home_rest - 7) - (r.away_rest - 7))
        hfa = 0.0 if str(r.location) == "Neutral" else HFA_PTS
        base_dr = (R[h] + hfa + rest) - R[a]
        p_base = expected(base_dr)

        # ---- EPA treatment adjustment (prediction only, prior-only features) ----
        epa_adj = 0.0
        if has_epa:
            hn = getattr(r, "home_net_epa", np.nan)
            an = getattr(r, "away_net_epa", np.nan)
            if pd.notna(hn) and pd.notna(an):
                epa_adj += w_epa * EPA_TO_ELO * (hn - an)
            hq = getattr(r, "home_qb_epa", np.nan)
            aq = getattr(r, "away_qb_epa", np.nan)
            if pd.notna(hq) and pd.notna(aq):
                epa_adj += w_qb * QB_TO_ELO * (hq - aq)
        p_full = expected(base_dr + epa_adj)

        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin = r.home_score - r.away_score
            if r.season >= START_SEASON:
                preds.append({"season": r.season, "week": r.week,
                              "game_id": getattr(r, "game_id", None),
                              "p_base": p_base, "p_full": p_full,
                              "home_win": int(margin > 0), "tie": int(margin == 0),
                              "home_ml": getattr(r, "home_moneyline", np.nan),
                              "away_ml": getattr(r, "away_moneyline", np.nan)})
            # team Elo update — IDENTICAL to nfl_model (uses p_base, no EPA)
            mov = math.log(abs(margin) + 1) * (2.2 / (
                (0.001 * abs(base_dr) if margin * base_dr > 0 else -0.001 * abs(base_dr)) + 2.2))
            s_home = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            delta = K * mov * (s_home - p_base)
            R[h] += delta; R[a] -= delta
    return pd.DataFrame(preds)


# =====================================================================
#  SCORING
# =====================================================================
def _dec(ml):
    return ml / 100 + 1 if ml > 0 else 100 / (-ml) + 1


def market_p_home(row):
    if pd.isna(row.get("home_ml")) or pd.isna(row.get("away_ml")):
        return np.nan
    ih, ia = 1 / _dec(row["home_ml"]), 1 / _dec(row["away_ml"])
    return ih / (ih + ia)


def score(P, pcol):
    """Brier + accuracy for probability column `pcol` (ties dropped)."""
    P = P[P["tie"] == 0]
    if not len(P):
        return {"n": 0, "acc": None, "brier": None}
    acc = ((P[pcol] > 0.5).astype(int) == P["home_win"]).mean()
    brier = ((P[pcol] - P["home_win"]) ** 2).mean()
    return {"n": int(len(P)), "acc": round(100 * acc, 2), "brier": round(float(brier), 5)}


def market_disagreement_acc(P, pcol):
    """Accuracy of `pcol` on games where it disagrees with the closing market."""
    P = P[P["tie"] == 0].copy()
    P["mkt"] = P.apply(market_p_home, axis=1)
    M = P.dropna(subset=["mkt"])
    if not len(M):
        return {"n_mkt": 0, "market_acc": None, "n_disagree": 0, "acc_in_disagree": None}
    macc = ((M["mkt"] > 0.5).astype(int) == M["home_win"]).mean()
    dis = M[(M[pcol] > 0.5) != (M["mkt"] > 0.5)]
    dacc = (((dis[pcol] > 0.5).astype(int) == dis["home_win"]).mean()
            if len(dis) else None)
    return {"n_mkt": int(len(M)), "market_acc": round(100 * macc, 2),
            "n_disagree": int(len(dis)),
            "acc_in_disagree": (round(100 * dacc, 2) if dacc is not None else None)}


# =====================================================================
#  EXPERIMENT DRIVER + VERDICT
# =====================================================================
def experiment(years):
    g = load_games()
    print(f"games.csv: {len(g)} games, seasons {int(g.season.min())}-{int(g.season.max())}")

    print("Pulling nflverse play-by-play EPA (parquet -> csv.gz) ...")
    pbp, reachable = fetch_pbp(years)
    if pbp is None:
        print("\n" + "=" * 68)
        print("PBP UNREACHABLE from here — run on Actions.")
        print("This cloud host's egress is blocked to the nflverse GitHub releases")
        print("(expected: 403 'Tunnel connection failed'). On GitHub Actions the pull")
        print("succeeds and the full baseline-vs-EPA verdict is produced.")
        print("Run: python3 nfl_epa_experiment.py")
        print("=" * 68)
        return 0

    print(f"pbp rows: {len(pbp)}")
    team_game, qb = aggregate_epa(pbp)
    print(f"aggregated: {len(team_game)} team-game EPA rows, {len(qb)} qb-game rows")

    gf = build_game_features(g, team_game, qb, window=10)

    # baseline preds (w=0). Tie/market data identical across runs.
    base = run_backtest(gf, w_epa=0.0, w_qb=0.0)

    def split(P, pcol):
        tr = score(P[P["season"].isin(TRAIN_SEASONS)], pcol)
        ho = score(P[P["season"].isin(HOLD_SEASONS)], pcol)
        return tr, ho

    b_tr, b_ho = split(base, "p_base")
    print("\nBASELINE (team-only Elo):")
    print(f"  train   {b_tr}")
    print(f"  HOLDOUT {b_ho}")

    # ---- tune blend weights on TRAIN by Brier ----
    print("\nTuning EPA blend on TRAIN seasons (lower Brier = better):")
    best = None
    grid_epa = (0.0, 0.5, 1.0, 1.5, 2.0)
    grid_qb  = (0.0, 0.5, 1.0, 1.5)
    for w_epa in grid_epa:
        for w_qb in grid_qb:
            if w_epa == 0.0 and w_qb == 0.0:
                continue
            P = run_backtest(gf, w_epa=w_epa, w_qb=w_qb)
            tr = score(P[P["season"].isin(TRAIN_SEASONS)], "p_full")
            if tr["brier"] is None:
                continue
            if best is None or tr["brier"] < best[0]:
                best = (tr["brier"], w_epa, w_qb, tr)
    print(f"  best-on-train: w_epa={best[1]} w_qb={best[2]}  train {best[3]}")

    # ---- apply best weights; verdict on HOLDOUT ----
    Pt = run_backtest(gf, w_epa=best[1], w_qb=best[2])
    t_tr, t_ho = split(Pt, "p_full")

    base_dis = market_disagreement_acc(base[base["season"].isin(HOLD_SEASONS)], "p_base")
    treat_dis = market_disagreement_acc(Pt[Pt["season"].isin(HOLD_SEASONS)], "p_full")

    print("\nTREATMENT (Elo + rolling-EPA + QB-EPA), best train weights:")
    print(f"  train   {t_tr}")
    print(f"  HOLDOUT {t_ho}")

    # ---- per-season holdout deltas ----
    print("\nPer-season HOLDOUT deltas (treatment - baseline):")
    print(f"  {'season':7} {'base_brier':>11} {'treat_brier':>11} {'dBrier':>8} "
          f"{'base_acc':>9} {'treat_acc':>9} {'dAcc':>7}")
    for s in HOLD_SEASONS:
        bs = score(base[base["season"] == s], "p_base")
        ts = score(Pt[Pt["season"] == s], "p_full")
        if bs["n"] == 0:
            continue
        db = round(ts["brier"] - bs["brier"], 5)
        da = round(ts["acc"] - bs["acc"], 2)
        print(f"  {s:<7} {bs['brier']:>11} {ts['brier']:>11} {db:>8} "
              f"{bs['acc']:>9} {ts['acc']:>9} {da:>7}")

    # ---- VERDICT ----
    dBrier = round(t_ho["brier"] - b_ho["brier"], 5)
    dAcc = round(t_ho["acc"] - b_ho["acc"], 2)
    seasons_improved = 0
    seasons_total = 0
    for s in HOLD_SEASONS:
        bs = score(base[base["season"] == s], "p_base")
        ts = score(Pt[Pt["season"] == s], "p_full")
        if bs["n"] == 0:
            continue
        seasons_total += 1
        if ts["brier"] < bs["brier"]:
            seasons_improved += 1
    # "robust" = lower aggregate holdout Brier AND a majority of holdout seasons improved
    robust = (dBrier < 0) and (seasons_improved > seasons_total / 2)

    print("\n" + "=" * 68)
    print("VERDICT — does play-by-play EPA beat team-Elo out of sample?")
    print("=" * 68)
    print(f"  HOLDOUT Brier:    baseline {b_ho['brier']}  ->  treatment {t_ho['brier']}  "
          f"(delta {dBrier}, negative=better)")
    print(f"  HOLDOUT accuracy: baseline {b_ho['acc']}%  ->  treatment {t_ho['acc']}%  "
          f"(delta {dAcc} pts)")
    print(f"  Market on holdout: {base_dis['market_acc']}%  "
          f"(baseline right in {base_dis['n_disagree']} disagreements: {base_dis['acc_in_disagree']}%; "
          f"treatment right in {treat_dis['n_disagree']}: {treat_dis['acc_in_disagree']}%)")
    print(f"  Seasons EPA improved Brier: {seasons_improved}/{seasons_total}")
    if robust:
        print("  ==> EPA ROBUSTLY WINS: lower holdout Brier AND a majority of holdout seasons.")
    elif dBrier < 0:
        print("  ==> EPA edges ahead on aggregate Brier but NOT a season majority — not robust.")
    else:
        print("  ==> EPA does NOT beat team-Elo out of sample. Keep the current model.")
    print("=" * 68)
    return 0


# =====================================================================
#  OFFLINE SELF-TEST (NO NETWORK, SYNTHETIC DATA)
# =====================================================================
def _synthetic_games(seed=0, n_seasons=3, teams=None):
    """Build a small synthetic schedule with the columns the Elo loop needs."""
    rng = np.random.default_rng(seed)
    teams = teams or ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for si, season in enumerate(range(2018, 2018 + n_seasons)):
        for week in range(1, 9):
            rng.shuffle(teams)
            for i in range(0, len(teams), 2):
                h, a = teams[i], teams[i + 1]
                hs = int(rng.integers(10, 35)); as_ = int(rng.integers(10, 35))
                if hs == as_:
                    hs += 1
                rows.append({
                    "game_id": f"{season}_{week:02d}_{a}_{h}",
                    "season": season, "week": week,
                    "gameday": pd.Timestamp("2018-09-01") + pd.Timedelta(days=si * 200 + week * 7 + i),
                    "home_team": h, "away_team": a,
                    "home_qb_id": f"QB_{h}", "away_qb_id": f"QB_{a}",
                    "home_score": hs, "away_score": as_,
                    "home_rest": 7, "away_rest": 7, "location": "Home",
                    "home_moneyline": -150, "away_moneyline": 130,
                    "spread_line": -3.0,
                })
    g = pd.DataFrame(rows)
    return g.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def _elo_pbase_series(g):
    """Reimplemented-Elo p_home sequence via run_backtest (w=0)."""
    P = run_backtest(g, 0.0, 0.0)
    return P["p_base"].to_numpy()


def _nflmodel_phome_series(g):
    """nfl_model.run_elo p_home sequence for the same frame."""
    import nfl_model
    _, P = nfl_model.run_elo(g, start_season=START_SEASON)
    return P["p_home"].to_numpy()


def selftest():
    print("SELFTEST (offline, synthetic) — no network used")
    ok = True

    # (a) LEAK-FREE rolling feature: shift(1) must exclude the current game.
    df = pd.DataFrame({
        "team": ["X", "X", "X", "X"],
        "gameday": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15", "2020-01-22"]),
        "val": [1.0, 2.0, 3.0, 100.0],   # last game is a huge outlier
    })
    out = rolling_shift_mean(df, "team", "gameday", ["val"], window=10)
    got = out.sort_values("gameday")["val_roll"].tolist()
    # row0: no history -> NaN; row1: mean(1)=1; row2: mean(1,2)=1.5; row3: mean(1,2,3)=2
    assert math.isnan(got[0]), f"(a) row0 should be NaN, got {got[0]}"
    assert abs(got[1] - 1.0) < 1e-12, f"(a) row1 {got[1]}"
    assert abs(got[2] - 1.5) < 1e-12, f"(a) row2 {got[2]}"
    assert abs(got[3] - 2.0) < 1e-12, (
        f"(a) LEAK: row3 rolling feature={got[3]} must equal mean(1,2,3)=2.0 and "
        f"must NOT include the current game's 100.0")
    # explicit leak check: the current game's value never enters its own feature
    assert 100.0 not in [round(x, 6) for x in got if not math.isnan(x)], \
        "(a) LEAK: current-game value 100.0 appeared in a pre-game feature"
    print("  (a) leak-free rolling shift(1): PASS "
          "(current game excluded; row0 NaN; means = prior-only)")

    # window bound: a window of 2 must only see the 2 prior games.
    out2 = rolling_shift_mean(df, "team", "gameday", ["val"], window=2)
    g2 = out2.sort_values("gameday")["val_roll"].tolist()
    assert abs(g2[3] - 2.5) < 1e-12, f"(a2) window=2 row3 should be mean(2,3)=2.5, got {g2[3]}"
    print("  (a2) bounded window uses only prior N games: PASS")

    # (b) SCORING math on a hand-computed fixture.
    fix = pd.DataFrame({
        "tie": [0, 0, 0, 0],
        "home_win": [1, 0, 1, 0],
        "p_base": [0.8, 0.6, 0.4, 0.1],
        "p_full": [0.8, 0.6, 0.4, 0.1],
        "home_ml": [np.nan] * 4, "away_ml": [np.nan] * 4,
    })
    # Brier = mean((p-y)^2) = [(0.8-1)^2+(0.6-0)^2+(0.4-1)^2+(0.1-0)^2]/4
    #       = [0.04 + 0.36 + 0.36 + 0.01]/4 = 0.77/4 = 0.1925
    # Acc: preds>0.5 -> [1,1,0,0] vs y [1,0,1,0] -> correct [T,F,F,T] = 2/4 = 50%
    sc = score(fix, "p_base")
    assert abs(sc["brier"] - 0.1925) < 1e-9, f"(b) brier {sc['brier']} != 0.1925"
    assert abs(sc["acc"] - 50.0) < 1e-9, f"(b) acc {sc['acc']} != 50.0"
    assert sc["n"] == 4, f"(b) n {sc['n']}"
    # ties must be dropped:
    fix2 = pd.concat([fix, pd.DataFrame([{"tie": 1, "home_win": 0, "p_base": 0.9,
                                          "p_full": 0.9, "home_ml": np.nan,
                                          "away_ml": np.nan}])], ignore_index=True)
    assert score(fix2, "p_base")["n"] == 4, "(b) tie not dropped"
    print("  (b) scoring math (Brier=0.1925, acc=50%, ties dropped): PASS")

    # market-disagreement scoring on a tiny fixture
    md = pd.DataFrame({
        "tie": [0, 0],
        "home_win": [1, 0],
        "p_base": [0.7, 0.7],            # model likes home both games
        "home_ml": [120, -200],          # game1 mkt underdog home, game2 mkt fav home
        "away_ml": [-140, 170],
    })
    dd = market_disagreement_acc(md, "p_base")
    # game1: mkt p_home<0.5 but model>0.5 -> disagreement, model says home, home won -> right
    # game2: mkt p_home>0.5 and model>0.5 -> agreement, excluded
    assert dd["n_disagree"] == 1 and dd["acc_in_disagree"] == 100.0, \
        f"(b2) disagreement {dd}"
    print("  (b2) market-disagreement accuracy: PASS")

    # (c) ELO BASELINE reproduces nfl_model.run_elo byte-for-byte.
    g = _synthetic_games(seed=7)
    mine = _elo_pbase_series(g)
    try:
        theirs = _nflmodel_phome_series(g)
        assert len(mine) == len(theirs), f"(c) length {len(mine)} vs {len(theirs)}"
        maxdiff = float(np.max(np.abs(mine - theirs)))
        assert maxdiff < 1e-12, f"(c) Elo p_home diff {maxdiff} exceeds 1e-12"
        print(f"  (c) baseline Elo reproduces nfl_model.run_elo "
              f"(max |diff|={maxdiff:.2e} over {len(mine)} games): PASS")
    except ImportError:
        print("  (c) nfl_model not importable here; skipping cross-check (non-fatal)")

    # (c2) w=0 treatment == baseline exactly, and an EPA feature actually moves p.
    Pz = run_backtest(g, 0.0, 0.0)
    assert (Pz["p_base"] == Pz["p_full"]).all(), "(c2) w=0 treatment != baseline"
    # attach synthetic EPA features and confirm p_full diverges from p_base
    syn = g.copy()
    rng = np.random.default_rng(1)
    syn["home_net_epa"] = rng.normal(0, 0.1, len(syn))
    syn["away_net_epa"] = rng.normal(0, 0.1, len(syn))
    syn["home_qb_epa"] = rng.normal(0, 0.1, len(syn))
    syn["away_qb_epa"] = rng.normal(0, 0.1, len(syn))
    Pt = run_backtest(syn, w_epa=1.0, w_qb=1.0)
    assert (Pt["p_base"] == Pz["p_base"]).all(), "(c2) baseline changed under treatment!"
    assert (Pt["p_full"] != Pt["p_base"]).any(), "(c2) EPA layer had no effect"
    print("  (c2) treatment leaves baseline byte-identical AND EPA layer moves p_full: PASS")

    # (c3) end-to-end synthetic pipeline: aggregate -> features -> backtest runs.
    pbp = _synthetic_pbp(g)
    tg, qb = aggregate_epa(pbp)
    assert len(tg) > 0, "(c3) no team-game EPA aggregated"
    gf = build_game_features(g, tg, qb, window=5)
    assert "home_net_epa" in gf.columns and "home_qb_epa" in gf.columns
    _ = run_backtest(gf, w_epa=1.0, w_qb=1.0)   # must not raise
    # first appearance of each team has no prior EPA -> NaN feature (leak-free at boundary)
    assert gf["home_net_epa"].isna().any(), "(c3) expected NaN pre-game features early"
    print("  (c3) synthetic aggregate->features->backtest pipeline: PASS")

    print("\nSELFTEST: ALL PASSED")
    return 0


def _synthetic_pbp(g):
    """Fabricate a tiny pbp table consistent with the synthetic schedule so the
    aggregation + feature code paths are exercised offline."""
    rng = np.random.default_rng(3)
    rows = []
    for r in g.itertuples():
        for team, opp in ((r.home_team, r.away_team), (r.away_team, r.home_team)):
            for _ in range(20):  # 20 offensive plays
                rows.append({
                    "game_id": r.game_id, "season": r.season,
                    "posteam": team, "defteam": opp,
                    "epa": float(rng.normal(0, 0.5)),
                    "qb_epa": float(rng.normal(0, 0.5)),
                    "passer_player_id": f"QB_{team}",
                    "pass": 1, "rush": 0, "play_type": "pass",
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    # full experiment: default season range 1999-2025 (nflverse pbp coverage)
    years = range(1999, 2026)
    if "--years" in sys.argv:
        i = sys.argv.index("--years")
        lo, hi = sys.argv[i + 1].split("-")
        years = range(int(lo), int(hi) + 1)
    sys.exit(experiment(years))
