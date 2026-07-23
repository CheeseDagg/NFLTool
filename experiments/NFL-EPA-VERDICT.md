games.csv not present — downloading from nflverse ...
games.csv: 7548 games, seasons 1999-2026
Pulling nflverse play-by-play EPA (parquet -> csv.gz) ...
pbp rows: 1279628
aggregated: 14546 team-game EPA rows, 17918 qb-game rows

BASELINE (team-only Elo):
  train   {'n': 2662, 'acc': np.float64(65.4), 'brier': 0.2193}
  HOLDOUT {'n': 1688, 'acc': np.float64(63.39), 'brier': 0.2224}

Tuning EPA blend on TRAIN seasons (lower Brier = better):
  best-on-train: w_epa=0.5 w_qb=0.0  train {'n': 2662, 'acc': np.float64(64.12), 'brier': 0.224}

TREATMENT (Elo + rolling-EPA + QB-EPA), best train weights:
  train   {'n': 2662, 'acc': np.float64(64.12), 'brier': 0.224}
  HOLDOUT {'n': 1688, 'acc': np.float64(63.74), 'brier': 0.22699}

Per-season HOLDOUT deltas (treatment - baseline):
  season   base_brier treat_brier   dBrier  base_acc treat_acc    dAcc
  2020        0.21379     0.21194 -0.00185     64.55     66.04    1.49
  2021        0.23148     0.24003  0.00855     60.56     61.27    0.71
  2022        0.22308      0.2303  0.00722     63.48     62.77   -0.71
  2023        0.23104     0.23741  0.00637     60.35     62.11    1.76
  2024        0.21142     0.21434  0.00292     67.37     66.67    -0.7
  2025        0.22312     0.22711  0.00399     64.08     63.73   -0.35

====================================================================
VERDICT — does play-by-play EPA beat team-Elo out of sample?
====================================================================
  HOLDOUT Brier:    baseline 0.2224  ->  treatment 0.22699  (delta 0.00459, negative=better)
  HOLDOUT accuracy: baseline 63.39%  ->  treatment 63.74%  (delta 0.35 pts)
  Market on holdout: 66.71%  (baseline right in 276 disagreements: 39.86%; treatment right in 252: 40.08%)
  Seasons EPA improved Brier: 1/6
  ==> EPA does NOT beat team-Elo out of sample. Keep the current model.
====================================================================
