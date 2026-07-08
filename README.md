# NFLTool — Elo model, honest backtest, line-shop edges

Walk-forward Elo (64.6% over 16 seasons) measured against the closing market
(66.4%). In 693 disagreements the model was right 44.3% — so win% is decision
support and **line-shopping is the only signal this tool bets**. Every published
probability is logged pre-game and graded against the final score on the
Calibration tab, including a live continuation of the disagreement study.

Pipeline: `nfl_model.py` (Elo + backtest) → `nfl_odds.py` (The Odds API) →
`nfl_edge.py` (devig/consensus/best-price/quarter-Kelly) → `nfl_grade.py`
(settle + calibrate) → `nfl_publish.py` (slate.json) → `index.html` (dashboard).
Runs daily via `.github/workflows/nfl-daily.yml`. Data: nflverse.
