# macro-forecasting-lab

![CI](https://github.com/aydinmonavvari/macro-forecasting-lab/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 1 · Short description

A reproducible study of how classical statistical time-series models and machine-learning
models forecast key US macroeconomic indicators — **under strict chronological
(out-of-sample) validation** — compared against honest naive baselines.

*Educational research project. Not economic or investment advice.*

## 2 · Research question

> How accurately can classical statistical models (ARIMA/SARIMA) and machine-learning
> models (OLS, Random Forest, Gradient Boosting) forecast US inflation and unemployment
> relative to naive baselines, under a leakage-free, chronological evaluation design?

## 3 · Motivation

Macroeconomic forecasting is a core task in applied economics, central-bank analysis and
financial planning, and it is also one of the easiest tasks to get *methodologically
wrong*. Random train/test splits, target leakage through contemporaneous regressors, and
silent comparisons against weak baselines produce inflated accuracy claims that collapse
in real use. This project is built around the discipline that macro forecasting demands:
strict temporal ordering, direct multi-step targets, explicit baselines, and formal
significance testing rather than eyeballing RMSE tables.

## 4 · Why this matters

- **For economics:** knowing *whether* sophisticated models actually beat a seasonal
  naive benchmark at a given horizon is a substantive empirical question, and the honest
  answer in this study is nuanced (some yes, many no — see Results).
- **For machine-learning practice:** macro data are short (≈ 800 monthly observations),
  non-stationary, and subject to structural breaks — an ideal setting to learn why
  cross-validation-as-usual fails and what replaces it.
- **For the research portfolio:** this repository establishes the time-series
  methodology (chronological splits, direct *h*-step targets, Diebold–Mariano testing)
  that the later financial-ML projects in this portfolio build upon.

## 5 · Methodology

1. **Data acquisition** — five monthly FRED series are downloaded (public CSV endpoint;
   optional official API when `FRED_API_KEY` is set), cached locally, validated and
   transformed (see §7).
2. **Stationarity analysis** — ADF and KPSS tests on the train+validation region guide
   the differencing order `d` for ARIMA/SARIMA (selected: d=1 for inflation, d=0 for
   unemployment).
3. **Model order selection** — a small AIC grid search over ARIMA/SARIMA orders uses
   **train+validation only**; the test window is never touched until final scoring.
4. **Direct multi-step forecasting** — for horizons h ∈ {1, 12} the supervised models are
   trained on targets `y_{t+h}` using features available at time `t` (explicit
   `shift(h)`), avoiding recursive error accumulation and, more importantly, avoiding any
   path where future information can leak into features.
5. **Rolling-origin evaluation** — models are re-fit at each evaluation origin and scored
   on every 3rd test month (evaluation step; Tashman 2000), producing genuine
   out-of-sample forecasts across 2016-01 → 2026-07.
6. **Significance testing** — Diebold–Mariano (1995) tests with a HAC long-run variance
   whose bandwidth is `h − 1` lags for h-step forecasts (`dm_max_lag: auto`; the fixed
   lag-1 specification is reported as a sensitivity) compare every non-benchmark model —
   `naive_last` included — against the seasonal-naive benchmark; a Bonferroni/Holm-style
   multiple-testing caution is applied in the interpretation (§13.4).
7. **Structural-break analysis** — all metrics are reported for the full test window and
   for a window excluding the COVID disruption (2020-02 → 2020-12).

## 6 · Dataset

| FRED ID | Modeled series | Transform | Units |
| --- | --- | --- | --- |
| CPIAUCSL | `inflation_yoy` | YoY log-difference ×100 | percent |
| UNRATE | `unemployment_rate` | level | percent of labor force |
| INDPRO | `industrial_production_yoy` | YoY log-difference ×100 | percent (feature) |
| FEDFUNDS | `fed_funds` | level | percent (feature) |
| M2SL | `m2_yoy` | YoY log-difference ×100 | percent (feature) |

Coverage after transformation: **1960-01 → 2026-08 (800 monthly observations)**.

## 7 · Data sources

All series come from [FRED](https://fred.stlouisfed.org) (Federal Reserve Bank of St.
Louis), retrieved either through the public chart-data CSV endpoint or the official FRED
API (free key, read from the `FRED_API_KEY` environment variable only — never hardcoded
or committed). FRED data are freely redistributable with attribution to the source.
Raw CSVs are cached under `data/raw/` (git-ignored) so the full study reruns offline.
Interior monthly gaps of at most 2 months are linearly interpolated on the raw levels
*before* any transformation (`fill_monthly_gaps` in `src/macro_forecasting_lab/data.py`);
no gap required filling in this sample, and the policy is a documented data-handling
decision.

## 8 · Architecture

```
FRED (CSV / API) ──► download + cache ──► transforms + validation
                                              │
                     stationarity (ADF/KPSS) ◄─┘
                              │
              AIC grid: ARIMA / SARIMA orders (train+val only)
                              │
        ┌─────────────────────┼──────────────────────┐
   baselines            univariate               supervised
 (naive, seasonal,   (ARIMA, SARIMA,      (OLS + lags, RandomForest,
  rolling mean)       direct h-step)       GradientBoosting; direct h-step)
        └─────────────────────┼──────────────────────┘
                       rolling-origin evaluation (h = 1, 12)
                              │
             RMSE · MAE · MASE · Diebold–Mariano · intervals
                              │
              reports/results.csv · figures · research report
```

## 9 · Experimental design

**Chronological split (no shuffling, ever):**

| Region | Period | Used for |
| --- | --- | --- |
| Train | 1960-01 → 2009-12 | model fitting |
| Validation | 2010-01 → 2015-12 | AIC order selection, model comparison |
| Test | 2016-01 → 2026-07 | rolling-origin out-of-sample scoring (43 evaluation months) |

**Leakage controls** (each enforced in code and covered by unit tests):

- direct *h*-step targets built with a single explicit `shift(h)`;
- supervised features use only information dated ≤ the forecast origin;
- ARIMA/SARIMA order selection and supervised training never see test-window data;
- the evaluation step subsamples target months identically in the prediction and scoring
  stages so no origin ever sees its own target.

**Structural breaks:** the test window contains the 2020 COVID disruption. Metrics are
reported for the full window and excluding 2020-02 → 2020-12, and the difference is
discussed in the results.

## 10 · Models

| Family | Models |
| --- | --- |
| Baselines | naive random walk (last value), seasonal naive (12 months back), rolling 12-month mean |
| Classical | ARIMA (AIC-selected per series), SARIMA with annual seasonality (AIC-selected) |
| Machine learning | OLS on lagged features, Random Forest (200 trees), Gradient Boosting |

Selected orders (AIC, train+validation): inflation — ARIMA(1,1,2), SARIMA(2,1,2)(1,0,1)₁₂;
unemployment — ARIMA(2,0,2), SARIMA(2,0,2)(1,0,0)₁₂.

## 11 · Evaluation metrics

- **RMSE / MAE** in the natural units of each series.
- **MASE** — mean absolute scaled error against the in-sample seasonal-naive benchmark
  (Hyndman & Koehler 2006): MASE < 1 beats the seasonal naive on absolute error.
- **Diebold–Mariano p-values** — whether a model's loss difference vs the seasonal
  naive is statistically significant. The HAC variance bandwidth is `h − 1` lags for
  h-step forecasts (`dm_max_lag: auto`): overlapping h-step forecast errors follow an
  MA(h−1) process, so a *smaller* bandwidth is anti-conservative (it understates the
  long-run variance and inflates |DM|). A fixed lag-1 bandwidth is reported alongside
  as a sensitivity. Raw p-values are always shown; a Bonferroni/Holm-style
  multiple-testing caution is applied across the 7 non-benchmark model comparisons per
  series/horizon/window (§13.4).
- **Prediction-interval coverage** visualized for ARIMA/SARIMA.

## 12 · Results

Actual rolling-origin results (full test window, 43 evaluation months, 2016-01 → 2026-07).
Full tables: [`reports/results.csv`](reports/results.csv); window excluding COVID months
2020-02→2020-12 shows the same qualitative pattern.

**US CPI inflation (YoY, %):**

| Horizon | Best model | RMSE | MASE | DM p (h−1) | DM p (lag-1) | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| h = 1 | SARIMA(2,1,2)(1,0,1)₁₂ | 0.246 | 0.123 | **0.0001** | 0.014 | significantly beats the seasonal naive; the only model that also clearly beats `naive_last` |
| h = 12 | OLS + lags | 1.652 | 0.961 | 0.131 | 0.133 | nominally best; **not** significant under either bandwidth |

**US unemployment rate (%):**

| Horizon | Best model (RMSE) | RMSE | MASE | DM p (h−1) | DM p (lag-1) | Reading |
| --- | --- | --- | --- | --- | --- | --- |
| h = 1 | Gradient Boosting | 1.595 | 0.512 | 0.022 | 0.081 | raw p < 0.05 vs the seasonal naive, but it does **not** survive the family correction (§13.4); `naive_last` has the better MASE (0.470) |
| h = 12 | rolling 12m mean | 2.448 | 1.799 | 0.157 | 0.596 | **no model beats the seasonal naive**; SARIMA is worst by RMSE (5.13) and Random Forest worst by MASE (2.66) |

Every non-benchmark model — `naive_last` included — is tested against the seasonal-naive
benchmark (7 comparisons per series/horizon/window; the benchmark itself has no DM row).
At h = 12 `naive_last` and the seasonal naive coincide by construction, so `naive_last`'s
DM row there is degenerate (statistic exactly 0).

Key figures (generated by the pipeline from the actual run):

| Figure | Content |
| --- | --- |
| [`series_splits_inflation_yoy.png`](figures/series_splits_inflation_yoy.png) | transformed series with train/val/test shading |
| [`forecasts_inflation_yoy_h1.png`](figures/forecasts_inflation_yoy_h1.png) | h=1 forecasts vs actuals across the test window |
| [`forecasts_inflation_yoy_h12.png`](figures/forecasts_inflation_yoy_h12.png) | h=12 forecast paths vs actuals |
| [`intervals_inflation_yoy_sarima_h12.png`](figures/intervals_inflation_yoy_sarima_h12.png) | SARIMA 95% prediction intervals |
| [`dm_inflation_yoy_h1.png`](figures/dm_inflation_yoy_h1.png) | Diebold–Mariano loss differentials vs seasonal naive |
| [`residual_acf_inflation_yoy_sarima_h1.png`](figures/residual_acf_inflation_yoy_sarima_h1.png) | SARIMA residual autocorrelation diagnostics |

Equivalent figures for unemployment are committed under the same naming scheme.

## 13 · Interpretation

1. **Horizon is destiny.** At h=1, structure-exploiting models deliver real gains where
   the target has strong dynamics (inflation: SARIMA, raw DM p ≈ 0.0001 under the
   primary h−1 bandwidth — a result that survives the 28-comparison family correction;
   under the lag-1 sensitivity, p = 0.014, which does not survive). At h=12, *no* model
   significantly beats the seasonal naive for either series under either bandwidth — the
   information in the monthly history simply does not support accurate year-ahead point
   forecasts, which is consistent with the published macro-forecasting literature.
2. **Machine learning adds little here, honestly measured.** At inflation h=1 the tree
   ensembles clear the seasonal-naive bar exactly like every other non-benchmark model —
   but that bar is very weak (the seasonal naive has MASE 1.14 for inflation). Against
   the *strong* naive baseline (`naive_last`), Random Forest and Gradient Boosting win
   nowhere, at any horizon or series; at unemployment their raw DM p ≈ 0.022 (primary
   bandwidth) fails the family correction, and the lag-1 sensitivity does not even reach
   nominal significance (p ≈ 0.08). On short monthly macro samples (600 training months,
   1960-01 → 2009-12), flexible learners mostly fit noise; the linear model with lagged
   features (OLS) is the only ML-family model that is ever nominally best.
3. **The best "model" is often persistence.** For unemployment excluding COVID, the
   naive last-value forecast is essentially unbeatable at h=1 (MASE 0.122) — monthly
   unemployment is extremely persistent outside recessions. Any claimed ML edge in the
   full window comes almost entirely from the COVID episode.
4. **Multiple-testing caution.** Seven non-benchmark models are compared against the
   same seasonal-naive benchmark per series/horizon/window. Per scoring window the
   family is therefore 7 models × 2 series × 2 horizons = **28 comparisons**, and a
   Bonferroni-corrected threshold of 0.05/28 ≈ 0.0018 (the first Holm step) is the
   appropriate reading. Under the primary h−1 bandwidth the seven inflation h=1
   comparisons (raw p ≤ 0.0004) survive; every other rejection — including unemployment
   h=1 (raw p ≈ 0.022) — does not. Under the lag-1 sensitivity no comparison survives
   (smallest raw p = 0.014). Raw p-values are always reported so readers can apply any
   other correction.
5. **COVID as a structural break.** Excluding 2020-02→2020-12 changes unemployment
   metrics by an order of magnitude (naive h=1 RMSE falls from 1.60 to 0.14), while
   inflation metrics barely move. Forecast evaluations on 2016+ macro data are,
   effectively, evaluations of how models handled a pandemic.

## 14 · Limitations

- **Revised, contemporaneous data (no real-time vintages).** FRED series are final
  revised values, and the information set is contemporaneous: the supervised features
  include the target's own origin-month value (`y_l0`), and cross-series features enter
  lagged one month. In real time, the CPI observation for a month is published ≈2–3
  weeks and the unemployment rate ≈4–5 weeks after the reference month, and the earliest
  available observations would be preliminary, not revised. This design is therefore
  **not** a real-time/vintage simulation, and the results are an upper bound on
  real-time accuracy. Vintage analysis (ALFRED) is future work.
- **Small sample.** 600 training months (1960-01 → 2009-12), 72 validation months,
  43 scored test months. At h=12 the structurally correct h−1 HAC bandwidth is
  conservative and the lag-1 sensitivity is anti-conservative; either way the DM tests
  have limited power — "not significant" must not be read as "no difference".
- **Fixed evaluation step.** Scoring every 3rd test month (43 points) trades granularity
  for compute; the step is applied identically in prediction and scoring and is a
  standard rolling-origin design (Tashman 2000).
- **No structural-break modeling.** Breaks are handled by exclusion windows and
  discussion, not by time-varying-parameter or regime-switching models.
- **Two target series only.** INDPRO, FEDFUNDS and M2SL serve as features; forecasting
  them is future work.
- **Point forecasts only.** Density/probability forecasting (e.g., recession
  probabilities) is out of scope.

## 15 · Reproducibility

```bash
# 1) create environment (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# 2) download + cache FRED data (network; optional FRED_API_KEY for the official API)
python scripts/download_data.py

# 3) run the full study (~6 min on 4 CPU cores; offline from the cache afterwards)
python scripts/run_experiments.py

# 4) verify: lint + 48 offline unit tests
ruff check .
pytest -q
```

Determinism: seeds are pinned (`seed = 42`); AIC order selection and all model fits are
deterministic given the cached data. Every number and figure in this README regenerates
exactly from steps 2–3.

## 16 · Installation

```bash
git clone https://github.com/aydinmonavvari/macro-forecasting-lab.git
cd macro-forecasting-lab
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

## 17 · Usage

```bash
python scripts/download_data.py                 # fetch/cache FRED series
python scripts/run_experiments.py               # full study -> reports/ + figures/
pytest -q                                       # offline test suite
```

As a library:

```python
from pathlib import Path

from macro_forecasting_lab.data import load_series, transform_series
from macro_forecasting_lab.models import SARIMAForecaster

level = load_series("CPIAUCSL", raw_dir=Path("data/raw"))
inflation = transform_series(level, "yoy_log_diff_pct")

model = SARIMAForecaster(order=(2, 1, 2), seasonal_order=(1, 0, 1, 12))
model.fit(inflation.loc[:"2024-12-31"])
point_1m = model.forecast(1)  # scalar: the 1-month-ahead point forecast
print(f"Next-month inflation forecast: {point_1m:.2f}%")
```

## 18 · Example

Actual rolling-origin predictions produced by the study
(from [`reports/predictions.csv`](reports/predictions.csv)):

| Series | Horizon | Model | Target month | Prediction | Actual |
| --- | --- | --- | --- | --- | --- |
| inflation_yoy | 1 | sarima | 2026-07-01 | 3.12% | 3.25% |
| unemployment_rate | 1 | arima | 2026-07-01 | 4.25% | 4.10% |

And the same mechanics as a small library snippet:

```python
from pathlib import Path

from macro_forecasting_lab.data import load_series, transform_series
from macro_forecasting_lab.models import SARIMAForecaster

level = load_series("CPIAUCSL", raw_dir=Path("data/raw"))
inflation = transform_series(level, "yoy_log_diff_pct")

model = SARIMAForecaster(order=(2, 1, 2), seasonal_order=(1, 0, 1, 12))
model.fit(inflation.loc[:"2024-12-31"])
point_1m = model.forecast(1)  # scalar: the 1-month-ahead point forecast
print(f"Next-month inflation forecast: {point_1m:.2f}%")
```

> Note: long-horizon forecasts from *integrated* models (d=1) anchored on a
> near-zero regime can diverge (e.g. fitting in Dec-2015, right after the oil-crash
> disinflation, produces an explosive 12-month path). The rolling-origin study never
> relies on a single long-horizon extrapolation — it re-fits at every origin, which is
> exactly why the h=12 table above is honest about the failure modes.

## 19 · Project structure

```
macro-forecasting-lab/
├── README.md
├── LICENSE · CITATION.cff · pyproject.toml · requirements.txt
├── .python-version · .gitignore
├── src/macro_forecasting_lab/
│   ├── config.py          # dataclass config + FRED series specs
│   ├── data.py            # FRED download (CSV/API), caching, transforms, validation
│   ├── features.py        # lag matrices, direct-h target alignment
│   ├── models.py          # baselines, ARIMA/SARIMA wrappers, sklearn regressors
│   ├── evaluation.py      # RMSE/MAE/MASE, Diebold–Mariano, COVID-exclusion masks
│   ├── plots.py           # split shading, forecasts, intervals, DM charts
│   └── pipeline.py        # end-to-end orchestration
├── tests/                 # 48 offline unit tests (config, splits, leakage, metrics, DM bandwidth)
├── notebooks/             # EDA of the real FRED series
├── scripts/               # download_data.py, run_experiments.py
├── configs/default.yaml
├── data/raw · data/processed   # git-ignored caches (.gitkeep tracked)
├── reports/               # results.csv, predictions.csv, summary.md (generated)
├── figures/               # 16 generated figures (committed)
├── docs/research_report.md
└── .github/workflows/ci.yml
```

## 20 · Future work

- Real-time vintage evaluation via ALFRED to quantify the revision and
  publication-lag penalty (see §14).
- Regime-switching / time-varying-parameter models for structural breaks.
- Density forecasting and probability-of-recession classifiers.
- Extend targets to INDPRO, FEDFUNDS, M2SL and a global (World Bank) panel.
- Conformal prediction intervals for the ML models.

## 21 · Citation

If you use this work, please cite (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{monavvari2026macroforecastinglab,
  author  = {Monavvari, Aydin},
  title   = {macro-forecasting-lab: chronological-validation forecasting of US macro indicators},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/aydinmonavvari/macro-forecasting-lab}
}
```

## 22 · License

MIT — see [`LICENSE`](LICENSE).

## 23 · Acknowledgments

Data by [FRED](https://fred.stlouisfed.org), Federal Reserve Bank of St. Louis. Built
with pandas, statsmodels, scikit-learn and matplotlib.
