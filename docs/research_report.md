# Research Report — Macro Forecasting Lab

**Chronological-validation forecasting of US inflation and unemployment: classical time-series models vs machine learning against honest baselines**

*Author: Aydin Monavvari — educational research project (not economic or investment advice).*
*Run date: 2026-10-08. All numbers are actual outputs of the committed pipeline.*

---

## Abstract

This study evaluates whether classical statistical time-series models (ARIMA, SARIMA) and machine-learning models (OLS on lagged features, Random Forest, Gradient Boosting) provide out-of-sample forecasting gains over naive baselines for two core US macroeconomic indicators — year-over-year CPI inflation and the unemployment rate — under a strict chronological validation design. Five monthly FRED series (1960-01 → 2026-08, 800 observations) are transformed, tested for stationarity (ADF/KPSS), and forecast at horizons h = 1 and h = 12 months using a direct multi-step scheme with rolling-origin evaluation over 43 test months (2016-01 → 2026-07). Model differences are assessed with Diebold–Mariano tests against a seasonal-naive benchmark — every non-benchmark model, `naive_last` included — using a HAC variance whose bandwidth is h−1 lags for h-step forecasts (the structurally correct choice for overlapping forecast errors), with a fixed lag-1 bandwidth reported as a sensitivity; RMSE, MAE and MASE are reported alongside. The headline findings are horizon-dependent and bandwidth-dependent: at h = 1, seasonal ARIMA structure delivers the strongest inflation improvement (raw DM p ≈ 0.0001 under the primary h−1 bandwidth, which survives a 28-comparison family correction; raw p = 0.014 under the lag-1 sensitivity, which does not), while at h = 12 no model significantly beats the seasonal naive for either series under either bandwidth, and for unemployment outside the COVID episode the naive last-value forecast is essentially unbeatable (MASE = 0.122). The information set uses contemporaneous final revised observations (the origin-month value of each target is included as a feature), so the reported accuracy is an upper bound on real-time performance. The study illustrates both the value and the limits of rigorously evaluated forecasting in macroeconomics: where dynamics are strong and horizons short, structure helps; where horizons lengthen, honest baselines remain remarkably hard to beat.

## Introduction

Forecasts of inflation and unemployment anchor central-bank policy decisions, fiscal planning, and financial asset allocation. They are also a domain where methodological shortcuts — shuffled train/test splits, leakage through contemporaneous regressors, and comparisons against straw-man baselines — routinely produce overconfident claims. This project implements the full forecasting workflow for two US macro targets with the discipline the domain requires: strict temporal ordering of every split, direct multi-step targets that structurally exclude future information, explicit naive baselines, formal loss-differential testing, and transparent treatment of the COVID-19 structural break embedded in the test window.

The repository is the second step of a research portfolio progressing from descriptive financial analytics (Project 1) through econometrics and machine learning toward deep learning and NLP for finance (Projects 7–9). It establishes the chronological-validation toolkit reused by all subsequent forecasting work.

## Research Question

> How accurately can classical statistical models and machine-learning models forecast US inflation and unemployment relative to naive baselines, under a leakage-free chronological evaluation design — and where, specifically, do they add value?

## Related Work

- **Box, G. E. P., & Jenkins, G. M. (1970).** *Time Series Analysis: Forecasting and Control.* Holden-Day. The ARIMA framework and the iterate/forecast methodology used here.
- **Diebold, F. X., & Mariano, R. S. (1995).** "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253–263. The loss-differential test used for pairwise model comparison.
- **Hyndman, R. J., & Koehler, A. B. (2006).** "Another look at measures of forecast accuracy." *International Journal of Forecasting*, 22(4), 679–688. Source of MASE and the scaled-error argument against raw MAPE on sign-crossing series such as inflation.
- **Tashman, L. J. (2000).** "Out-of-sample tests of forecasting accuracy: an analysis and review." *International Journal of Forecasting*, 16(4), 437–450. Rolling-origin evaluation design, including the evaluation-step variant used here.
- **Hamilton, J. D. (1994).** *Time Series Analysis.* Princeton University Press. Stationarity, unit roots, and macro time-series inference.
- **Breiman, L. (2001).** "Random Forests." *Machine Learning*, 45(1), 5–32; and **Friedman, J. H. (2001).** "Greedy Function Approximation: A Gradient Boosting Machine." *Annals of Statistics*, 29(5), 1189–1232. The tree ensembles in the ML family.

## Data

| Property | Value (actual run) |
| --- | --- |
| Source | FRED (Federal Reserve Bank of St. Louis), public CSV endpoint; optional official API with a user-supplied `FRED_API_KEY` |
| Series | CPIAUCSL, UNRATE, INDPRO, FEDFUNDS, M2SL (monthly, seasonally adjusted) |
| Coverage | 1960-01 → 2026-08 (800 monthly observations after transformation) |
| Targets | CPI inflation (YoY log-difference ×100), unemployment rate (level) |
| Features | INDPRO growth, fed funds rate, M2 growth (lagged), plus target lags |
| Missing data | Monthly continuity validated; interior gaps up to 2 months linearly interpolated on raw levels before transformation (none required filling in this sample) |

Raw CSVs are cached under `data/raw/` (git-ignored); the pipeline is fully offline-reproducible from the cache. FRED data are redistributable with attribution.

All observations are final revised values as currently published by FRED (no vintages), and the supervised feature set is contemporaneous: it includes the target's own origin-month value (`y_l0`) and auxiliary series lagged one month. See Limitations for the publication-lag implications.

## Methodology

**Transforms.** CPI and INDPRO enter as year-over-year log-differences (×100); UNRATE and FEDFUNDS as levels; M2SL as YoY growth. Transforms are computed from cached raw levels at load time.

**Stationarity.** ADF and KPSS tests on the train+validation span of each *analysis* series determine the differencing order. The inflation series enters as the year-over-year log-difference transform (an already-differenced series — the diagnostic decides the integration order *on top of* the transform, not on the CPI index); unemployment enters as the level of the rate. Selected: d = 1 for inflation (ADF p = 0.039 on the transformed series), d = 0 for unemployment (ADF p = 0.032 on levels).

**Model selection.** ARIMA(p,d,q) and SARIMA(p,d,q)(P,D,Q)₁₂ orders are chosen by AIC over small grids using **train+validation only**. Selected: inflation — ARIMA(1,1,2), SARIMA(2,1,2)(1,0,1)₁₂; unemployment — ARIMA(2,0,2), SARIMA(2,0,2)(1,0,0)₁₂.

**Direct multi-step forecasting.** For h ∈ {1, 12}, supervised targets are `y_{t+h}` aligned to features dated ≤ t via an explicit `shift(h)`, eliminating recursive error accumulation and any path for target leakage. A dedicated unit test pins the target alignment (origin 2000-01 → h=12 target = 2001-01).

**Baselines.** Naive random walk (last value), seasonal naive (value 12 months prior), rolling 12-month mean.

**ML family.** OLS on lagged features; Random Forest (200 trees, seed 42); Gradient Boosting (seed 42).

**Evaluation.** Rolling-origin re-fitting at every evaluation origin; scoring on every 3rd test month (Tashman evaluation step) → 43 scored months. Metrics: RMSE, MAE, MASE (scaled vs in-sample seasonal-naive MAE; Hyndman & Koehler 2006), and Diebold–Mariano tests of each non-benchmark model — `naive_last` included — vs the seasonal naive. The HAC long-run variance bandwidth is h−1 lags for h-step forecasts: overlapping h-step forecast errors follow an MA(h−1) process, so a smaller bandwidth (the fixed lag-1 used before this revision) understates the long-run variance and is anti-conservative at h = 12; the fixed lag-1 specification is retained and reported as a sensitivity. All metrics are computed for the full test window and for a window excluding 2020-02 → 2020-12.

## Experimental Design

**Chronological split:** train 1960-01 → 2009-12; validation 2010-01 → 2015-12 (order selection, model comparison); test 2016-01 → 2026-07 (rolling-origin out-of-sample scoring). No observation from a later region ever informs an earlier one; the design is covered by unit tests asserting split boundary integrity.

**Why direct h-step.** Recursive strategies compound model error over the horizon and make leakage controls harder to audit; the direct strategy fixes the information set at the origin and is standard in the macro-forecasting literature.

**Structural breaks.** The test window contains the COVID-19 disruption, the largest macro shock in the sample. Rather than hiding it, the design reports all metrics with and without the COVID window and interprets the difference explicitly (see Discussion).

**Multiple testing.** Seven non-benchmark models are compared per series/horizon/window. Per scoring window the family is 7 models × 2 series × 2 horizons = 28 comparisons against the same benchmark; the Bonferroni threshold — equivalently the first Holm step — is 0.05/28 ≈ 0.0018. The pipeline's summary writer computes this family size from the data rather than hard-coding it. Raw p-values are always reported so that any other correction can be applied by the reader.

## Results

Actual rolling-origin results, full test window (43 scored months, 2016-01 → 2026-07). DM p-values are shown for the primary h−1 bandwidth and, in parentheses, the fixed lag-1 sensitivity. Complete tables (both windows, all columns): `reports/results.csv`.

**Inflation (YoY %):**

| h | Model | RMSE | MAE | MASE | DM p (h−1) | DM p (lag-1) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **SARIMA(2,1,2)(1,0,1)₁₂** | **0.246** | **0.171** | **0.123** | **0.0001** | 0.014 |
| 1 | ARIMA(1,1,2) | 0.362 | 0.273 | 0.195 | 0.0001 | 0.015 |
| 1 | OLS | 0.368 | 0.279 | 0.199 | 0.0001 | 0.015 |
| 1 | naive last | 0.394 | 0.271 | 0.194 | 0.0001 | 0.016 |
| 1 | Gradient Boosting | 0.399 | 0.273 | 0.195 | 0.0001 | 0.016 |
| 1 | Random Forest | 0.415 | 0.276 | 0.198 | 0.0001 | 0.016 |
| 1 | rolling 12m mean | 1.261 | 0.887 | 0.635 | 0.0003 | 0.022 |
| 1 | seasonal naive | 2.181 | 1.597 | 1.142 | — | — |
| 12 | **OLS** | **1.652** | **1.344** | **0.961** | 0.131 | 0.133 |
| 12 | SARIMA | 1.888 | 1.438 | 1.029 | 0.193 | 0.241 |
| 12 | ARIMA(1,1,2) | 2.156 | 1.593 | 1.139 | 0.279 | 0.584 |
| 12 | naive last¹ | 2.181 | 1.597 | 1.142 | 1.000 | 1.000 |
| 12 | seasonal naive | 2.181 | 1.597 | 1.142 | — | — |
| 12 | Gradient Boosting | 2.442 | 1.579 | 1.130 | 0.136 | 0.477 |
| 12 | Random Forest | 2.513 | 1.627 | 1.164 | 0.103 | 0.345 |
| 12 | rolling 12m mean | 2.521 | 1.816 | 1.299 | 0.169 | 0.281 |

**Unemployment rate (%):**

| h | Model | RMSE | MAE | MASE | DM p (h−1) | DM p (lag-1) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **Gradient Boosting** | **1.595** | 0.404 | 0.512 | 0.022 | 0.081 |
| 1 | naive last | 1.602 | **0.371** | **0.470** | 0.022 | 0.080 |
| 1 | Random Forest | 1.607 | 0.388 | 0.492 | 0.022 | 0.080 |
| 1 | ARIMA(2,0,2) | 1.615 | 0.398 | 0.504 | 0.022 | 0.080 |
| 1 | SARIMA | 1.617 | 0.408 | 0.517 | 0.022 | 0.080 |
| 1 | OLS | 1.624 | 0.440 | 0.558 | 0.022 | 0.082 |
| 1 | rolling 12m mean | 1.935 | 0.750 | 0.951 | 0.044 | 0.112 |
| 1 | seasonal naive | 2.685 | 1.355 | 1.718 | — | — |
| 12 | rolling 12m mean | 2.448 | 1.419 | 1.799 | 0.157 | 0.596 |
| 12 | naive last¹ | 2.685 | 1.355 | 1.718 | 1.000 | 1.000 |
| 12 | seasonal naive | 2.685 | 1.355 | 1.718 | — | — |
| 12 | Gradient Boosting | 2.834 | 2.025 | 2.567 | 0.675 | 0.710 |
| 12 | Random Forest | 3.015 | 2.095 | 2.657 | 0.515 | 0.518 |
| 12 | OLS | 3.094 | 1.661 | 2.106 | 0.030⁻ | 0.291 |
| 12 | ARIMA(2,0,2) | 4.222 | 1.801 | 2.283 | 0.108 | 0.310 |
| 12 | SARIMA | 5.129 | 1.937 | 2.456 | 0.109 | 0.307 |

¹ At h = 12 `naive_last` and the seasonal naive coincide by construction (the value one full season back *is* the last observation at a 12-month origin), so `naive_last`'s DM test against the benchmark is degenerate (statistic exactly 0, p = 1).

Negative DM statistics favour the model; positive statistics mark models *worse* than the benchmark. Under the primary bandwidth OLS at unemployment h=12 is nominally significantly **worse** than the seasonal naive (raw p = 0.030) — a fact the former lag-1-only specification (p = 0.29) hid.

Under the family correction (0.05/28 ≈ 0.0018 per scoring window; §Experimental Design), the seven inflation h=1 rejections survive the primary-bandwidth test; every unemployment h=1 rejection (raw p ≈ 0.022) and the OLS-worse-than-benchmark result do not. Under the lag-1 sensitivity no rejection survives.

**Structural-break window (excluding 2020-02 → 2020-12):** the qualitative picture sharpens. For unemployment h=1 the naive last-value forecast becomes almost exact (RMSE 0.136, MASE 0.122; raw DM p = 0.086 vs the benchmark); for inflation h=1 SARIMA remains the best model (RMSE 0.223, raw p = 0.0001 primary, 0.015 lag-1); at h=12 nothing changes qualitatively — no model beats the seasonal naive for either series (OLS at unemployment h=12 is again nominally significantly worse, raw p = 0.013).

## Discussion

1. **Short horizons reward structure; long horizons reward humility.** Inflation at h=1 has strong serial and seasonal dynamics that SARIMA exploits — its MASE of 0.123 means one-eighth the absolute error of the seasonal naive, and the DM test agrees (raw p ≈ 0.0001 under the primary h−1 bandwidth, 0.014 under the lag-1 sensitivity). At h=12 the information in 12 months of history does not support beating "last year's value this month" for either series. This is consistent with the empirical macro-forecasting literature: year-ahead macro point forecasts are dominated by naive benchmarks on revised data.
2. **Machine learning did not add value here, honestly measured.** The distinction that matters is *which baseline*. Against the very weak seasonal-naive benchmark, the tree ensembles — like every non-benchmark model — clear the bar at inflation h=1 (raw p ≈ 0.0001 primary, 0.016 lag-1). Against the strong `naive_last` baseline they win nowhere: `naive_last` has the best unemployment h=1 MASE (0.470 vs 0.512 GBM / 0.492 RF), matches them for inflation h=1, and is unbeatable ex-COVID. At unemployment h=12 they are severely degraded (MASE 2.57 and 2.66 for GBM/RF) and their unemployment h=1 raw significance (p ≈ 0.022) fails the 28-comparison family correction (lag-1: p ≈ 0.08, not significant). With 600 training months (1960-01 → 2009-12), flexible learners mostly model noise. Notably, the *linear* model with lagged features (OLS) is the only ML-family model that is ever nominally best (inflation h=12) — a textbook illustration of the bias–variance trade-off in small-sample macro data.
3. **Persistence is the real benchmark for unemployment.** Outside the COVID window, monthly unemployment is so persistent that the last value is nearly optimal at h=1 (MASE 0.122). The GBM's nominally better RMSE in the full window comes almost entirely from correctly "predicting" the direction of the COVID spike and recovery — i.e., one event. This is precisely the kind of single-event overfitting that DM testing refuses to certify: raw p = 0.022 under the primary bandwidth (which fails the family correction) and p = 0.081 under the lag-1 sensitivity.
4. **Multiple testing and bandwidth choice both matter.** The SARIMA inflation h=1 result has raw p ≈ 0.0001 under the structurally correct h−1 bandwidth and survives a 28-comparison Bonferroni/Holm correction (threshold 0.0018); under the fixed lag-1 sensitivity its raw p = 0.014 and it does not survive. Both specifications are reported because the conclusion is bandwidth-dependent: statistical significance here is robust to multiple testing under the primary bandwidth but not under the sensitivity — and, economically, the gain is confined to one series at one short horizon and does not extend to beating `naive_last` anywhere else. The defensible claim remains "suggestive evidence that seasonal structure helps short-horizon inflation forecasting", now with the correction-robustness stated per specification rather than asserted once.
5. **Divergence failures of integrated models are real.** Anchored immediately after the 2015 oil-crash disinflation, the selected SARIMA extrapolates an explosive negative 12-month path; the rolling-origin design (re-fitting at every origin) contains such failures instead of relying on a single extrapolation, and the h=12 results honestly reflect the resulting instability (SARIMA h=12 unemployment MASE 2.46).

## Limitations

- **Revised, contemporaneous data — not a real-time simulation.** The study uses (a) final *revised* FRED values rather than real-time vintages, and (b) a *contemporaneous* information set: the supervised features include the target's own origin-month value (`y_l0`), and cross-series features enter lagged one month. In real time the CPI observation for the reference month is published ≈2–3 weeks after that month ends and the unemployment rate ≈4–5 weeks after, so at a forecast origin at the start of month *t* a real-time forecaster would not yet know *y_t* — and the most recent available observations would be preliminary estimates, later revised. The information set here is therefore strictly richer than any real-time forecaster's, and the reported accuracy is an **upper bound** on real-time performance. This is not a real-time/vintage simulation; vintage analysis with ALFRED is future work.
- **Sample size and power.** 600 training months (1960-01 → 2009-12), 72 validation months, and 43 scored test months give the DM tests limited power. At h=12 the structurally correct h−1 bandwidth is conservative (wider bandwidth → larger variance estimate → fewer rejections), while the fixed lag-1 sensitivity is anti-conservative there; "not significant" must not be read as "no difference" under either specification.
- **Evaluation step.** Scoring every 3rd test month trades resolution for compute; it is applied identically in prediction and scoring and is a standard design (Tashman 2000), but it does reduce the effective number of independent loss observations.
- **Fixed model set.** No state-space, TVP, regime-switching, or Bayesian shrinkage models; the ML family is deliberately standard.
- **Two targets.** INDPRO, FEDFUNDS and M2SL serve only as features.
- **Point forecasts only.** No density or scenario forecasts.

## Conclusion

Under strict chronological validation with honest baselines, the study finds: (i) short-horizon inflation gains from seasonal ARIMA structure that are statistically robust under the structurally correct HAC bandwidth (raw DM p ≈ 0.0001, surviving a 28-comparison family correction) but not under the fixed lag-1 sensitivity (raw p = 0.014) — an economically narrow result confined to one series at one short horizon; (ii) no significant gains over the seasonal naive at the 12-month horizon for either series under either bandwidth; (iii) no measurable value added by Random Forest or Gradient Boosting beyond clearing the very weak seasonal-naive bar at inflation h=1 — neither beats the strong `naive_last` baseline anywhere; and (iv) an unemployment process so persistent outside crises that naive persistence is nearly unbeatable at h=1. All of this is measured on final revised data with a contemporaneous information set, i.e. an upper bound on real-time accuracy. The methodological contribution is a fully tested, leakage-audited evaluation harness that future projects in this portfolio reuse for financial time series.

## Future Research

- Real-time vintage evaluation (ALFRED) to quantify the combined revision and publication-lag penalty.
- Regime-switching and time-varying-parameter models for structural breaks.
- Density forecasting and recession-probability classification.
- Multi-country panel (World Bank data) and pooled forecasting.
- Conformal prediction intervals for the ML family.

## References

- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Hamilton, J. D. (1994). *Time Series Analysis.* Princeton University Press.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437–450.
