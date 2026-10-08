# macro-forecasting-lab — experiment summary

_Generated automatically by the pipeline. Data coverage: 1960-01-01 .. 2026-08-01 (800 months)._

Split: train <= 2009-12, validation 2010-01..2015-12, test 2016-01..latest.
Windows: `full` = whole test window; `ex_covid` = test window excluding 2020-02..2020-12 (COVID-19 structural break).

## inflation_yoy

- ADF p = 0.03874, KPSS p = 0.01 (train+val) -> integration order d = 1.
- Selected ARIMA(1, 1, 2); SARIMA(2, 1, 2)(1, 0, 1, 12) (AIC on train+val).

### inflation_yoy — h=1, window=full

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| sarima | 43 | 0.246 | 0.171 | 11.61 | 0.123 | -2.45 | 0.014 |
| arima | 43 | 0.362 | 0.273 | 15.54 | 0.195 | -2.44 | 0.015 |
| ols | 43 | 0.368 | 0.279 | 16.14 | 0.199 | -2.43 | 0.015 |
| naive_last | 43 | 0.394 | 0.271 | 18.48 | 0.194 | — | — |
| gradient_boosting | 43 | 0.399 | 0.273 | 18.90 | 0.195 | -2.41 | 0.016 |
| random_forest | 43 | 0.415 | 0.276 | 19.06 | 0.198 | -2.41 | 0.016 |
| rolling_mean_12 | 43 | 1.261 | 0.887 | 39.44 | 0.635 | -2.29 | 0.022 |
| seasonal_naive | 43 | 2.181 | 1.597 | 63.49 | 1.142 | — | — |

### inflation_yoy — h=1, window=ex_covid

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| sarima | 40 | 0.223 | 0.158 | 5.98 | 0.113 | -2.44 | 0.015 |
| arima | 40 | 0.350 | 0.267 | 9.62 | 0.191 | -2.42 | 0.015 |
| ols | 40 | 0.354 | 0.270 | 9.66 | 0.193 | -2.42 | 0.015 |
| naive_last | 40 | 0.361 | 0.251 | 9.48 | 0.180 | — | — |
| gradient_boosting | 40 | 0.361 | 0.257 | 9.85 | 0.183 | -2.41 | 0.016 |
| random_forest | 40 | 0.380 | 0.254 | 9.59 | 0.182 | -2.41 | 0.016 |
| rolling_mean_12 | 40 | 1.278 | 0.894 | 27.61 | 0.639 | -2.31 | 0.021 |
| seasonal_naive | 40 | 2.241 | 1.642 | 51.83 | 1.175 | — | — |

- **Best model by full-test RMSE:** `sarima` (RMSE 0.246, MAE 0.171, MASE 0.123).
- **Significantly better than seasonal-naive at 5% (DM, full window):** `rolling_mean_12`, `arima`, `sarima`, `ols`, `random_forest`, `gradient_boosting`.

### inflation_yoy — h=12, window=full

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| ols | 43 | 1.652 | 1.344 | 58.97 | 0.961 | -1.50 | 0.133 |
| sarima | 43 | 1.888 | 1.438 | 67.05 | 1.029 | -1.17 | 0.241 |
| arima | 43 | 2.156 | 1.593 | 64.59 | 1.139 | -0.55 | 0.584 |
| naive_last | 43 | 2.181 | 1.597 | 63.49 | 1.142 | — | — |
| seasonal_naive | 43 | 2.181 | 1.597 | 63.49 | 1.142 | — | — |
| gradient_boosting | 43 | 2.442 | 1.579 | 66.51 | 1.130 | +0.71 | 0.477 |
| random_forest | 43 | 2.513 | 1.627 | 61.82 | 1.164 | +0.95 | 0.345 |
| rolling_mean_12 | 43 | 2.521 | 1.816 | 66.43 | 1.299 | +1.08 | 0.281 |

### inflation_yoy — h=12, window=ex_covid

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| ols | 40 | 1.682 | 1.360 | 46.17 | 0.973 | -1.52 | 0.128 |
| sarima | 40 | 1.891 | 1.414 | 45.16 | 1.011 | -1.36 | 0.173 |
| arima | 40 | 2.213 | 1.634 | 52.72 | 1.169 | -0.57 | 0.568 |
| naive_last | 40 | 2.241 | 1.642 | 51.83 | 1.175 | — | — |
| seasonal_naive | 40 | 2.241 | 1.642 | 51.83 | 1.175 | — | — |
| gradient_boosting | 40 | 2.485 | 1.577 | 45.53 | 1.128 | +0.63 | 0.526 |
| random_forest | 40 | 2.579 | 1.654 | 47.22 | 1.183 | +0.92 | 0.357 |
| rolling_mean_12 | 40 | 2.590 | 1.866 | 52.58 | 1.335 | +1.06 | 0.289 |

- **Best model by full-test RMSE:** `ols` (RMSE 1.652, MAE 1.344, MASE 0.961).
- **Significantly better than seasonal-naive at 5% (DM, full window):** none.

## unemployment_rate

- ADF p = 0.03232, KPSS p = 0.1 (train+val) -> integration order d = 0.
- Selected ARIMA(2, 0, 2); SARIMA(2, 0, 2)(1, 0, 0, 12) (AIC on train+val).

### unemployment_rate — h=1, window=full

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| gradient_boosting | 43 | 1.595 | 0.404 | 5.23 | 0.512 | -1.75 | 0.081 |
| naive_last | 43 | 1.602 | 0.371 | 4.20 | 0.470 | — | — |
| random_forest | 43 | 1.607 | 0.388 | 4.60 | 0.492 | -1.75 | 0.080 |
| arima | 43 | 1.615 | 0.398 | 4.68 | 0.504 | -1.75 | 0.080 |
| sarima | 43 | 1.617 | 0.408 | 4.84 | 0.517 | -1.75 | 0.080 |
| ols | 43 | 1.624 | 0.440 | 5.77 | 0.558 | -1.74 | 0.082 |
| rolling_mean_12 | 43 | 1.935 | 0.750 | 11.26 | 0.951 | -1.59 | 0.112 |
| seasonal_naive | 43 | 2.685 | 1.355 | 23.00 | 1.718 | — | — |

### unemployment_rate — h=1, window=ex_covid

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| naive_last | 40 | 0.136 | 0.096 | 2.23 | 0.122 | — | — |
| arima | 40 | 0.152 | 0.112 | 2.63 | 0.142 | -1.32 | 0.187 |
| random_forest | 40 | 0.163 | 0.118 | 2.70 | 0.149 | -1.32 | 0.188 |
| sarima | 40 | 0.182 | 0.125 | 2.82 | 0.158 | -1.32 | 0.187 |
| ols | 40 | 0.196 | 0.155 | 3.81 | 0.197 | -1.31 | 0.190 |
| gradient_boosting | 40 | 0.219 | 0.152 | 3.57 | 0.192 | -1.31 | 0.191 |
| rolling_mean_12 | 40 | 0.694 | 0.413 | 9.05 | 0.524 | -1.27 | 0.202 |
| seasonal_naive | 40 | 1.828 | 0.934 | 20.06 | 1.184 | — | — |

- **Best model by full-test RMSE:** `gradient_boosting` (RMSE 1.595, MAE 0.404, MASE 0.512).
- **Significantly better than seasonal-naive at 5% (DM, full window):** none.

### unemployment_rate — h=12, window=full

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| rolling_mean_12 | 43 | 2.448 | 1.419 | 26.95 | 1.799 | -0.53 | 0.596 |
| naive_last | 43 | 2.685 | 1.355 | 23.00 | 1.718 | — | — |
| seasonal_naive | 43 | 2.685 | 1.355 | 23.00 | 1.718 | — | — |
| gradient_boosting | 43 | 2.834 | 2.025 | 41.80 | 2.567 | +0.37 | 0.710 |
| random_forest | 43 | 3.015 | 2.095 | 44.65 | 2.657 | +0.65 | 0.518 |
| ols | 43 | 3.094 | 1.661 | 31.08 | 2.106 | +1.06 | 0.291 |
| arima | 43 | 4.222 | 1.801 | 32.30 | 2.283 | +1.02 | 0.310 |
| sarima | 43 | 5.129 | 1.937 | 34.43 | 2.456 | +1.02 | 0.307 |

### unemployment_rate — h=12, window=ex_covid

| model | n | RMSE | MAE | MAPE % | MASE | DM stat | DM p |
|---|---|---|---|---|---|---|---|
| rolling_mean_12 | 40 | 1.463 | 1.010 | 24.38 | 1.281 | -0.49 | 0.627 |
| naive_last | 40 | 1.828 | 0.934 | 20.06 | 1.184 | — | — |
| seasonal_naive | 40 | 1.828 | 0.934 | 20.06 | 1.184 | — | — |
| gradient_boosting | 40 | 2.101 | 1.669 | 40.44 | 2.117 | +0.45 | 0.650 |
| random_forest | 40 | 2.363 | 1.756 | 43.66 | 2.227 | +0.72 | 0.470 |
| ols | 40 | 2.483 | 1.285 | 28.98 | 1.630 | +1.20 | 0.231 |
| arima | 40 | 3.896 | 1.447 | 30.42 | 1.835 | +1.06 | 0.289 |
| sarima | 40 | 4.932 | 1.597 | 32.74 | 2.025 | +1.05 | 0.294 |

- **Best model by full-test RMSE:** `rolling_mean_12` (RMSE 2.448, MAE 1.419, MASE 1.799).
- **Significantly better than seasonal-naive at 5% (DM, full window):** none.

## Honesty notes

- DM p-values are raw; with 7 models x 2 series x 2 horizons compared against the
  same benchmark, a Bonferroni-corrected threshold (~0.05/14 ≈ 0.0036 per window)
  is the appropriate reading — see README §14.
- MAPE is unstable for inflation near zero (deflation episodes); rely on MASE.
- MASE is scaled by the in-sample seasonal-naive MAE on the training window
  (Hyndman & Athanasopoulos, §5.8).
