# EconoForecast: Global Inflation Forecasting

**Final Year Project, BSc (Hons) Computer Science (Data Analytics), Asia Pacific University, 2026**

A forecasting system that predicts annual inflation across countries from World Bank macroeconomic indicators. It compares machine-learning models with classical time-series models and serves the results in an interactive Streamlit dashboard for analysts and policymakers.

## Data

The raw indicator files in [`data/`](data) come from the World Bank and cover **266 economies (countries and regional aggregates) from 1960 to 2024**:

| File | Indicator |
| --- | --- |
| `inflation.csv` | Inflation, consumer prices (annual %), the target |
| `GDP Deflator.csv` | GDP deflator annual growth |
| `Real Exchange_Rate.csv` | Real effective exchange rate |
| `Real Interest_Rate.csv` | Real interest rate |
| `Unemployment.csv` | Unemployment (% of labour force) |

Two feature sets were built: a **base** set from the raw indicators, and an **extended** set with engineered macroeconomic features, such as lags and changes. The target was clipped to limit the influence of hyperinflation episodes.

## Models and results

Six model variants were trained and compared on a held-out test set ([`notebooks/Models Result/Final_Model_Comparison.csv`](notebooks/Models%20Result/Final_Model_Comparison.csv)):

| Rank | Model | MAE (pp) | RMSE (pp) | R² |
| --- | --- | --- | --- | --- |
| 1 | **Random Forest, extended features** | **3.64** | **9.59** | **0.56** |
| 2 | Random Forest, base features | 4.06 | 10.55 | 0.47 |
| 3 | XGBoost, extended features | 4.54 | 11.88 | 0.33 |
| 4 | XGBoost, base features | 4.63 | 12.42 | 0.26 |
| 5 | ARIMA(2,1,2) | 3.11 | 3.81 | −8.69 |
| 6 | VAR(1) | unstable | unstable | — |

**What the comparison shows:**
- **Feature engineering paid off.** The extended features raised Random Forest R² from 0.47 to 0.56 and lowered MAE by 10%. XGBoost improved the same way.
- **Random Forest beat XGBoost** on this noisy, heavy-tailed target after hyperparameter tuning ([`Hyperparameter_Tuning_Analysis.csv`](notebooks/Models%20Result/Hyperparameter_Tuning_Analysis.csv)).
- **The time-series baselines did not generalise across countries.** ARIMA has a low MAE but a strongly negative R², meaning it predicts the level but not the variation between countries. VAR(1) became numerically unstable on the panel. These results are reported as they came out, because a failed baseline is a useful finding.

## Dashboard

[`src/app.py`](src/app.py) is a multi-page Streamlit application with:
- **Forecasting:** country-level inflation forecasts five years ahead by default, under different economic assumption scenarios (for example conservative or optimistic), with the best model chosen per country
- **Model performance:** side-by-side metrics and charts for all models
- **Global comparison and analytics:** cross-country rankings, risk categories and correlation heatmaps of the indicators
- **Live economic feed:** economic news from RSS feeds and market data from Yahoo Finance
- **Reports:** a downloadable PDF report (ReportLab) and CSV export of forecasts
- **Policy notes:** rule-based recommendations from the forecast inflation level

## Repository structure

```
data/                      raw World Bank indicator files
notebooks/
  Random_Forest/           base and extended Random Forest models
  XGBoost/                 base and extended XGBoost models
  TimeSeries/              ARIMA and VAR
  Models Result/           saved models, tuning results, final comparison
src/app.py                 Streamlit dashboard
```

The dashboard loads the merged datasets and trained models produced by the notebooks (the paths are set at the top of `src/app.py`).

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels (ARIMA, VAR), Streamlit, Plotly, ReportLab, feedparser, yfinance

## Run it

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Methodology

The project followed **CRISP-DM**: business understanding (what policymakers need from an inflation forecast), data understanding and preparation, modelling, evaluation, and deployment as a dashboard.

---
Author: **Abakar Sougui Fatime** · [GitHub](https://github.com/Abakar-Fatime) · [LinkedIn](https://linkedin.com/in/abakar-sougui-fatime)
