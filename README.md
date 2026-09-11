# Rossmann Store Sales Forecasting

Forecasting daily sales for Rossmann drugstores using historical sales, store metadata, promotions, and holiday data.

## Business framing

Rossmann store managers currently forecast their own sales up to six weeks ahead, with accuracy varying a lot from manager to manager. A single, consistent forecasting model would let the business plan staffing, inventory, and promotions more reliably across all 1,115 stores. This project builds and compares several forecasting approaches - from simple baselines to a feature-engineered ML model and evaluates them the way a real forecasting problem demands: using a chronological (not random) held-out test period.

## Why this project (vs. Projects 1 and 2)

Project 1 (loan default) was tabluar binary classification. Project 2 (Yelp Reviews) was multi-class NLP. This project is deliberately different again: time series forecasting, with its own distinct correctness traps - most importantly, **why a random train/test split is wrong here** (it leaks future information into training) in a way it isn't for the previous two projects.

## Project structure

```
ossmann-sales-forecasting/
├── README.md
├── data/
│   ├── raw/                 # train.csv, store.csv — not committed, see data/raw/README.md
│   └── processed/            # cleaned, merged, feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb                          # trends, seasonality, data quality
│   ├── 02_baselines_time_split.ipynb          # naive baselines + correct time-aware splitting
│   ├── 03_statistical_model.ipynb             # Prophet/SARIMA
│   ├── 04_ml_xgboost_features.ipynb           # XGBoost with lag/rolling/date features
│   └── 05_model_comparison_evaluation.ipynb   # final comparison, error analysis
├── src/
│   ├── preprocessing.py     # shared cleaning (StateHoliday fix, store merge, missing data)
│   ├── features.py          # shared feature engineering (lags, rolling windows, date parts)
│   ├── eval_utils.py        # shared results tracker across notebooks
│   └── predict.py           # loads the final model, runs inference
├── app/
│   ├── main.py       # FastAPI service exposing POST /predict
│   └── Dockerfile
├── models/            # saved model artifacts
├── tests/
│   └── test_preprocessing.py
├── requirements.txt       # full, for local dev
├── requirements-app.txt   # lean, for deployment
└── .gitignore

```

## Status 
- [] EDA
- [] Baselines + time-aware train/test split
- [] Statitical model (Prophet/SARIMA)
- [] ML model (XGBoost + engineered features)
- [] FastAPI service
- [] Dockerized + deployed
- [] Write-up

## Results

## Limitations & next steps

## Setup

Two requirements files, same pattern as Project 1 and 2:
- `requirements.txt` - full set, for local development
- `requirements-app.txt` - lean set, for deployment

```bash
python -m venv venv
source venv/bin/activate # or venv/Scripts/activate on Windows
pip install -r requirements.txt
```

Download the data (see `data/raw/README.md`) before running notebooks.