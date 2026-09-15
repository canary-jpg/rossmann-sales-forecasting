"""
Shared preprocessing for Rossmann dataset - merging train/store data
and handling known data quality issues. Used by notebooks 02-05 and 
by the deployed API (src/predict.py), so cleaning logic can't drift
between training and inference 
"""

import pandas as pd 

def fix_state_holiday(df: pd.DataFrame) -> pd.DataFrame:
    """StateHoliday is documented to mix the string "0" with other 
    representations for "not a holiday" - normalize to a consistent string
    type so downstream grouping/encoding doesn't siliently split
    "not a holiday" into multiple distinct categories """

    df = df.copy()
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['StateHoliday'] = df['StateHoliday'].replace({"0.0": "0"})
    return df 

def merge_store_data(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """Left merge store metadata onto sales data by Store ID"""
    return sales_df.merge(store_df, on="Store", how="left")

def handle_missing_competition_distance(df: pd.DataFrame) -> pd.DataFrame:
    """CompetitionDistance has missing values (store.csv). Median-impute and 
    flag, same pattern as Project 1's MonthlyIncome handling - a 
    missing competitor distance may itself carry signal (e.g. truly
    isolated stores vs unreported data) rather than being pure noise"""
    df = df.copy()
    median_distance = df["CompetitionDistance"].median()
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(median_distance)
    return df 

def filter_closed_days(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where Open==0 have Sales trivially at 0 - including them
    in training would teach a model to predict 0 on closed days, which
    isn't a meaningful forecasting signal (a store's closure schedule is
    know in advance, not something to forecast). Excluding them here; the
    "is this store open today" question is a separate, simpler problem
    then "what will sales be given the store is open."""
    return df[df['Open'] == 1].copy()

def clean_and_merge(sales_df: pd.DataFrame, store_df: pd.DataFrame) -> pd.DataFrame:
    """full pipeline: merge, fix StateHoliday, handle missing competition
    distance, filter closed days. Order matters - filter closed days last,
    after merging, so store metadata is available for all rows during the 
    earlier steps"""
    df = merge_store_data(sales_df, store_df)
    df = fix_state_holiday(df)
    df = handle_missing_competition_distance(df)
    return df 