"""
Shared feature engineering for the ML approach (notebook 04)
and the deployed API. Kept separate from preprocessing.py since 
these are model-input features built on top of already-cleaned data,
not data cleaning itself.
IMPORTANT: lag and rolling features require historical data per store -
at inference/deployment time, this means the API needs access to a
store's recent sales history, not just the date being forecast. This is
a real, meaningful constraint worth understanding before deployment: this
is fundamentally different than Projects 1 and 2, where a single row of
input features was self-contained. See notebook 04's discussion of this.
"""

import pandas as pd 

def create_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Calendar features- day of week, month, year, week of year, 
    weekend flag. Cheap, standard and often surprisely predictive for
    retail sales (weekday patterns, end-of-month effects, etc.)
    """
    df  = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['DayOfWeek_num'] = df[date_col].dt.dayofweek 
    df['Month'] = df[date_col].dt.month 
    df['Year'] = df[date_col].dt.year 
    df['WeekOfYear'] = df[date_col].dt.isocalendar().week.astype(int)
    df['IsWeekend'] = (df['DayOfWeek_num'] >= 5).astype(int)
    return df 

def create_lag_features(
    df: pd.DataFrame,
    group_col: str = 'Store',
    target_col: str = 'Sales',
    date_col: str = "Date",
    lags=(1,7,14,28),
) -> pd.DataFrame:
    """per store lagged sales values - e.g. Sales_lag_7 is what this
    store sold exactly one week ago. Requires data sorted by date within each
    store group, or lags will be computed against the wrong rows."""
    df = df.sort_values([group_col, date_col]).copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    return df 

def create_rolling_features(
    df: pd.DataFrame,
    group_col: str = 'Store',
    target_col: str = 'Sales',
    date_col: str = 'Date',
    windows=(7,28),
) -> pd.DataFrame:
    """Per store rolling mean sales - smoothed recent trend. Shifted
    by 1 first os the rolling window doesn't include the current day's own
    sales values (that would be leaking the target into a feature)"""
    df = df.sort_values([group_col, date_col]).copy()
    for window in windows:
        df[f"{target_col}_rolling_mean_{window}"] = (
            df.groupby(group_col)[target_col]
            .shift(1)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df 

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """full feature engineering pipeline: date parts, lags,
    rolling means. Note: lag/rolling features will be NaN for each store's
    earliest rows (not enough history yet) - these rows need to be dropped
    before training, not imputed, since imputing a lag feature would be
    fabricating history that doesn't exist"""

    df = create_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    return df 
