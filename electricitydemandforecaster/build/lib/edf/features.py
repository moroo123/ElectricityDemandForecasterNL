import pandas as pd
import holidays
import numpy as np


def build_time_features(df: pd.DataFrame, ):
    """Build time-based features for the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Raises:
        TypeError: If the index of the dataframe is not a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with time-based features added.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Index of dataframe must be pd.DatetimeIndex, got {type(df.index).__name__}")

    df_return = pd.DataFrame(index=df.index)
    df_return['sin_hour'] = np.sin(2*np.pi*df.index.hour / 24)
    df_return['cos_hour'] = np.cos(2*np.pi*df.index.hour / 24)
    df_return['sin_dayofweek'] = np.sin(2*np.pi*(df.index.weekday) / 7)
    df_return['cos_dayofweek'] = np.cos(2*np.pi*(df.index.weekday) / 7)
    df_return['dayofmonth'] = df.index.day
    df_return['sin_month'] = np.sin(2*np.pi*(df.index.month-1) / 12)
    df_return['cos_month'] = np.cos(2*np.pi*(df.index.month-1) / 12)
    df_return['sin_dayofyear'] = np.sin(2*np.pi*(df.index.dayofyear-1) / 365)
    df_return['cos_dayofyear'] = np.cos(2*np.pi*(df.index.dayofyear-1) / 365)
    df_return['year'] = df.index.year
    return df_return


def build_holiday_feature(df: pd.DataFrame):
    """Build holiday feature for the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: DataFrame with holiday feature added.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Index of dataframe must be pd.DatetimeIndex, got {type(df.index).__name__}")

    df_return = pd.DataFrame(index=df.index)
    years = df.index.year.unique()

    # Get dictionary of holidays in NL
    nl_holidays = holidays.Netherlands(years=years)
    # Get holiday dates as pd datetime
    nl_holidays_datetime = pd.to_datetime(list(nl_holidays.keys()))

    df_return['is_holiday'] = df.index.normalize().isin(nl_holidays_datetime)
    return df_return


def build_lagroll_features(df: pd.DataFrame, target_column: str, lags: list[int] | None, rolls: list[int] | None):
    """Build lagging and rolling features for the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target column to create lagging/rolling features for.
        lags (list[int]): List of lag periods.
        rolls (list[int]): List of rolling window sizes.

    Returns:
        pd.DataFrame: DataFrame with lagging and rolling features added.
    """
    df_return = pd.DataFrame(index=df.index)

    lags = [] if lags is None else lags
    rolls = [] if rolls is None else rolls

    # Create lagging features
    for lag in lags:
        df_return[f'lag_{lag}'] = df[target_column].shift(lag)

    # Create rolling features
    for roll in rolls:
        df_return[f'roll_mean_{roll}'] = df[target_column].shift(1).rolling(
            roll).mean()
        df_return[f'roll_std_{roll}'] = df[target_column].shift(
            1).rolling(roll).std()

    return df_return


def align_weather_to_index(df_weather: pd.DataFrame, index: pd.DatetimeIndex, method: str = 'ffill', tolerance: str = '45min'):
    """Align weather dataframe to the given index by reindexing and interpolating.

    Args:
        df_weather (pd.DataFrame): Input weather dataframe.
        index (pd.DatetimeIndex): Target index to align to.
        method (str): Interpolation method to use (default is 'ffill').
        tolerance (str): Maximum distance between original and interpolated index (default is '45min').
    Raises:
        TypeError: If the index of the weather dataframe is not a DatetimeIndex.

    Returns:
        pd.DataFrame: Aligned weather dataframe.
    """
    if not isinstance(df_weather.index, pd.DatetimeIndex):
        raise TypeError(
            f"Index of dataframe must be pd.DatetimeIndex, got {type(df_weather.index).__name__}")

    df_w = df_weather.sort_index().copy()
    target_index = index.sort_values().copy()

    if method == 'ffill':
        df_weather_aligned = df_weather.reindex(target_index, method='ffill')
    elif method == 'nearest':
        df_weather_aligned = df_weather.reindex(
            target_index, method='nearest', tolerance=pd.Timedelta(nearest_tolerance))

    return df_weather_aligned


def build_feature_dataframe(df: pd.DataFrame, target_column: str, df_weather: pd.DataFrame | None = None,  lags: list[int] | None = None, rolls: list[int] | None = None):
    """Build feature and target dataframes from the input dataframe.

    Args:
        df(pd.DataFrame): Input dataframe containing the features and target.
        target_column(str): Name of the target column.
        lags(list[int] | None): List of lag periods.
        rolls(list[int] | None): List of rolling window sizes.
    Returns:
        tuple: A tuple containing the feature dataframe(X) and target dataframe(y).
    """

    df_time_features = build_time_features(df)
    df_holiday_features = build_holiday_feature(df)
    df_lagroll_features = build_lagroll_features(
        df, target_column, lags, rolls)

    if df_weather is not None:
        df_weather = align_weather_to_index(
            df_weather, df.index, method='ffill')

        X = pd.concat([df_time_features, df_holiday_features,
                       df_lagroll_features, df_weather], axis=1)
    else:
        X = pd.concat([df_time_features, df_holiday_features,
                       df_lagroll_features], axis=1)
    print(X.columns)
    y = df[[target_column]].copy()

    combined = pd.concat([X, y], axis=1).dropna()

    X = combined.drop(columns=[target_column])
    y = combined[[target_column]]

    return X, y
