

import numpy as np
import pandas as pd
from typing import Dict, List
from xgboost import XGBRegressor
from edf.metrics import compute_metrics
from itertools import product
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm


class SeasonalNaiveBaseline:
    def __init__(self, seasonality: int = 24):
        """Seasonal Naive Baseline model for time series forecasting.

        Args:
            seasonality (int): The seasonal period.
        """
        self.seasonality = seasonality
        self.train_series = None

    def fit(self, y: pd.Series):
        self.train_series = y.sort_index()
        print(len(self.train_series))

    def predict(self, forecast_h: pd.DatetimeIndex):
        """Predict the future values using the seasonal naive baseline.

        Args:
            forecast_h (pd.DatetimeIndex): The forecast horizon.
        """
        aligned = self.train_series.reindex(
            self.train_series.index.union(forecast_h)
        ).sort_index()

        predictions = aligned.shift(self.seasonality).reindex(forecast_h)

        return predictions


class XGBoostBaseline:
    def __init__(self, lags: List[int], horizons: List[int], rolling_windows: List[int]):
        """XGBoost Baseline model for time series forecasting.

        Args:
            lags (List[int]): List of lagged features to create.
            horizons (List[int]): List of forecast horizons.
        """
        self.lags = lags
        self.horizons = horizons
        self.rolling_windows = rolling_windows
        self.models = {}
        self.feature_cols: dict[int, List[str]] = {}
        self.feature_column: str = None  # Main feature column

    def create_lag_features(self, df: pd.DataFrame, column_name: str):
        df_features = df.copy()
        for lag in self.lags:
            df_features[f'lag_{lag}'] = df_features[column_name].shift(lag)
        return df_features

    def create_horizon_features(self, df: pd.DataFrame, column_name: str):
        df_horizon = df.copy()
        for horizon in self.horizons:
            df_horizon[f'horizon_{horizon}'] = df_horizon[column_name].shift(
                -horizon)
        return df_horizon

    def create_rolling_features(self, df: pd.DataFrame, column_name: str):
        df_rolling = df.copy()
        for window in self.rolling_windows:
            df_rolling[f'rolling_mean_{window}'] = df_rolling[column_name].shift(1).rolling(
                window).mean()
            df_rolling[f'rolling_std_{window}'] = df_rolling[column_name].shift(1).rolling(
                window).std()
        return df_rolling

    def create_time_features(self, df: pd.DataFrame):
        assert isinstance(
            df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
        df_time = df.copy()
        df_time['year'] = df_time.index.year
        df_time['month'] = df_time.index.month
        df_time['day'] = df_time.index.day
        df_time['hour'] = df_time.index.hour
        df_time['dow'] = df_time.index.dayofweek
        # cyclical hour encodings (help trees a bit)
        df_time['sin_h'] = np.sin(2 * np.pi * df_time['hour'] / 24)
        df_time['cos_h'] = np.cos(2 * np.pi * df_time['hour'] / 24)
        return df_time

    def create_features(self, df: pd.DataFrame, column_name: str):
        """Create all features for the XGBoost model."""
        df = self.create_lag_features(df, column_name)
        df = self.create_rolling_features(df, column_name)

        df = self.create_time_features(df)

        return df

    def prepare_dataset(self, df: pd.DataFrame, horizon: int, feature_column: str):
        df = df.copy()

        keep_target = f"horizon_{horizon}"

        df = df.drop(columns=[c for c in df.columns
                              if c.startswith("horizon_") and c != keep_target])

        # Remove head and tail
        head = max(self.lags + self.rolling_windows + [0])
        tail = horizon
        if tail > 0:
            df = df.iloc[head: len(df) - tail]
        else:
            df = df.iloc[head:]

        feature_cols = [c for c in df.columns
                        if not c.startswith("horizon_")
                        and c != feature_column]
        # Drop NaNs only on the columns we actually use
        df = df.dropna(subset=feature_cols + [keep_target], how="any")

        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[keep_target].to_numpy(dtype=np.float32).ravel()
        return X, y

    def train(self, df: pd.DataFrame, train_frac: float, feature_column: str, param_grid: Dict[str, list] | None = None, ):
        """Train the XGBoost model with optional grid search per horizon.

        Args:
            df (pd.DataFrame): The input dataframe.
            train_frac (float): The fraction of data to use for training.
            feature_column (str): The name of the target feature column.
            param_grid (Dict[str, list] | None, optional): The parameter grid for hyperparameter tuning. Defaults to None.

        Raises:
            ValueError: If the index of the dataframe is not a DatetimeIndex.
        """
        self.feature_column = feature_column

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index must be DatetimeIndex")
        df = df.sort_index(ascending=True)

        # Split into train and test sets
        split_idx = int(len(df) * train_frac)
        df_train_raw, df_test_raw = df.iloc[:split_idx].copy(
        ), df.iloc[split_idx:].copy()

        # Build features for each split
        df_train = self.create_features(
            df_train_raw, column_name=feature_column)
        df_test = self.create_features(
            df_test_raw,  column_name=feature_column)

        df_train = self.create_horizon_features(df_train, feature_column)
        df_test = self.create_horizon_features(df_test, feature_column)

        # Define validation fraction
        val_frac = 0.1

        for _, horizon in enumerate(tqdm(self.horizons, desc="Horizon")):
            # Build training+validation for this horizon
            X_full, y_full = self.prepare_dataset(
                df_train, horizon, feature_column=feature_column)

            gap = max(self.lags + self.rolling_windows + [horizon])
            n_tr = int(len(X_full) * (1 - val_frac)) - gap
            X_tr, y_tr = X_full[:n_tr], y_full[:n_tr]
            X_va, y_va = X_full[n_tr+gap:], y_full[n_tr+gap:]

            # Build test set
            X_te, y_te = self.prepare_dataset(
                df_test, horizon, feature_column=feature_column)

            # Base params
            base_params = dict(
                n_estimators=4000,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.9,
                reg_lambda=5.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                eval_metric="mae",
                early_stopping_rounds=200,
            )

            best_params = {}
            best_mae = float("inf")
            best_model = None

            if param_grid:
                # Manual grid over provided dictionary (cartesian product)
                keys = list(param_grid.keys())
                values = [param_grid[k] for k in keys]
                for combo in tqdm(product(*values), total=np.prod([len(v) for v in values]), desc=f"h={horizon}"):
                    params = base_params.copy()
                    params.update({k: v for k, v in zip(keys, combo)})

                    model = XGBRegressor(**params)
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                    )
                    pred_va = model.predict(X_va)
                    mae = mean_absolute_error(y_va, pred_va)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = {k: params[k] for k in keys}
                        best_model = model
            else:
                # No grid: fit once with base params
                best_params = {}
                best_model = XGBRegressor(**base_params)
                best_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                )

            # Refit with best params on full train (recreate a small val tail for early stopping)
            final_params = base_params.copy()
            final_params.update(best_params)
            model = XGBRegressor(**final_params)

            # Use the same val split logic on X_full for stability
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )

            # Evaluate on test
            y_pred = model.predict(X_te)
            metrics = compute_metrics(y_te, y_pred)

            print(
                f"Best params for h={horizon}: {best_params} | Test metrics: {metrics}")

            self.models[horizon] = (model, metrics)

    def predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon not in self.models:
            raise ValueError(f"Model for horizon {horizon} not trained.")
        if self.feature_column is None:
            raise ValueError("feature_column is unknown; call train() first.")

        model, _ = self.models[horizon]

        df_features = self.create_lag_features(df, self.feature_column)
        df_features = self.create_rolling_features(
            df_features, self.feature_column)
        df_features = self.create_time_features(df_features)

        head = max(self.lags + self.rolling_windows + [0])
        df_features = df_features.iloc[head:].copy()

        feature_cols = [c for c in df_features.columns
                        if not c.startswith("horizon_") and c != self.feature_column]

        X = df_features[feature_cols].dropna(axis=0, how="any")
        y_predicted = model.predict(X.to_numpy(dtype=np.float32))

        df_out = df.copy()

        # Add predictions to df
        pred_col = f"predicted_horizon_{horizon}"
        df_out.loc[X.index, pred_col] = y_predicted

        # Add actual values to df
        true_col = f"actual_horizon_{horizon}"
        df_out[true_col] = df_out[self.feature_column].shift(-horizon)

        df_out.dropna(subset=[pred_col, true_col], inplace=True)

        return df_out

    def predict_next(self, df: pd.DataFrame, horizon: int) -> float | np.ndarray:
        if horizon not in self.models:
            raise ValueError(f"Model for horizon {horizon} not trained.")
        if self.feature_column is None:
            raise ValueError("feature_column is unknown; call train() first.")

        model, _ = self.models[horizon]

        df_features = self.create_lag_features(df, self.feature_column)
        df_features = self.create_rolling_features(
            df_features, self.feature_column)
        df_features = self.create_time_features(df_features)

        head = max(self.lags + self.rolling_windows + [0])
        df_features = df_features.iloc[head:].copy()

        feature_cols = [c for c in df_features.columns
                        if not c.startswith("horizon_") and c != self.feature_column]

        X = df_features[feature_cols].dropna(axis=0, how="any")
        if X.empty:
            raise ValueError(
                "Not enough history to build features for prediction.")
        x_last = X.iloc[[-1]].to_numpy(dtype=np.float32)   # last valid row
        return model.predict(x_last)[0]


# %%
