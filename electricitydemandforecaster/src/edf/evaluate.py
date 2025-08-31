import torch
from pathlib import Path
import yaml
import json
from edf.models import MODEL_REGISTRY
import joblib
from edf.inout import read_dataframe_from_sql
from edf.features import build_feature_dataframe
from edf.datasets import ElectricityDemandDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from edf.metrics import compute_metrics
from edf.utils import set_seed
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _require_file(p: Path):
    """Check if a required file exists. If not, raise FileNotFoundError.

    Args:
        p (Path): Path to the file.

    Raises:
        FileNotFoundError: Error if the file does not exist.
    """
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")


def load_artifact(run_dir: Path) -> Tuple[Dict, Dict, torch.nn.Module, StandardScaler, StandardScaler]:
    """Load all artifacts from the run directory. These include:
    - config.yaml: The configuration used for training.
    - split_info.json: Information about the train/validation/test split.
    - scaler_X.pkl: StandardScaler for the input features.
    - scaler_y.pkl: StandardScaler for the target variable.

    Args:
        run_dir (Path): Path to the run directory containing the artifacts.

    Returns:
        Tuple[Dict, Dict, torch.nn.Module, StandardScaler, StandardScaler]: Output containing:
            - config: The configuration dictionary.
            - split_info: The train/validation/test split information.
            - model: The trained model.
            - scaler_X: StandardScaler for input features.
            - scaler_y: StandardScaler for the target variable.
    """
    run_dir = Path(run_dir)
    _require_file(run_dir / 'config.yaml')
    _require_file(run_dir / 'split_info.json')
    _require_file(run_dir / 'scaler_X.pkl')
    _require_file(run_dir / 'scaler_y.pkl')

    with open(run_dir / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    with open(run_dir / 'split_info.json', 'r') as f:
        split_info = json.load(f)

    scaler_X = joblib.load(run_dir / 'scaler_X.pkl')
    scaler_y = joblib.load(run_dir / 'scaler_y.pkl')

    # Build model from config and load weights
    model_config = config['model']
    model_cls = MODEL_REGISTRY[model_config['name']]
    model = model_cls(**{k: v for k, v in model_config.items() if k != 'name'})
    state_dict = torch.load(
        run_dir / (model_config['name'] + '.pt'), map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return config, split_info, model, scaler_X, scaler_y


def run_test(run_dir: Path, plot_horizons: list[int], save_plots: bool = True):
    """Function to run test.

    Args:
        run_dir (Path): Path to the file.
        plot_horizons (list[int]):  List of horizons to plot.
        save_plots (bool, optional): Whether to save the plots. Defaults to True.

    Returns:
        dict: A dictionary containing the evaluation results.
    """

    run_dir = Path(run_dir)
    # Load artifacts
    config, split_info, model, scaler_X, scaler_y = load_artifact(run_dir)
    horizon = config['data']['horizon']
    lookback = config['data']['lookback']
    batch_size = config['train']['batch_size']
    target = config['data']['target']
    lags = config['features']['lags']
    rolls = config['features']['rolls']

    # Set seed
    set_seed(config['seed'])

    # Load the data
    df = read_dataframe_from_sql(db_path=config['data']['source'],
                                 table_name=config['data']['table'],
                                 column_names=config['data']['columns'],
                                 timestamp_col=config['data']['timestamp_col'],
                                 timestamp_format=config['data']['timestamp_format'], )

    if config['data']['weather']['use']:
        df_weather = read_dataframe_from_sql(db_path=config['data']['weather']['source'],
                                             table_name=config['data']['weather']['table'],
                                             column_names=config['data']['weather']['columns'],
                                             timestamp_col=config['data']['weather']['timestamp_col'],
                                             timestamp_format=config['data']['weather']['timestamp_format'], )
    else:
        df_weather = None

    # Create trainval/test split
    df_test = df.iloc[split_info['test']['start_index']:
                      split_info['test']['end_index']+1]

    # Add features
    X_df_test, y_df_test = build_feature_dataframe(
        df_test, target, df_weather=df_weather, lags=lags, rolls=rolls)

    # Convert to numpy
    X_test = X_df_test.to_numpy(dtype='float32')
    y_test = y_df_test.to_numpy(dtype='float32').reshape(-1, 1)

    # Scaler X and y training and validation data
    X_test = scaler_X.transform(X_test)
    y_test = scaler_y.transform(y_test)

    # Create a dataset and dataloader
    test_dataset = ElectricityDemandDataset(
        X_test, y_test, lookback, horizon)
    test_dataloader = DataLoader(
        test_dataset, batch_size, shuffle=False)

    # Evaluate the model
    predictions = []
    trues = []
    with torch.inference_mode():
        for x, y in test_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            predictions.extend(y_pred.detach().cpu().numpy())
            trues.extend(y.detach().cpu().numpy())

    predictions_rescaled = scaler_y.inverse_transform(np.vstack(predictions))
    trues_rescaled = scaler_y.inverse_transform(np.vstack(trues)).reshape(-1,
                                                                          horizon)
    overall, per_h = compute_metrics(trues_rescaled, predictions_rescaled)

    for h in plot_horizons:
        h = int(h)
        if not (1 <= h <= horizon):
            continue
        fig = plt.figure()
        plt.plot(predictions_rescaled[:, h - 1], label=f"Predicted h={h}")
        plt.plot(trues_rescaled[:, h - 1], label=f"Actual h={h}")
        plt.legend()
        plt.title(f"Forecast horizon h={h}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_plots:
            plot_dir = run_dir / "eval_plots"
            plot_dir.mkdir(exist_ok=True)
            out = plot_dir / f"eval_plot_h{h}.png"
            plt.savefig(out, dpi=150)
        plt.close(fig)

    metrics_out = {
        "overall": {k: float(v) for k, v in overall.items()},
        "per_horizon": {k: [float(x) for x in v] for k, v in per_h.items()},
    }
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    return {
        "overall": overall,
        "per_horizon": per_h,
        "predictions_rescaled": predictions_rescaled,
        "targets_rescaled": trues_rescaled,
    }


def _feature_history(lags, rolls) -> int:
    """Return minimal raw-history rows needed to compute one feature row.
    - lags: list[int]
    - rolls: list[int] of rolling window sizes
    """
    max_lag = max(lags) if lags else 0
    max_roll_need = (max(rolls) - 1) if rolls else 0
    return max(max_lag, max_roll_need)


def predict_at(run_dir: Path, timestamp: datetime | None = None,):
    """Predict electricity demand.

    Args:
        run_dir (Path): Path to the run directory containing settings and the model to use.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the predicted values and the corresponding timestamps.
    """

    run_dir = Path(run_dir)
    # Load artifacts
    config, split_info, model, scaler_X, scaler_y = load_artifact(run_dir)
    horizon = config['data']['horizon']
    lookback = config['data']['lookback']
    batch_size = config['train']['batch_size']
    target = config['data']['target']
    lags = config['features']['lags']
    rolls = config['features']['rolls']
    freq_min = config['data']['freq_minutes']

    # Load the necessary rows from the database
    rows_needed = _feature_history(lags, rolls) + lookback

    # Load the data
    df_full = read_dataframe_from_sql(db_path=config['data']['source'],
                                      table_name=config['data']['table'],
                                      column_names=config['data']['columns'],
                                      timestamp_col=config['data']['timestamp_col'],
                                      timestamp_format=config['data']['timestamp_format'],
                                      last_rows=None)

    # Read weather data
    if config['data']['weather']['use']:
        df_weather = read_dataframe_from_sql(db_path=config['data']['weather']['source'],
                                             table_name=config['data']['weather']['table'],
                                             column_names=config['data']['weather']['columns'],
                                             timestamp_col=config['data']['weather']['timestamp_col'],
                                             timestamp_format=config['data']['weather']['timestamp_format'], )
    else:
        df_weather = None

    if not isinstance(df_full.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas.DatetimeIndex.")

    # Determine the as-of position (latest, or nearest to provided timestamp)
    if timestamp is None:
        pos = len(df_full) - 1
    else:
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime.datetime")
        pos = df_full.index.get_indexer(
            [pd.Timestamp(timestamp)], method="nearest")[0]
        if pos == -1:
            raise ValueError(f"No index near {timestamp!r}.")

    # Slice minimal past window; ensure we have enough rows
    start_pos = max(0, pos - rows_needed + 1)  # inclusive start
    df_past = df_full.iloc[start_pos: pos + 1]
    if len(df_past) < rows_needed:
        raise ValueError(
            f"Not enough rows up to {df_full.index[pos]}: have {len(df_past)}, need {rows_needed} "
            f"(lookback={lookback}, feature_history={rows_needed - lookback})."
        )

    # Add features
    X_df, _ = build_feature_dataframe(
        df_past, target, df_weather=df_weather, lags=lags, rolls=rolls)

    # Convert to numpy
    X = X_df.to_numpy(dtype='float32')

    # Scaler X and y training and validation data
    X_scaled = scaler_X.transform(X)

    # Create sequences
    X_seq = X_scaled[-lookback:]

    # Convert to torch
    X_tensor = torch.from_numpy(X_seq).float().unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        y_pred = model(X_tensor).detach().cpu().numpy()[0]

    y_pred_rescaled = scaler_y.inverse_transform(
        y_pred.reshape(1, -1)).reshape(-1)

    # Create timestamps
    step = pd.to_timedelta(freq_min, unit="m")
    asof_ts = df_full.index[pos]
    ts_idx = pd.date_range(start=asof_ts + step, periods=horizon, freq=step)
    timestamps = tuple(ts_idx.to_pydatetime())

    # Get true values if timestamp is provided
    y_true = None
    if timestamp is not None:
        end_pos = pos + horizon
        if end_pos < len(df_full):
            y_true_slice = df_full.iloc[pos + 1: end_pos +
                                        1][target].to_numpy(dtype="float64")
            if len(y_true_slice) == horizon:
                y_true = y_true_slice

    return y_pred_rescaled, y_true, timestamps
