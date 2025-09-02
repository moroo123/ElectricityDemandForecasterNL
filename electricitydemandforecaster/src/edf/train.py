from edf.datasets import ElectricityDemandDataset
from edf.models import MODEL_REGISTRY
from edf.features import build_feature_dataframe
from edf.utils import set_seed
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import optuna

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _check_data_for_cv(df_len: int, lags: list[int], rolls: list[int], lookback: int, horizon: int, n_splits: int):
    """Checks if there is enough data for cross-validation."""
    max_lag = max(lags) if lags else 0
    max_roll = max(rolls) if rolls else 0
    feature_history_needed = max(max_lag, max_roll)

    # After feature creation, we have (df_len - feature_history_needed) rows.
    n_feature_rows = df_len - feature_history_needed
    if n_feature_rows <= 0:
        raise ValueError(
            f"Not enough data to create any features. Need at least {feature_history_needed + 1} data points, but got {df_len}."
        )

    # TimeSeriesSplit splits the data. The first training fold is the smallest.
    # It will have n_feature_rows // (n_splits + 1) samples.
    first_fold_train_size = n_feature_rows // (n_splits + 1)

    # This fold must be large enough to create at least one sample for the model.
    min_rows_for_dataset = lookback + horizon
    if first_fold_train_size < min_rows_for_dataset:
        # Calculate minimum total feature rows needed
        min_feature_rows_needed = min_rows_for_dataset * (n_splits + 1)
        # Calculate minimum raw data points needed
        min_raw_data_points = min_feature_rows_needed + feature_history_needed
        raise ValueError(
            f"Insufficient data for cross-validation with n_splits={n_splits}. "
            f"The first training fold has only {first_fold_train_size} samples, but at least {min_rows_for_dataset} are required to create one lookback/horizon window. "
            f"A total of {min_raw_data_points} data points are needed, but only {df_len} are available for training/validation. "
            f"Breakdown: feature history={feature_history_needed}, lookback={lookback}, horizon={horizon}, n_splits={n_splits}."
        )


def _check_data_for_final_train(df_len: int, lags: list[int], rolls: list[int], lookback: int, horizon: int, train_val_split: float):
    """Checks if there is enough data for final training and validation."""
    max_lag = max(lags) if lags else 0
    max_roll = max(rolls) if rolls else 0
    feature_history_needed = max(max_lag, max_roll)

    n_feature_rows = df_len - feature_history_needed
    if n_feature_rows <= 0:
        raise ValueError(
            f"Not enough data to create any features. Need at least {feature_history_needed + 1} data points, but got {df_len}."
        )

    min_rows_for_dataset = lookback + horizon

    # Check training set size
    train_size = int(n_feature_rows * train_val_split)
    if train_size < min_rows_for_dataset:
        min_feature_rows_needed = int(
            np.ceil(min_rows_for_dataset / train_val_split))
        min_raw_data_points = min_feature_rows_needed + feature_history_needed
        raise ValueError(
            f"Insufficient data for training set with train_val_split={train_val_split}. "
            f"Training set has {train_size} samples, but at least {min_rows_for_dataset} are required to create one lookback/horizon window. "
            f"Need at least {min_raw_data_points} total data points for training/validation. Provided: {df_len}."
        )

    # Check validation set size
    val_size = n_feature_rows - train_size
    if val_size < min_rows_for_dataset:
        min_feature_rows_needed = int(
            np.ceil(min_rows_for_dataset / (1 - train_val_split)))
        min_raw_data_points = min_feature_rows_needed + feature_history_needed
        raise ValueError(
            f"Insufficient data for validation set with train_val_split={train_val_split}. "
            f"Validation set has {val_size} samples, but at least {min_rows_for_dataset} are required to create one lookback/horizon window. "
            f"Need at least {min_raw_data_points} total data points for training/validation. Provided: {df_len}."
        )


def _fit_one(model: nn.Module, train_dataloader: DataLoader, val_dataloader: DataLoader, scaler_y: StandardScaler, epochs: int, lr: float, weight_decay: float = 0.0, patience: int = 10, trial: optuna.trial.Trial | None = None, step_offset: int | None = None):
    """Train the model for one fold.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The training dataloader.
        val_dataloader (DataLoader): The validation dataloader.
        scaler_y (StandardScaler): The scaler for the target variable.
        epochs (int): The number of epochs to train.
        lr (float): The learning rate.
        weight_decay (float, optional): The weight decay. Defaults to 0.0.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 10.
        trial (optuna trial, optional): Optuna trial object for hyperparameter optimization.
        step_offset (int, optional): Step offset for the current fold. Defaults to None.

    Returns:
        float: The best validation loss.
    """

    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr, weight_decay=weight_decay)

    # Add scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # Define the loss function
    criterion = nn.MSELoss()

    # Initiate early stopping
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    epoch_no_improvement = 0

    # Training loop
    for epoch in tqdm(range(epochs), desc='Training epochs', leave=False):
        model.train()
        for x_window, x_features, y in train_dataloader:
            x_window, x_features, y = x_window.to(
                DEVICE), x_features.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()  # Remove gradient
            y_pred = model(x_window, x_features)  # Predict
            loss = criterion(y_pred, y.squeeze(-1))  # Calculate loss
            loss.backward()  # Backwards step
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update weights

        # Validation loop
        model.eval()
        val_loss = 0.0
        n_examples = 0

        with torch.no_grad():
            for x_window, x_features, y in val_dataloader:

                x_window, x_features, y = x_window.to(
                    DEVICE), x_features.to(DEVICE), y.to(DEVICE)
                y_pred = model(x_window, x_features)

                # Move to CPU
                y_pred_np = y_pred.detach().cpu().numpy()
                y_true_np = y.detach().cpu().numpy()

                y_true_np = np.squeeze(
                    y_true_np, axis=-1) if y_true_np.ndim == 3 and y_true_np.shape[-1] == 1 else y_true_np

                y_pred_flat = y_pred_np.reshape(-1, 1)
                y_true_flat = y_true_np.reshape(-1, 1)

                if scaler_y is None:
                    raise RuntimeError(
                        "scaler_y must be provided to compute unscaled validation loss.")
                y_pred_unscaled = scaler_y.inverse_transform(
                    y_pred_flat).reshape(y_pred_np.shape)
                y_true_unscaled = scaler_y.inverse_transform(
                    y_true_flat).reshape(y_true_np.shape)

                # Batch MSE in original units
                batch_mse = np.mean((y_pred_unscaled - y_true_unscaled) ** 2)

                # Weight by batch size
                bsz = x_window.size(0)
                val_loss += batch_mse * bsz
                n_examples += bsz

            val_loss = val_loss / max(n_examples, 1)

        # Step scheduler
        scheduler.step(val_loss)

        if trial:
            global_step = step_offset + epoch

            # Report results to the pruner
            trial.report(val_loss, global_step)

            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epoch_no_improvement = 0
        else:
            epoch_no_improvement += 1
            if epoch_no_improvement >= patience:
                # print(f'Early stopping at epoch {epoch + 1}')
                break

    tqdm.write(f'Final validation loss: {best_val_loss:.6f}')

    return best_val_loss


def cross_validate(df: pd.DataFrame, model_name: str, target_column: str, df_weather: pd.DataFrame | None,  lags: list[int], rolls: list[int], lookback: int, horizon: int, model_kwargs: dict, batch_size: int, n_splits: int, epochs: int, lr: float, weight_decay: float = 0.0, patience: int = 10, seed: int = 42, trial: optuna.trial.Trial | None = None):
    """Train the model using cross-validation.

    Args:
        df (pd.DataFrame): The input dataframe.
        model_name (str): The name of the model to train.
        target_column (str): The name of the target column.
        lags (list[int]): The list of lagged features to use.
        rolls (list[int]): The list of rolling window features to use.
        lookback (int): The number of time steps to back.
        horizon (int): The number of time steps to predict.
        model_kwargs (dict): Additional keyword arguments for the model.
        batch_size (int): The batch size for training.
        n_splits (int): The number of cross-validation splits.
        epochs (int): The number of epochs to train.
        lr (float): The learning rate.
        weight_decay (float, optional): The weight decay. Defaults to 0.0.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 10.
        seed (int, optional): The random seed. Defaults to 42.
        trial (Optuna trial, optional): Optuna trial for hyperparameter optimization

    Returns:
        tuple: A tuple containing the mean cross-validation loss, the standard deviation of the cross-validation loss, and the cross-validation splits.
    """

    set_seed(seed)

    _check_data_for_cv(len(df), lags, rolls, lookback,
                       horizon, n_splits)

    X_window_df, X_feature_df, y_df = build_feature_dataframe(
        df, target_column, df_weather=df_weather, lags=lags, rolls=rolls, horizon=horizon)

    X_window = X_window_df.to_numpy(dtype='float32')
    X_features = X_feature_df.to_numpy(dtype='float32')
    y = y_df.to_numpy(dtype='float32')

    # Train/validation split
    tscv = TimeSeriesSplit(n_splits)
    fold_losses = []
    cv_splits = {'cv': {'n_splits': n_splits,
                        'folds': []}}

    for fold, (train_idx, val_idx) in tqdm(enumerate(tscv.split(X_features)), total=n_splits, desc='Cross-validation folds', leave=False):

        cv_splits['cv']['folds'].append({
            'train_index_start': train_idx.tolist()[0],
            'train_index_end': train_idx.tolist()[-1],
            'val_index_start': val_idx.tolist()[0],
            'val_index_end': val_idx.tolist()[-1],
        })

        X_features_train, X_features_val = X_features[train_idx], X_features[val_idx]

        X_window_train, X_window_val = X_window[train_idx], X_window[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale
        scaler_X_window = StandardScaler()
        X_window_train = scaler_X_window.fit_transform(X_window_train)
        X_window_val = scaler_X_window.transform(X_window_val)

        scaler_X_feature = StandardScaler()
        X_features_train = scaler_X_feature.fit_transform(X_features_train)
        X_features_val = scaler_X_feature.transform(X_features_val)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val = scaler_y.transform(y_val.reshape(-1, 1))

        # Create a dataset
        train_dataset = ElectricityDemandDataset(
            X_window_train, X_features_train, y_train, lookback, horizon)
        val_dataset = ElectricityDemandDataset(
            X_window_val, X_features_val, y_val, lookback, horizon)

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

        model = MODEL_REGISTRY[model_name](
            windows_input_size=X_window_train.shape[1], feature_input_size=X_features_train.shape[1], **model_kwargs, output_size=horizon).to(DEVICE)
        step_offset = fold * epochs
        score = _fit_one(model, train_dataloader, val_dataloader, scaler_y,
                         epochs, lr, weight_decay, patience, trial, step_offset)

        fold_losses.append(score)

    return (np.mean(fold_losses), np.std(fold_losses), cv_splits)


def train_final(df: pd.DataFrame, model_name: str, target_column: str, df_weather: pd.DataFrame | None, lags: list[int], rolls: list[int], lookback: int, horizon: int, model_kwargs: dict, batch_size: int,  epochs: int, lr: float, weight_decay: float = 0.0, train_val_split: float = 0.9, patience: int = 10, seed: int = 42, ):
    """Train the final model on the entire dataset.

    Args:
        df (pd.DataFrame): The input dataframe.
        model_name (str): The name of the model to train.
        target_column (str): The name of the target column.
        df_weather (pd.DataFrame | None): The weather data dataframe.
        lags (list[int]): The list of lagged features to use.
        rolls (list[int]): The list of rolling window features to use.
        lookback (int): The number of time steps to look back.
        horizon (int): The number of time steps to predict.
        model_kwargs (dict): Additional keyword arguments for the model.
        batch_size (int): The batch size for training.
        epochs (int): The number of epochs to train.
        lr (float): The learning rate.
        weight_decay (float, optional): The weight decay. Defaults to 0.0.
        train_val_split (float, optional): The train/validation split. Defaults to 0.1.
        patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 10.
        seed (int, optional): The random seed. Defaults to 42.


    Returns:
        tuple: A tuple containing the trained model, the scaler for X, and the scaler for y.
    """
    set_seed(seed)

    _check_data_for_final_train(
        len(df), lags, rolls, lookback, horizon, train_val_split)

    X_window_df, X_features_df, y_df = build_feature_dataframe(
        df, target_column, df_weather=df_weather, lags=lags, rolls=rolls, horizon=horizon)

    X_window = X_window_df.to_numpy(dtype='float32')
    X_features = X_features_df.to_numpy(dtype='float32')
    y = y_df.to_numpy(dtype='float32')

    idx_split = int(X_window.shape[0] * (train_val_split))
    X_window_train, X_window_val = X_window[:idx_split], X_window[idx_split:]
    X_features_train, X_features_val = X_features[:idx_split], X_features[idx_split:]
    y_train, y_val = y[:idx_split], y[idx_split:]

    # Scale
    scaler_X_window = StandardScaler()
    X_window_train = scaler_X_window.fit_transform(X_window_train)
    X_window_val = scaler_X_window.transform(X_window_val)

    scaler_X_feature = StandardScaler()
    X_features_train = scaler_X_feature.fit_transform(X_features_train)
    X_features_val = scaler_X_feature.transform(X_features_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    # Create a dataset
    train_dataset = ElectricityDemandDataset(
        X_window_train, X_features_train, y_train, lookback, horizon)
    val_dataset = ElectricityDemandDataset(
        X_window_val, X_features_val, y_val, lookback, horizon)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    windows_input_size = X_window_train.shape[1]
    feature_input_size = X_features_train.shape[1]
    model = MODEL_REGISTRY[model_name](
        windows_input_size=windows_input_size, feature_input_size=feature_input_size, **model_kwargs, output_size=horizon).to(DEVICE)

    final_val_loss = _fit_one(model, train_dataloader, val_dataloader, scaler_y,
                              epochs, lr, weight_decay, patience)

    return (model, scaler_X_window, scaler_X_feature, scaler_y, windows_input_size, feature_input_size, final_val_loss)
