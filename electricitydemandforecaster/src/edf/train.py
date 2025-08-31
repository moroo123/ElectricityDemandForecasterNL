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
        for x, y in train_dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()  # Remove gradient
            y_pred = model(x)  # Predict
            loss = criterion(y_pred, y.squeeze(-1))  # Calculate loss
            loss.backward()  # Backwards step
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update weights

        # Validation loop
        model.eval()
        val_loss = 0.0
        n_examples = 0

        with torch.no_grad():
            for x, y in val_dataloader:

                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)

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
                bsz = x.size(0)
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

    X_df, y_df = build_feature_dataframe(
        df, target_column, df_weather=df_weather, lags=lags, rolls=rolls)

    X = X_df.to_numpy(dtype='float32')
    y = y_df.to_numpy(dtype='float32')

    # Train/validation split
    tscv = TimeSeriesSplit(n_splits)
    fold_losses = []
    cv_splits = {'cv': {'n_splits': n_splits,
                        'folds': []}}

    for fold, (train_idx, val_idx) in tqdm(enumerate(tscv.split(X)), total=n_splits, desc='Cross-validation folds', leave=False):

        cv_splits['cv']['folds'].append({
            'train_index_start': train_idx.tolist()[0],
            'train_index_end': train_idx.tolist()[-1],
            'val_index_start': val_idx.tolist()[0],
            'val_index_end': val_idx.tolist()[-1],
        })

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scaler X and y training and validation data
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val = scaler_y.transform(y_val.reshape(-1, 1))

        # Create a dataset
        train_dataset = ElectricityDemandDataset(
            X_train, y_train, lookback, horizon)
        val_dataset = ElectricityDemandDataset(
            X_val, y_val, lookback, horizon)

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

        input_size = X.shape[1]
        model = MODEL_REGISTRY[model_name](
            input_size=input_size, **model_kwargs, output_size=horizon).to(DEVICE)
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

    X_df, y_df = build_feature_dataframe(
        df, target_column, df_weather=df_weather, lags=lags, rolls=rolls)

    X = X_df.to_numpy(dtype='float32')
    y = y_df.to_numpy(dtype='float32')

    idx_split = int(X.shape[0] * (train_val_split))
    X_train, X_val = X[:idx_split], X[idx_split:]
    y_train, y_val = y[:idx_split], y[idx_split:]

    # Scaler X and y training and validation data
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    X_val = scaler_X.transform(X_val)
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    # Create a dataset
    train_dataset = ElectricityDemandDataset(
        X_train, y_train, lookback, horizon)
    val_dataset = ElectricityDemandDataset(
        X_val, y_val, lookback, horizon)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    input_size = X.shape[1]
    model = MODEL_REGISTRY[model_name](
        input_size=input_size, **model_kwargs, output_size=horizon).to(DEVICE)

    final_val_loss = _fit_one(model, train_dataloader, val_dataloader, scaler_y,
                              epochs, lr, weight_decay, patience)

    return (model, scaler_X, scaler_y, input_size, final_val_loss)
