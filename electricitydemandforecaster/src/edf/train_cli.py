import yaml
import sys
import os
import numpy as np
from edf.train import cross_validate, train_final
from edf.utils import expand_config, make_run_dir
import torch
import json
from edf.inout import read_dataframe_from_sql
import argparse
from tqdm.auto import tqdm
import joblib
import optuna
import pandas as pd
from pathlib import Path


def objective(trial: optuna.trial.Trial, config: dict, df_trainval: pd.DataFrame, df_weather: pd.DataFrame | None):

    hpo_params = config['train']['hpo']

    # Suggest values for hyperparameters
    lr = trial.suggest_float("learning_rate", **hpo_params['learning_rate'])
    weight_decay = trial.suggest_float(
        "weight_decay", **hpo_params['weight_decay'])
    batch_size = trial.suggest_categorical(
        "batch_size", **hpo_params['batch_size'])

    model_kwargs = {k: v for k, v in config['model'].items() if k not in [
        'name']}

    model_kwargs['hidden_size'] = trial.suggest_categorical(
        "hidden_size", **hpo_params['hidden_size'])
    model_kwargs['num_layers'] = trial.suggest_categorical(
        "num_layers", **hpo_params['num_layers'])
    model_kwargs['conv_channels'] = trial.suggest_categorical(
        'conv_channels', **hpo_params['conv_channels'])
    model_kwargs['num_conv_layers'] = trial.suggest_categorical(
        'num_conv_layers', **hpo_params['num_conv_layers'])
    model_kwargs['conv_dropout'] = trial.suggest_float(
        'conv_dropout', **hpo_params['conv_dropout'])
    model_kwargs['dropout'] = trial.suggest_float(
        'dropout', **hpo_params['dropout'])
    model_kwargs['kernel_size'] = trial.suggest_categorical(
        'kernel_size', **hpo_params['kernel_size'])

    # Unpack parameters from config
    target = config['data']['target']
    lags = config['features']['lags']
    rolls = config['features']['rolls']
    lookback = config['data']['lookback']
    horizon = config['data']['horizon']
    model_name = config['model']['name']

    mean_cv, _, _ = cross_validate(
        df_trainval, model_name, target, df_weather, lags, rolls, lookback,
        horizon, model_kwargs,
        batch_size=batch_size,
        n_splits=config['cv']['n_splits'],
        epochs=config['train']['epochs'],
        lr=lr,
        weight_decay=weight_decay,
        patience=config['train']['patience'],
        seed=config['seed'],
        trial=trial
    )

    return mean_cv


def main(db_path: str, table_name: str, db_path_weather: str | None = None, table_name_weather: str | None = None, config_path: str = 'configs', runs_root: str = 'runs', run_optimization: bool = True, fraction: int = 1, resume_from: str | None = None):
    """Train the model using the specified configuration.
    """

    column_names = ['utc_timestamp', 'NL_load_actual_entsoe_transparency']
    timestamp_col = 'utc_timestamp'
    timestamp_format = '%Y-%m-%dT%H:%M:%SZ'

    df = read_dataframe_from_sql(
        db_path, table_name, column_names, timestamp_col, timestamp_format,)

    df = df.iloc[:(len(df)//fraction)]

    if resume_from:
        run_dir = Path(resume_from)
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Run directory to resume from does not exist: {resume_from}")
        print(f"Resuming run from: {run_dir}")
    else:
        run_dir = make_run_dir(runs_root)

    if table_name_weather != 'None':
        column_names_weather = [
            'utc_timestamp', 'NL_temperature', 'NL_radiation_direct_horizontal']
        timestamp_col = 'utc_timestamp'
        timestamp_format = '%Y-%m-%dT%H:%M:%SZ'
        df_weather = read_dataframe_from_sql(
            db_path_weather, table_name_weather, column_names_weather, timestamp_col, timestamp_format,)
    else:
        df_weather = None
        print('No weather data provided')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    lags = config['features']['lags']
    rolls = config['features']['rolls']
    lookback = config['data']['lookback']
    horizon = config['data']['horizon']

    # Create trainval/test split
    ratio = config['split']['trainval_test_split']
    split_idx = int(len(df) * ratio)
    df_trainval, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

    if run_optimization:
        pruner = optuna.pruners.HyperbandPruner()

        study = optuna.create_study(
            storage=f'sqlite:///{run_dir}/optuna_optimization.db',
            study_name='hyperparameter_optimization',
            direction="minimize",
            pruner=pruner,
            load_if_exists=True)
        study.optimize(
            lambda trial: objective(trial, config, df_trainval, df_weather),
            n_trials=config['train']['hpo']['n_trials']
        )

        print(f"Best trial finished with value: {study.best_value}")
        print("Best hyperparameters found: ", study.best_params)
        best_params = study.best_params
    else:
        print("Skipping hyperparameter optimization, using default parameters from config.")
        best_params['learning_rate'] = config['train']['hpo']['learning_rate']['low']
        best_params['weight_decay'] = config['train']['hpo']['weight_decay']['low']
        best_params['batch_size'] = config['train']['hpo']['batch_size']['choices'][0]
        best_params['hidden_size'] = config['model']['hidden_size']
        best_params['num_layers'] = config['model']['num_layers']

    best_cfg = config.copy()
    best_cfg['train']['learning_rate'] = best_params.get('learning_rate')
    best_cfg['train']['weight_decay'] = best_params.get('weight_decay')
    best_cfg['train']['batch_size'] = best_params.get('batch_size')
    best_cfg['model']['hidden_size'] = best_params.get('hidden_size')
    best_cfg['model']['num_layers'] = best_params.get('num_layers')
    best_cfg['model']['conv_channels'] = best_params.get('conv_channels')
    best_cfg['model']['num_conv_layers'] = best_params.get('num_conv_layers')
    best_cfg['model']['kernel_size'] = best_params.get('kernel_size')
    best_cfg['model']['conv_dropout'] = best_params.get('conv_dropout')
    best_cfg['train'].pop('hpo', None)

    target = best_cfg['data']['target']
    lookback = best_cfg['data']['lookback']
    horizon = best_cfg['data']['horizon']
    model_name = best_cfg['model']['name']

    model_kwargs = {k: v for k, v in best_cfg['model'].items() if k not in [
        'name']}

    model, scaler_X_window, scaler_X_feature, scaler_y, windows_input_size, feature_input_size, final_val_loss = train_final(
        df_trainval, model_name, target, df_weather, lags, rolls, lookback,
        horizon, model_kwargs,
        batch_size=best_cfg['train']['batch_size'],
        epochs=best_cfg['train']['epochs'],
        lr=best_cfg['train']['learning_rate'],
        weight_decay=best_cfg['train']['weight_decay'],
        train_val_split=best_cfg['split']['train_val_split'],
        patience=best_cfg['train']['patience'],
        seed=best_cfg['seed'],
    )

    # Add input and output size to config
    best_cfg['model']['windows_input_size'] = windows_input_size
    best_cfg['model']['feature_input_size'] = feature_input_size
    best_cfg['model']['output_size'] = best_cfg['data']['horizon']

    # Add data loading information to config
    best_cfg['data']['source'] = db_path
    best_cfg['data']['table'] = table_name
    best_cfg['data']['columns'] = column_names
    best_cfg['data']['timestamp_col'] = timestamp_col
    best_cfg['data']['timestamp_format'] = timestamp_format

    if table_name_weather != 'None':
        best_cfg['data']['weather'] = {}
        best_cfg['data']['weather']['use'] = True
        best_cfg['data']['weather']['source'] = db_path_weather
        best_cfg['data']['weather']['table'] = table_name_weather
        best_cfg['data']['weather']['columns'] = column_names_weather
        best_cfg['data']['weather']['timestamp_col'] = timestamp_col
        best_cfg['data']['weather']['timestamp_format'] = timestamp_format

    best_cfg['split']['trainval_test_split'] = best_cfg['split']['trainval_test_split']
    best_cfg['split']['train_val_split'] = best_cfg['split']['train_val_split']

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(run_dir, f'{model_name}.pt'))
    joblib.dump(scaler_X_window, os.path.join(run_dir, 'scaler_X_window.pkl'))
    joblib.dump(scaler_X_feature, os.path.join(
        run_dir, 'scaler_X_features.pkl'))
    joblib.dump(scaler_y, os.path.join(run_dir, 'scaler_y.pkl'))

    # Save config of best model
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(best_cfg, f)

    # Save split info
    split_info = {
        'trainval': {'start_index': 0, 'end_index': len(df_trainval) - 1},
        'test': {'start_index': len(df_trainval), 'end_index': len(df_trainval) + len(df_test) - 1},
    }

    with open(os.path.join(run_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Save final training scores
    with open(os.path.join(run_dir, 'final_training_scores.json'), 'w') as f:
        json.dump({'final_val_mse': float(final_val_loss)}, f, indent=2)

    print(f'Saved artifacts to: {run_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train energy demand forecasting model.")
    parser.add_argument('--db-path', type=str,
                        default='data/raw/time_series.sqlite', help='Path to SQLite database')
    parser.add_argument('--table-name', type=str,
                        default='time_series_15min_singleindex', help='Table name in the database')
    parser.add_argument('--db-path-weather', type=str,
                        default='None', help='Path to SQLite database containing weather data')
    parser.add_argument('--table-name-weather', type=str,
                        default='None', help='Table name in the database')
    parser.add_argument('--config', type=str,
                        default='configs/base.yaml', help='YAML config file')
    parser.add_argument('--runs-root', type=str,
                        default='runs', help='Root directory for runs')
    parser.add_argument('--run-optimization', type=str,
                        default='True', help='Whether to perform hyperparameter optimization')
    parser.add_argument('--fraction', type=int,
                        default=1, help='Divide complete data size by fraction')
    parser.add_argument('--resume-from', type=str,
                        default=None, help='Path to a previous run directory to resume optimization.')
    args = parser.parse_args()

    main(db_path=args.db_path,
         table_name=args.table_name,
         db_path_weather=args.db_path_weather,
         table_name_weather=args.table_name_weather,
         config_path=args.config,
         runs_root=args.runs_root,
         run_optimization=args.run_optimization.lower() in ('true', '1', 'yes'),
         fraction=args.fraction,
         resume_from=args.resume_from)

    sys.exit(0)

# python3 -m edf.train_cli --db-path data/raw/time_series.sqlite --table-name time_series_15min_singleindex --db-path-weather data/raw/weather_data.sqlite --table-name-weather weather_data --config electricitydemandforecaster/configs/base_lstm_optimization.yaml --crossvalidate false --runs-root electricitydemandforecaster/runs --fraction 60

# python3 -m edf.train_cli --db-path ../data/raw/time_series.sqlite --table-name time_series_15min_singleindex --db-path-weather ../data/raw/weather_data.sqlite --table-name-weather weather_data --config configs/base_lstm_optimization.yaml --crossvalidate false --runs-root runs --fraction 60
