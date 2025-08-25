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


def main(db_path: str, table_name: str, db_path_weather: str | None = None, table_name_weather: str | None = None, config_path: str = 'configs', runs_root: str = 'runs', run_cv: bool = True, fraction: int = 1):
    """Train the model using the specified configuration.
    """

    column_names = ['utc_timestamp', 'NL_load_actual_entsoe_transparency']
    timestamp_col = 'utc_timestamp'
    timestamp_format = '%Y-%m-%dT%H:%M:%SZ'

    df = read_dataframe_from_sql(
        db_path, table_name, column_names, timestamp_col, timestamp_format,)

    df = df.iloc[:(len(df)//fraction)]
    if table_name_weather != 'None':
        column_names_weather = ['utc_timestamp', 'NL_temperature']
        timestamp_col = 'utc_timestamp'
        timestamp_format = '%Y-%m-%dT%H:%M:%SZ'
        df_weather = read_dataframe_from_sql(
            db_path_weather, table_name_weather, column_names_weather, timestamp_col, timestamp_format,)
    else:
        df_weather = None
        print('No weather data provided')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    list_configs = expand_config(config)

    lags = config['features']['lags']
    rolls = config['features']['rolls']
    lookback = config['data']['lookback']
    horizon = config['data']['horizon']

    mean_cvs, std_cvs = [], []

    sweep_results = []

    if run_cv:
        for cfg in tqdm(list_configs, total=len(list_configs), desc="Sweeping hyperparameters", leave=False):

            target = cfg['data']['target']
            lookback = cfg['data']['lookback']
            horizon = cfg['data']['horizon']
            model_name = cfg['model']['name']

            model_kwargs = {k: v for k, v in cfg['model'].items() if k not in [
                'name']}

            # Create trainval/test split
            ratio = cfg['split']['trainval_test_split']
            split_idx = int(len(df) * ratio)
            df_trainval, df_test = df.iloc[:split_idx], df.iloc[split_idx:]

            # Run cross-validation
            mean_cv, std_cv, splits = cross_validate(
                df_trainval, model_name, target, df_weather, lags, rolls, lookback,
                horizon, model_kwargs,
                batch_size=cfg['train']['batch_size'],
                n_splits=cfg['cv']['n_splits'],
                epochs=cfg['train']['epochs'],
                lr=cfg['train']['learning_rate'],
                weight_decay=cfg['train']['weight_decay'],
                patience=cfg['train']['patience'],
                seed=cfg['seed']
            )

            mean_cvs.append(mean_cv)
            std_cvs.append(std_cv)

            sweep_results.append(
                {'config': cfg, 'mean_cv': float(mean_cv), 'std_cv': float(std_cv)})

        # Get best configuration based on mean CV loss
        best_idx = np.argmin(mean_cvs)
    else:
        best_idx = 0
    best_cfg = list_configs[best_idx]

    if run_cv:
        # Add splits to splits dictionary
        splits['trainval_test_split'] = best_cfg['split']['trainval_test_split']
        splits['test'] = {'start_index': len(df_trainval),
                          'end_index': len(df_trainval) + len(df_test) - 1, }
    else:
        ratio = best_cfg['split']['trainval_test_split']
        split_idx = int(len(df) * ratio)
        df_trainval, df_test = df.iloc[:split_idx], df.iloc[split_idx:]
        splits = {}
        splits['test'] = {'start_index': len(df_trainval),
                          'end_index': len(df_trainval) + len(df_test) - 1, }

    target = best_cfg['data']['target']
    lookback = best_cfg['data']['lookback']
    horizon = best_cfg['data']['horizon']
    model_name = best_cfg['model']['name']

    model_kwargs = {k: v for k, v in best_cfg['model'].items() if k not in [
        'name']}

    model, scaler_X, scaler_y, input_size, final_val_loss = train_final(
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
    best_cfg['model']['input_size'] = input_size
    best_cfg['model']['output_size'] = best_cfg['data']['horizon']

    # Add data loading information to config
    best_cfg['data']['source'] = db_path
    best_cfg['data']['table'] = table_name
    best_cfg['data']['columns'] = column_names
    best_cfg['data']['timestamp_col'] = timestamp_col
    best_cfg['data']['timestamp_format'] = timestamp_format

    best_cfg['split']['trainval_test_split'] = best_cfg['split']['trainval_test_split']
    best_cfg['split']['train_val_split'] = best_cfg['split']['train_val_split']

    # Save artifacts
    run_dir = make_run_dir(runs_root)
    torch.save(model.state_dict(), os.path.join(run_dir, f'{model_name}.pt'))
    joblib.dump(scaler_X, os.path.join(run_dir, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(run_dir, 'scaler_y.pkl'))

    # Save config of best model
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(best_cfg, f)

    if run_cv:
        # Save CV scores
        with open(os.path.join(run_dir, 'cv_scores.json'), 'w') as f:
            json.dump({'mean_cv_mse': float(mean_cvs[best_idx]),
                      'std_cv_mse': float(std_cvs[best_idx])}, f, indent=2)

        # Save all configs and their CV losses
        with open(os.path.join(run_dir, 'sweep_results.json'), 'w') as f:
            json.dump(sweep_results, f, indent=2)

    # Save final training scores
    with open(os.path.join(run_dir, 'final_training_scores.json'), 'w') as f:
        json.dump({'final_val_mse': float(final_val_loss)}, f, indent=2)

    # Save split info
    with open(os.path.join(run_dir, 'split_info.json'), 'w') as f:
        json.dump(splits, f, indent=2)

    print(f'Saved artifacts to: {run_dir}')
    if run_cv:
        print(f'CV MSE: {mean_cvs[best_idx]:.6f} +- {std_cvs[best_idx]:.6f}')
    else:
        print("Skipped cross-validation, trained with first config.")


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
    parser.add_argument('--crossvalidate', type=str,
                        default='True', help='Whether to perform cross-validation')
    parser.add_argument('--fraction', type=int,
                        default=1, help='Divide complete data size by fraction')
    args = parser.parse_args()

    main(db_path=args.db_path,
         table_name=args.table_name,
         db_path_weather=args.db_path_weather,
         table_name_weather=args.table_name_weather,
         config_path=args.config,
         runs_root=args.runs_root,
         run_cv=args.crossvalidate.lower() in ('true', '1', 'yes'),
         fraction=args.fraction)

    sys.exit(0)

# python3 -m edf.train_cli --db-path data/raw/time_series.sqlite --table-name time_series_15min_singleindex --db-path-weather data/raw/weather_data.sqlite --table-name-weather weather_data --config electricitydemandforecaster/configs/base_lstm_optimization.yaml --crossvalidate false --runs-root electricitydemandforecaster/runs --fraction 60

# python3 -m edf.train_cli --db-path ../data/raw/time_series.sqlite --table-name time_series_15min_singleindex --db-path-weather ../data/raw/weather_data.sqlite --table-name-weather weather_data --config configs/base_lstm_optimization.yaml --crossvalidate false --runs-root runs --fraction 60
