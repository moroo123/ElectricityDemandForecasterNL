import numpy as np


def compute_metrics(y_true, y_pred, eps=1e-8):
    """Compute evaluation metrics for the model predictions.

    Args:
        y_true (np.array): Ground truth values.
        y_pred (np.array): Predicted values.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-8.

    Returns:
        tuple: Overall and per-horizon metrics.
    """
    diff = y_pred - y_true
    mae = np.mean(np.abs(diff), axis=0)
    mse = np.mean(diff**2, axis=0)
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    mape = np.mean(np.abs(diff) / (np.abs(y_true) + eps), axis=0) * 100.0

    overall = {
        'MAE': float(np.mean(mae)),
        'MSE': float(np.mean(mse)),
        'RMSE': float(np.mean(rmse)),
        'MAPE': float(np.mean(mape)),
    }
    per_h = {
        'MAE': mae.tolist(),
        'MSE': mse.tolist(),
        'RMSE': rmse.tolist(),
        'MAPE': mape.tolist(),
    }
    return (overall, per_h)
