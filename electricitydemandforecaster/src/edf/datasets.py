import numpy as np
from torch.utils.data import Dataset
import torch


class ElectricityDemandDataset(Dataset):
    """
    A PyTorch Dataset for loading electricity demand data.
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 lookback_length: int,
                 horizon_length: int,
                 ):
        """Electricity Demand Dataset

        Args:
            X (np.array): Input data
            y (np.array): Target data
            lookback_length (int): Length of the lookback window
            horizon_length (int): Length of the forecast horizon
        """
        self.lookback_length = lookback_length
        self.horizon_length = horizon_length

        self.X = X
        self.y = y

        self.num_timesteps, self.num_features = self.X.shape

        max_start = self.num_timesteps - lookback_length - horizon_length + 1

        # Compute indices for lookback and horizon windows
        start_lookback = np.arange(0, max_start, 1)
        base_lookback = np.arange(0, lookback_length)
        self.window_lookback = (
            start_lookback[:, None] + base_lookback[None, :])

        base_horizon = np.arange(0, horizon_length)
        self.window_horizon = (
            start_lookback[:, None] + lookback_length + base_horizon[None, :])

    def __getitem__(self, idx):
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple(torch.Tensor, torch.Tensor): A tuple containing the X tensor and target tensor.
        """
        X = self.X[self.window_lookback[idx]]
        y = self.y[self.window_horizon[idx]]

        return (torch.Tensor(X), torch.Tensor(y))

    def __len__(self):
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.window_lookback)
