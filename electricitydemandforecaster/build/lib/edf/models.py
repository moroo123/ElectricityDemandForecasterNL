import torch.nn as nn
import torch
import torch.nn.functional as F


class BaselineLSTM(nn.Module):
    """
    Baseline LSTM model for time series forecasting.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.0):
        """
        Initialize the BaselineLSTM model.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                in_features=hidden_size,
                out_features=output_size,
            )
        )

    def forward(self, x):
        """
        Forward pass for the model.
        """
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


class CnnLSTM(nn.Module):
    """
    Cnn_LSTM model for time series forecasting.
    """

    def __init__(self,
                 input_size: int,
                 # CNN
                 conv_channels: int,
                 num_conv_layers: int,
                 kernel_size: int,
                 dilation_base: int,
                 conv_dropout: float,
                 use_batchnorm: bool,
                 # LSTM
                 hidden_size: int,
                 num_layers: int,
                 output_size: int,
                 dropout: float = 0.0):
        """
        Initialize the CNN_LSTM model.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Define CNN
        conv_layers = []
        conv_kernel_dilations = []
        dilation = 1
        in_layers = input_size
        for i in range(num_conv_layers):
            layer = nn.Conv1d(
                in_channels=in_layers,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                dilation=dilation
            )
            block = [layer]
            if use_batchnorm:
                block.append(nn.BatchNorm1d(conv_channels))
            block.extend([nn.ReLU(), nn.Dropout(conv_dropout)])
            conv_layers.append(nn.Sequential(*block))
            conv_kernel_dilations.append((kernel_size, dilation))
            in_layers = conv_channels
            dilation = dilation * dilation_base
        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_kernel_dilations = conv_kernel_dilations

        # Define LSTM
        lstm_input_size = conv_channels if len(
            self.conv_layers) > 0 else input_size
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                in_features=hidden_size,
                out_features=output_size,
            )
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the model.
        """
        x = x.permute(0, 2, 1)  # (B, F, T)
        for layer, (k, d) in zip(self.conv_layers, self.conv_kernel_dilations):
            pad_left = (k - 1) * d
            x = F.pad(x, (pad_left, 0))  # pad on time axis only
            x = layer(x)

        x = x.permute(0, 2, 1)  # (B, T, C)

        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


MODEL_REGISTRY = {
    'BaselineLSTM': BaselineLSTM,
    'CnnLSTM': CnnLSTM,
}
