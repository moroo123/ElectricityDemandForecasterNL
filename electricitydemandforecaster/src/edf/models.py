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
                 cnn_conv_channels: int,
                 cnn_num_conv_layers: int,
                 cnn_kernel_size: int,
                 cnn_dilation_base: int,
                 cnn_conv_dropout: float,
                 cnn_use_batchnorm: bool,
                 # LSTM
                 lstm_hidden_size: int,
                 lstm_num_layers: int,
                 output_size: int,
                 lstm_dropout: float = 0.0):
        """
        Initialize the CNN_LSTM model.
        """
        super().__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size
        self.lstm_num_layers = lstm_num_layers

        # Define CNN
        conv_layers = []
        conv_kernel_dilations = []
        dilation = 1
        in_layers = input_size
        for i in range(cnn_num_conv_layers):
            layer = nn.Conv1d(
                in_channels=in_layers,
                out_channels=cnn_conv_channels,
                kernel_size=cnn_kernel_size,
                dilation=dilation
            )
            block = [layer]
            if cnn_use_batchnorm:
                block.append(nn.BatchNorm1d(cnn_conv_channels))
            block.extend([nn.ReLU(), nn.Dropout(cnn_conv_dropout)])
            conv_layers.append(nn.Sequential(*block))
            conv_kernel_dilations.append((cnn_kernel_size, dilation))
            in_layers = cnn_conv_channels
            dilation = dilation * cnn_dilation_base
        self.conv_layers = nn.ModuleList(conv_layers)
        self.conv_kernel_dilations = conv_kernel_dilations

        # Define LSTM
        lstm_input_size = cnn_conv_channels if len(
            self.conv_layers) > 0 else input_size
        lstm_dropout = lstm_dropout if lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.fc = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(
                in_features=lstm_hidden_size,
                out_features=self.output_size,
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

        h0 = torch.zeros(self.lstm_num_layers, x.size(
            0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(
            0), self.lstm_hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


MODEL_REGISTRY = {
    'BaselineLSTM': BaselineLSTM,
    'CnnLSTM': CnnLSTM,
}
