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
                 windows_input_size: int,
                 feature_input_size: int,
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
        self.windows_input_size = windows_input_size
        self.feature_input_size = feature_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.output_size = output_size

        # Define CNN
        conv_layers = []
        conv_kernel_dilations = []
        dilation = 1
        in_layers = self.windows_input_size
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
        cnn_output_channels = cnn_conv_channels if cnn_num_conv_layers > 0 else self.windows_input_size
        lstm_input_size = cnn_output_channels + self.feature_input_size
        lstm_dropout = lstm_dropout if self.lstm_num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(
                in_features=lstm_hidden_size,
                out_features=output_size,
            )
        )

    def forward(self, x_window: torch.Tensor, x_features: torch.Tensor):
        """
        Forward pass for the model.
        """

        # Pass through the convolutional layer only with unengineered features
        x_conv = x_window.permute(0, 2, 1)  # (B, F, T)
        for layer, (k, d) in zip(self.conv_layers, self.conv_kernel_dilations):
            pad_left = (k - 1) * d
            x_conv = F.pad(x_conv, (pad_left, 0))  # pad on time axis only
            x_conv = layer(x_conv)
        x_conv = x_conv.permute(0, 2, 1)  # (B, T, C)

        # Expand engineered features to match the sequence length of the CNN output
        x_features_expanded = x_features.unsqueeze(
            1).expand(-1, x_conv.size(1), -1)

        # Concatenate CNN output with expanded engineered features
        lstm_input = torch.cat((x_conv, x_features_expanded), dim=2)

        # Pass combined features through LSTM
        h0 = torch.zeros(self.lstm_num_layers, lstm_input.size(
            0), self.lstm_hidden_size).to(lstm_input.device)
        c0 = torch.zeros(self.lstm_num_layers, lstm_input.size(
            0), self.lstm_hidden_size).to(lstm_input.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))

        # Get the last hidden state of the LSTM
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)

        return out


MODEL_REGISTRY = {
    'BaselineLSTM': BaselineLSTM,
    'CnnLSTM': CnnLSTM,
}
