import torch.nn as nn
import torch


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


MODEL_REGISTRY = {
    'BaselineLSTM': BaselineLSTM,
}
