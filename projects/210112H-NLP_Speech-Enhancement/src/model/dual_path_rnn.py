import torch.nn as nn

class DualPathRNN(nn.Module):
    """
    Dimension-preserving two-stage RNN:
      - Input:  [B, T, D]
      - Output: [B, T, D]  (same D)
    We use bidirectional LSTMs and choose hidden_size so that
    2*hidden_size == D at each stage.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # First BiLSTM: in=input_size, out=2*hidden_size == input_size
        self.intra = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # Second BiLSTM: in=2*hidden_size (== input_size), out=2*hidden_size (== input_size)
        self.inter = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [B, T, D]
        x, _ = self.intra(x)   # -> [B, T, 2*hidden] == [B, T, D]
        x, _ = self.inter(x)   # -> [B, T, 2*hidden] == [B, T, D]
        return x
