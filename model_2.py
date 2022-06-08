from sqlalchemy import collate
import torch
import math
from tqdm.auto import tqdm
"""

The model is a recurrent neural network that takes in 

"""

class ConvMiddleLayer(torch.nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(channels)

    def forward(self, x:torch.Tensor):
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        return x

class SweVoice(torch.nn.Module):
    def __init__(self, hidden_size, n_chars, input_channels, num_layers = 10, bidirectional=True):
        self.hidden_size = hidden_size
        self.n_chars = n_chars
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # Input (Batch, Mel_bins, Length)
        super().__init__()

        self.conv_stack = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=7, stride=2, padding=7//2),
            ConvMiddleLayer(64, 0.2),
            # torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=3//2),
            # ConvMiddleLayer(64, 0.2),
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 64),
            torch.nn.LayerNorm(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )

        self.rec = torch.nn.LSTM(
            input_size=64,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            dropout = 0,
            bidirectional = self.bidirectional,
            batch_first = True
        )
        bidir_const = 2 if self.bidirectional else 1
        self.output = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(self.hidden_size*bidir_const),
            torch.nn.Linear(self.hidden_size*bidir_const, self.n_chars)
        )

    def init_hidden(self, batch_size, device='cpu'):
        n, hs = self.num_layers, self.hidden_size
        if self.bidirectional:
            n*=2
        return (torch.zeros(n, batch_size, hs).to(device),
                torch.zeros(n, batch_size, hs).to(device))

    def forward(self, x:torch.Tensor, hidden):
        x = x.squeeze(1)
        x = self.conv_stack(x)

        x = x.permute(0, 2, 1)
        x = self.dense(x)
        x, (h, c) = self.rec(x, hidden)
        x = self.output(x)

        return x, (h, c)


if __name__ == '__main__':
    pass