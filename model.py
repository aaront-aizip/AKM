import torch
from torch import nn
import pytorch_lightning as pl

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        out = self.linear(x)
        out = self.softmax(out)
        return out
    
class LSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_stacked_layers, dropout, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        # Batch x Patch Size x Features (3)
        self.lstm = nn.LSTM(num_features, hidden_size, num_stacked_layers, dropout=dropout, batch_first=True)

        self.linear = nn.Linear(hidden_size, 1)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:,-1,:])
        # out = self.softmax(out)

        return out.squeeze(-1)

class LSTM_Full(nn.Module):
    def __init__(self, num_features, hidden_size, num_stacked_layers, num_classes, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        # Batch x Patch Size x Features (3)
        self.lstm = nn.LSTM(num_features, hidden_size, num_stacked_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out)
        # out = self.softmax(out)

        return out.squeeze(-1)

class GRU(nn.Module):
    def __init__(self, num_features, hidden_size, num_stacked_layers, dropout, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        # Batch x Patch Size x Features (3)
        self.gru = nn.GRU(num_features, hidden_size, num_stacked_layers, dropout=dropout, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, 1)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        out = self.linear(out[:,-1,:])
        # out = self.softmax(out)

        return out.squeeze(-1)
    
class Transformer(nn.Module):
    def __init__(self, num_features, num_stacked_layers, dropout, device):
        super().__init__()
        self.num_features = num_features
        self.num_stacked_layers = num_stacked_layers
        self.device = device

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_stacked_layers)
        
        self.linear = nn.Linear(num_features, 1)
        # self.softmax = nn.Softmax()
    def forward(self, x):
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)
        out = self.linear(out)

        return out.squeeze(-1)
