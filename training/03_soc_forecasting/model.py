"""
LSTM-Transformer for DeltaSoC forecasting, after Feng et al. (Energy 2024):
input projection -> LSTM (local temporal deps) -> sinusoidal positional
encoding -> Transformer encoder (long-range deps) -> linear head on the last
timestep (the forecast anchor). A learned vehicle embedding is concatenated to
every timestep so the model can absorb per-vehicle capacity / efficiency
differences.
"""

import math

import torch
import torch.nn as nn

import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class LSTMTransformer(nn.Module):
    def __init__(
        self,
        n_channels: int = config.N_INPUT_CHANNELS,
        n_vehicles: int = len(config.VEHICLE_VOCAB),
        d_model: int = config.D_MODEL,
        n_heads: int = config.N_HEADS,
        n_encoder_layers: int = config.N_ENCODER_LAYERS,
        dim_feedforward: int = config.DIM_FEEDFORWARD,
        lstm_layers: int = config.LSTM_LAYERS,
        dropout: float = config.DROPOUT,
        vehicle_emb_dim: int = config.VEHICLE_EMB_DIM,
    ):
        super().__init__()
        self.vehicle_emb = nn.Embedding(n_vehicles, vehicle_emb_dim)
        self.input_proj = nn.Linear(n_channels + vehicle_emb_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=lstm_layers, batch_first=True)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, vehicle_idx):
        # x: (B, T, n_channels), vehicle_idx: (B,)
        emb = self.vehicle_emb(vehicle_idx)  # (B, emb_dim)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        z = self.input_proj(torch.cat([x, emb], dim=-1))
        z, _ = self.lstm(z)
        z = self.dropout(self.pos_enc(z))
        z = self.encoder(z)
        return self.head(z[:, -1]).squeeze(-1)  # (B,) standardized DeltaSoC


if __name__ == "__main__":
    model = LSTMTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(4, config.INPUT_LEN, config.N_INPUT_CHANNELS)
    veh = torch.randint(0, len(config.VEHICLE_VOCAB), (4,))
    with torch.no_grad():
        y = model(x, veh)
    print(f"params: {n_params:,}")
    print("output shape:", tuple(y.shape))
    assert y.shape == (4,)
    print("forward pass OK")
