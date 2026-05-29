import random

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import TransformerEncoderLayer


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        res = self.res_norm(self.residual_path(input_value)) if self.residual_path is not None else input_value
        return F.relu(x + res)


class GaddyRawEMGEncoder(nn.Module):
    """Raw EMG encoder: raw_emg [B, 8T, 8] -> latent [B, T, D]."""

    def __init__(self, model_size=768, num_layers=6, dropout=0.2, apply_train_shift=True):
        super().__init__()
        self.apply_train_shift = apply_train_shift
        self.conv_blocks = nn.Sequential(
            ResBlock(8, model_size, 2),
            ResBlock(model_size, model_size, 2),
            ResBlock(model_size, model_size, 2),
        )
        self.w_raw_in = nn.Linear(model_size, model_size)
        encoder_layer = TransformerEncoderLayer(
            d_model=model_size,
            nhead=8,
            relative_positional=True,
            relative_positional_distance=100,
            dim_feedforward=3072,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, raw_emg: torch.Tensor) -> torch.Tensor:
        x_raw = raw_emg
        if self.training and self.apply_train_shift:
            shift = random.randrange(8)
            if shift > 0:
                x_raw = raw_emg.clone()
                x_raw[:, :-shift, :] = raw_emg[:, shift:, :]
                x_raw[:, -shift:, :] = 0

        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.w_raw_in(x_raw)
        x = self.transformer(x_raw.transpose(0, 1)).transpose(0, 1)
        return x


class CTCHead(nn.Module):
    """CTC output head: latent [B, T, D] -> logits [B, T, vocab+1]."""

    def __init__(self, model_size=768, vocab_size=37):
        super().__init__()
        self.linear = nn.Linear(model_size, vocab_size + 1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.linear(latent)


class BaselineCTCModel(nn.Module):
    def __init__(self, model_size=768, num_layers=6, dropout=0.2, vocab_size=37):
        super().__init__()
        self.encoder = GaddyRawEMGEncoder(model_size=model_size, num_layers=num_layers, dropout=dropout)
        self.ctc_head = CTCHead(model_size=model_size, vocab_size=vocab_size)

    def forward(self, raw_emg):
        return self.ctc_head(self.encoder(raw_emg))
