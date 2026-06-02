import math
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
    """Raw EMG encoder: raw_emg [B, F*T, 8] -> latent [B, T, D].

    ``conv_strides`` sets the **token temporal resolution** (phase-2 knob): the
    encoder downsamples raw EMG by ``F = prod(conv_strides)`` (one stride-``s``
    ResBlock per entry). Default ``(2, 2, 2)`` reproduces the phase-1 8× rate
    (~86 Hz). Coarser units (subwords/phonemes) tolerate a larger factor
    (e.g. ``(2, 2, 2, 2)`` = 16×, fewer output frames); characters want the
    finer 8× (or 4× via ``(2, 2)``). ``F`` must stay <= the shortest target's
    CTC length and must divide the training ``fixed_raw_len``.
    """

    def __init__(self, model_size=768, num_layers=6, dropout=0.2, apply_train_shift=True,
                 conv_strides=(2, 2, 2)):
        super().__init__()
        self.apply_train_shift = apply_train_shift
        self.conv_strides = tuple(conv_strides)
        self.downsample_factor = math.prod(self.conv_strides)
        blocks, in_ch = [], 8
        for stride in self.conv_strides:
            blocks.append(ResBlock(in_ch, model_size, stride))
            in_ch = model_size
        self.conv_blocks = nn.Sequential(*blocks)
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
        if self.training and self.apply_train_shift and self.downsample_factor > 1:
            shift = random.randrange(self.downsample_factor)
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
    def __init__(self, model_size=768, num_layers=6, dropout=0.2, vocab_size=37,
                 conv_strides=(2, 2, 2)):
        super().__init__()
        self.encoder = GaddyRawEMGEncoder(
            model_size=model_size, num_layers=num_layers, dropout=dropout,
            conv_strides=conv_strides,
        )
        self.ctc_head = CTCHead(model_size=model_size, vocab_size=vocab_size)

    @property
    def downsample_factor(self):
        return self.encoder.downsample_factor

    def forward(self, raw_emg):
        return self.ctc_head(self.encoder(raw_emg))
