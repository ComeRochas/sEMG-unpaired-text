import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import TransformerEncoderLayer

FIXED_RAW_LEN = 1600  # training reshape length; the downsample factor must divide it.


def factor_to_strides(factor: int, fixed_raw_len: int = FIXED_RAW_LEN) -> tuple:
    """Decompose a downsample ``factor`` into per-ResBlock conv strides.

    ``factor`` must divide ``fixed_raw_len`` (= 2^6 * 5^2), i.e. be of the form
    ``2^a * 5^b``. Returns strides as 2s then 5s (largest stride **last**), e.g.
    ``20 -> (2, 2, 5)``, ``25 -> (5, 5)``, ``8 -> (2, 2, 2)``. The encoder's
    downsample factor is ``prod(strides) == factor``; the dataset right-crops raw
    EMG to a multiple of it. Standardizing the decomposition keeps run naming and
    the stride-5 kernel choice (see ``ResBlock``) consistent across the sweep.
    """
    if factor < 1 or fixed_raw_len % factor != 0:
        raise ValueError(
            f"downsample factor {factor} must divide fixed_raw_len={fixed_raw_len}")
    f, twos, fives = factor, 0, 0
    while f % 2 == 0:
        f //= 2
        twos += 1
    while f % 5 == 0:
        f //= 5
        fives += 1
    if f != 1:
        raise ValueError(
            f"downsample factor {factor} must be 2^a*5^b to divide {fixed_raw_len}")
    return tuple([2] * twos + [5] * fives)


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        # Keep kernel 3 for the proven stride<=3 stack; widen to an odd kernel >= stride
        # for stride>3 (the stride-5 layers that non-power-of-2 factors use) so no input
        # samples are skipped. padding=(kernel-1)/2 preserves the exact L/stride length.
        if stride > 3:
            kernel, pad = 2 * stride - 1, stride - 1
        else:
            kernel, pad = 3, 1
        self.conv1 = nn.Conv1d(num_ins, num_outs, kernel, padding=pad, stride=stride)
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
                 conv_strides=(2, 2, 2), num_private_layers=0, private_gate=True):
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

        def _make_transformer(n):
            layer = TransformerEncoderLayer(
                d_model=model_size,
                nhead=8,
                relative_positional=True,
                relative_positional_distance=100,
                dim_feedforward=3072,
                dropout=dropout,
            )
            return nn.TransformerEncoder(layer, n)

        # EMG-private pre-transformer (Option 1, frontend symmetry): extra EMG-only
        # attention layers applied BEFORE the shared transformer, so EMG — like audio
        # via wav2vec2's 12 layers — reaches the shared transformer already contextualized
        # by self-attention. The UML audio/text branches call ``self.transformer`` directly
        # (see ``UMLModel.forward_audio``), so they never traverse ``pre_transformer``.
        # num_private_layers=0 (default) reproduces the phase-1 encoder exactly (no extra
        # state-dict keys), so old checkpoints keep loading.
        self.num_private_layers = int(num_private_layers)
        self.pre_transformer = _make_transformer(self.num_private_layers) if self.num_private_layers > 0 else None

        # Diagnostic (2026-07): num_private_layers=6 (12 total from-scratch post-LN attention
        # layers before the CTC head) collapses to the CTC all-blank attractor within ~10 epochs
        # of pure supervised training — confirmed independent of clip_grad_norm (0.0 and 1.0 both
        # collapse; private=0/1/3 train fine at either clip). Root cause: a deep from-scratch
        # post-LN stack makes a large, unconstrained representational jump at init, and CTC's
        # all-blank minimum is right there to fall into. Fix: LayerScale-style residual gate,
        # a per-channel scale initialized to 0 around the WHOLE pre_transformer stack, so at
        # init pre_transformer is an exact no-op (output == input, identical to private=0) and
        # can only start contributing once gradients pull it away from zero.
        self.private_gate = bool(private_gate) and self.pre_transformer is not None
        if self.private_gate:
            self.private_gate_scale = nn.Parameter(torch.zeros(model_size))

        self.transformer = _make_transformer(num_layers)

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
        if self.pre_transformer is not None:
            pre_in = x_raw.transpose(0, 1)
            pre_out = self.pre_transformer(pre_in)
            if self.private_gate:
                # gate starts at 0 -> pre_out == pre_in -> exact no-op at init.
                pre_out = pre_in + self.private_gate_scale * (pre_out - pre_in)
            x_raw = pre_out.transpose(0, 1)
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
                 conv_strides=(2, 2, 2), num_private_layers=0, private_gate=True):
        super().__init__()
        self.encoder = GaddyRawEMGEncoder(
            model_size=model_size, num_layers=num_layers, dropout=dropout,
            conv_strides=conv_strides, num_private_layers=num_private_layers,
            private_gate=private_gate,
        )
        self.ctc_head = CTCHead(model_size=model_size, vocab_size=vocab_size)

    @property
    def downsample_factor(self):
        return self.encoder.downsample_factor

    def forward(self, raw_emg):
        return self.ctc_head(self.encoder(raw_emg))
