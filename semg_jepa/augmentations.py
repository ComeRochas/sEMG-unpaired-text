import random

import torch


class RawEMGAugment:
    def __init__(self, channel_dropout=0.0, time_mask_prob=0.0, time_mask_max=0, amp_scale=0.0, noise_std=0.0, temporal_shift=0):
        self.channel_dropout = channel_dropout
        self.time_mask_prob = time_mask_prob
        self.time_mask_max = time_mask_max
        self.amp_scale = amp_scale
        self.noise_std = noise_std
        self.temporal_shift = temporal_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        b, t, c = out.shape

        if self.channel_dropout > 0:
            drop_mask = (torch.rand(b, c, device=out.device) < self.channel_dropout).unsqueeze(1)
            out = out.masked_fill(drop_mask, 0.0)

        if self.time_mask_prob > 0 and self.time_mask_max > 0:
            for i in range(b):
                if random.random() < self.time_mask_prob:
                    width = random.randint(1, min(self.time_mask_max, t))
                    start = random.randint(0, t - width)
                    out[i, start:start + width, :] = 0.0

        if self.amp_scale > 0:
            scales = 1.0 + (torch.rand(b, 1, 1, device=out.device) * 2 - 1) * self.amp_scale
            out = out * scales

        if self.noise_std > 0:
            out = out + torch.randn_like(out) * self.noise_std

        if self.temporal_shift > 0:
            for i in range(b):
                shift = random.randint(-self.temporal_shift, self.temporal_shift)
                if shift > 0:
                    out[i, shift:, :] = out[i, :-shift, :].clone()
                    out[i, :shift, :] = 0
                elif shift < 0:
                    k = -shift
                    out[i, :-k, :] = out[i, k:, :].clone()
                    out[i, -k:, :] = 0
        return out
