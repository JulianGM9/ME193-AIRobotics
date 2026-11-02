"""Candidate A: a dilated 1D CNN that outputs a stride probability for every sample (PLAN.md D10, D12).

  input (batch, 6, T) -> [Conv1d -> ReLU -> Dropout] x 5, dilations 1-2-4-8-16 -> Conv1d 1x1 -> logits (batch, T)

Every layer keeps the time axis ('same' padding), so each layer's activations are themselves
signals over time, which is what the live view draws. With kernel 7 the receptive field is
1 + 6 x (1 + 2 + 4 + 8 + 16) = 187 samples = 1.87 s: more than one stride even when walking slowly.
"""
import torch
from torch import nn


class StepCNN(nn.Module):
    def __init__(self, channels=(16, 32, 32, 32, 32), dilations=(1, 2, 4, 8, 16), kernel=7, dropout=0.1, in_ch=6):
        super().__init__()
        assert len(channels) == len(dilations)
        layers, prev = [], in_ch
        for ch, d in zip(channels, dilations):
            layers += [nn.Conv1d(prev, ch, kernel, dilation=d, padding=d * (kernel - 1) // 2), nn.ReLU(), nn.Dropout(dropout)]
            prev = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev, 1, 1)
        self.kernel, self.dilations = kernel, list(dilations)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(1)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel - 1) * sum(self.dilations)

    @classmethod
    def from_config(cls, m):
        return cls(m.channels, m.dilations, m.kernel, m.dropout)
