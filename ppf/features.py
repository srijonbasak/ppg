"""Torch-only log-Mel front end (no torchaudio dependency)."""

import math
import torch
from torch import nn


def _hz_to_mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    mel_min, mel_max = _hz_to_mel(20), _hz_to_mel(sample_rate / 2)
    hz = [_mel_to_hz(mel_min + i * (mel_max - mel_min) / (n_mels + 1))
          for i in range(n_mels + 2)]
    bins = [min(n_fft // 2, int((n_fft + 1) * value / sample_rate)) for value in hz]
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center == left: center += 1
        if right == center: right += 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=8000, n_fft=256, hop_length=80, n_mels=40):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("mel", mel_filterbank(sample_rate, n_fft, n_mels))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        spec = torch.stft(waveform, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True).abs().square()
        mel = torch.einsum("mf,bft->bmt", self.mel, spec)
        feat = torch.log(mel.clamp_min(1e-6))
        mean = feat.mean(dim=(-2, -1), keepdim=True)
        std = feat.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
        return (feat - mean) / std

