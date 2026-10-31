"""Core NCM-inspired coarse-to-fine parallel possibility model.

The key constraint is structural: level L is predicted from acoustic memory and
quantized levels < L. Positions *inside* L are predicted in one Transformer
pass, never autoregressively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .features import LogMelSpectrogram


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 1), nn.GroupNorm(4, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class AcousticEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.frontend = LogMelSpectrogram(cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels)
        self.stem = nn.Conv1d(cfg.n_mels, cfg.model_dim, 5, padding=2)
        self.blocks = nn.Sequential(ConvResidualBlock(cfg.model_dim, 1),
                                    ConvResidualBlock(cfg.model_dim, 2),
                                    ConvResidualBlock(cfg.model_dim, 4))

    def forward(self, waveform):
        return self.blocks(self.stem(self.frontend(waveform)))


class VectorCodebook(nn.Module):
    """Straight-through VQ with explicit codebook and commitment losses."""
    def __init__(self, size: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(size, dim)
        nn.init.uniform_(self.embedding.weight, -1 / size, 1 / size)

    def forward(self, x):
        # x: B,T,D
        flat = x.reshape(-1, x.shape[-1])
        weight = self.embedding.weight
        distance = (flat.square().sum(1, keepdim=True) - 2 * flat @ weight.t()
                    + weight.square().sum(1).unsqueeze(0))
        codes = distance.argmin(1).view(x.shape[:-1])
        quantized = self.embedding(codes)
        codebook_loss = F.mse_loss(quantized, x.detach())
        commitment_loss = F.mse_loss(x, quantized.detach())
        straight_through = x + (quantized - x).detach()
        return straight_through, codes, codebook_loss + 0.25 * commitment_loss


@dataclass
class TokenizerOutput:
    quantized_levels: list[torch.Tensor]
    codes: list[torch.Tensor]
    reconstructed: torch.Tensor
    loss: torch.Tensor


class MultiscaleResidualVQ(nn.Module):
    """Genuine coarse-to-fine residual quantizer over multiple time scales.

    A coarse codebook first explains a pooled target. Each finer codebook sees
    only the residual left after all earlier levels are interpolated to its
    resolution. Their sum reconstructs the full-rate encoder latent.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.factors = cfg.level_factors
        self.proj_in = nn.Conv1d(cfg.model_dim, cfg.code_dim, 1)
        self.codebooks = nn.ModuleList([
            VectorCodebook(cfg.codebook_size, cfg.code_dim) for _ in self.factors
        ])

    @staticmethod
    def _resize(x, length):
        return F.interpolate(x.transpose(1, 2), size=length, mode="linear",
                             align_corners=False).transpose(1, 2)

    def forward(self, latent):
        target = self.proj_in(latent).transpose(1, 2)
        levels, codes, losses = [], [], []
        for factor, codebook in zip(self.factors, self.codebooks):
            length = max(1, math.ceil(target.shape[1] / factor))
            target_level = self._resize(target, length)
            explained = sum((self._resize(q, length) for q in levels),
                            torch.zeros_like(target_level))
            residual = target_level - explained
            quantized, indices, loss = codebook(residual)
            levels.append(quantized); codes.append(indices); losses.append(loss)
        reconstructed = sum((self._resize(q, target.shape[1]) for q in levels),
                            torch.zeros_like(target))
        reconstruction_loss = F.mse_loss(reconstructed, target.detach())
        return TokenizerOutput(levels, codes, reconstructed,
                               reconstruction_loss + torch.stack(losses).sum())


class SinusoidalPositions(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000) / dim))
        table = torch.zeros(max_len, dim)
        table[:, 0::2] = torch.sin(pos * scale)
        table[:, 1::2] = torch.cos(pos * scale)
        self.register_buffer("table", table, persistent=False)

    def forward(self, length):
        return self.table[:length]


class ParallelLevelPredictor(nn.Module):
    """Conditional next-level Transformer with parallel within-level logits."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.audio_proj = nn.Linear(cfg.model_dim, cfg.code_dim)
        self.positions = SinusoidalPositions(cfg.code_dim)
        self.level_embedding = nn.Embedding(len(cfg.level_factors), cfg.code_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            cfg.code_dim, cfg.transformer_heads, cfg.code_dim * 4,
            cfg.dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.transformer_layers)
        self.condition_proj = nn.Linear(cfg.code_dim, cfg.code_dim)
        self.heads = nn.ModuleList([nn.Linear(cfg.code_dim, cfg.codebook_size)
                                    for _ in cfg.level_factors])

    def forward_level(self, audio_latent, prior_quantized, level, length):
        memory = self.audio_proj(audio_latent.transpose(1, 2))
        batch = audio_latent.shape[0]
        query = self.positions(length).unsqueeze(0).expand(batch, -1, -1)
        query = query + self.level_embedding.weight[level].view(1, 1, -1)
        if prior_quantized:
            condition = sum(MultiscaleResidualVQ._resize(q, length)
                            for q in prior_quantized)
            query = query + self.condition_proj(condition)
        # No causal mask: every position at this level is computed concurrently.
        hidden = self.decoder(query, memory)
        return self.heads[level](hidden)

    def forward(self, audio_latent, target_levels):
        logits = []
        for level, target in enumerate(target_levels):
            logits.append(self.forward_level(audio_latent, target_levels[:level],
                                             level, target.shape[1]))
        return logits


class LatentDigitDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cfg.code_dim, cfg.model_dim, 3, padding=1), nn.GELU(),
            ConvResidualBlock(cfg.model_dim), nn.AdaptiveAvgPool1d(1))
        self.classifier = nn.Linear(cfg.model_dim, cfg.num_digits)

    def forward(self, reconstructed):
        return self.classifier(self.net(reconstructed.transpose(1, 2)).squeeze(-1))


class ParallelPossibilityFramework(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.encoder = AcousticEncoder(self.cfg)
        self.tokenizer = MultiscaleResidualVQ(self.cfg)
        self.predictor = ParallelLevelPredictor(self.cfg)
        self.decoder = LatentDigitDecoder(self.cfg)

    def forward(self, waveform, labels=None):
        latent = self.encoder(waveform)
        tokens = self.tokenizer(latent)
        level_logits = self.predictor(latent, tokens.quantized_levels)
        # A differentiable generated path trains the decoder for the same kind
        # of latent it receives at inference, while teacher VQ latents provide
        # an easier auxiliary classification target.
        soft_levels = [logit.softmax(-1) @ book.embedding.weight
                       for logit, book in zip(level_logits, self.tokenizer.codebooks)]
        generated_soft = sum(MultiscaleResidualVQ._resize(q, latent.shape[-1])
                             for q in soft_levels)
        digit_logits = self.decoder(generated_soft)
        teacher_digit_logits = self.decoder(tokens.reconstructed)
        losses = {"vq": tokens.loss}
        losses["codes"] = sum(F.cross_entropy(logit.flatten(0, 1), code.flatten())
                              for logit, code in zip(level_logits, tokens.codes))
        if labels is not None:
            losses["digit"] = (F.cross_entropy(digit_logits, labels)
                               + 0.5 * F.cross_entropy(teacher_digit_logits, labels))
        losses["total"] = losses["vq"] + losses["codes"] + losses.get("digit", 0)
        return {"digit_logits": digit_logits, "teacher_digit_logits": teacher_digit_logits,
                "level_logits": level_logits,
                "codes": tokens.codes, "losses": losses}

    def predict_latent(self, waveform, sample=False, temperature=1.0):
        """Run the actual coarse-to-fine generation path (no target-code input)."""
        latent = self.encoder(waveform)
        predicted_levels, predicted_codes = [], []
        full_length = latent.shape[-1]
        for level, (factor, codebook) in enumerate(zip(
                self.cfg.level_factors, self.tokenizer.codebooks)):
            length = max(1, math.ceil(full_length / factor))
            logits = self.predictor.forward_level(latent, predicted_levels, level, length)
            if sample:
                codes = torch.distributions.Categorical(logits=logits / temperature).sample()
            else:
                codes = logits.argmax(-1)
            predicted_codes.append(codes)
            predicted_levels.append(codebook.embedding(codes))
        reconstructed = sum(MultiscaleResidualVQ._resize(q, full_length)
                            for q in predicted_levels)
        return reconstructed, predicted_codes

    @torch.no_grad()
    def possibilities(self, waveform, top_k=3):
        generated, _ = self.predict_latent(waveform)
        probs = self.decoder(generated).softmax(-1)
        values, indices = probs.topk(min(top_k, self.cfg.num_digits), dim=-1)
        return [[{"digit": int(d), "probability": float(p)} for p, d in zip(v, i)]
                for v, i in zip(values.cpu(), indices.cpu())]


class CNNBaseline(nn.Module):
    """Direct discriminative baseline, deliberately without VQ/level prediction."""
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__(); self.cfg = cfg or ModelConfig()
        self.encoder = AcousticEncoder(self.cfg)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(self.cfg.model_dim, self.cfg.num_digits))

    def forward(self, waveform): return self.head(self.encoder(waveform))

