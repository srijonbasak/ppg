from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    sample_rate: int = 8000
    n_fft: int = 256
    hop_length: int = 80
    n_mels: int = 40
    model_dim: int = 64
    code_dim: int = 64
    codebook_size: int = 64
    level_factors: tuple[int, ...] = (4, 2, 1)
    transformer_layers: int = 2
    transformer_heads: int = 4
    dropout: float = 0.1
    num_digits: int = 10

    def to_dict(self):
        data = asdict(self)
        data["level_factors"] = list(self.level_factors)
        return data

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["level_factors"] = tuple(data.get("level_factors", (4, 2, 1)))
        return cls(**data)

