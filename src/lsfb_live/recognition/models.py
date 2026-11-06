"""Architectures de réseaux de neurones pour la reconnaissance de signes LSFB."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelArchitectureConfig:
    """Paramètres structurels du modèle de reconnaissance.

    Attributes:
        input_dim: Nombre de coordonnées d'entrée par frame (195).
        hidden_dim: Dimension interne des couches récurrentes ou d'attention (128).
        num_classes: Nombre de classes cibles (100).
        num_layers: Nombre de couches récurrentes ou blocs Transformer (2).
        dropout_rate: Taux de régularisation par dropout (0.3).
    """

    input_dim: int = 195
    hidden_dim: int = 128
    num_classes: int = 100
    num_layers: int = 2
    dropout_rate: float = 0.3


class BiGRUClassifier(nn.Module):
    """Classifieur séquentiel basé sur un GRU bidirectionnel avec double pooling."""

    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        )
        self.gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout_rate if config.num_layers > 1 else 0.0,
        )
        # Sortie bi-directionnelle = hidden_dim * 2
        gru_output_dim = config.hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, T, input_dim)
        projected = self.input_projection(x)  # (Batch, T, hidden_dim)
        gru_out, _ = self.gru(projected)  # (Batch, T, hidden_dim * 2)

        # Agrégation temporelle : moyenne globale + dernier pas temporel
        avg_pool = torch.mean(gru_out, dim=1)  # (Batch, hidden_dim * 2)
        last_step = gru_out[:, -1, :]  # (Batch, hidden_dim * 2)
        pooled = torch.cat([avg_pool, last_step], dim=-1)  # (Batch, hidden_dim * 4)

        return self.classifier(pooled)


class ConvTransformerClassifier(nn.Module):
    """Classifieur hybride combinant convolutions 1D temporelles et encodeur Transformer."""

    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=config.input_dim,
                out_channels=config.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=4,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, T, input_dim) -> transposé pour Conv1d : (Batch, input_dim, T)
        transposed = x.transpose(1, 2)
        convolved = self.conv1d(transposed).transpose(1, 2)  # (Batch, T, hidden_dim)
        encoded = self.transformer(convolved)  # (Batch, T, hidden_dim)

        pooled = torch.mean(encoded, dim=1)  # (Batch, hidden_dim)
        return self.classifier(pooled)

