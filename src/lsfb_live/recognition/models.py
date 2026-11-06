"""Architectures de réseaux de neurones pour la reconnaissance de signes LSFB."""

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelArchitectureConfig:
    """Paramètres structurels du modèle de reconnaissance.

    Attributes:
        input_dim: Nombre de coordonnées d'entrée par frame (195 ou 390 avec vitesse).
        hidden_dim: Dimension interne des couches récurrentes ou d'attention (144).
        num_classes: Nombre de classes cibles (100).
        num_layers: Nombre de couches récurrentes ou blocs Transformer (2).
        dropout_rate: Taux de régularisation par dropout (0.25).
    """

    input_dim: int = 195
    hidden_dim: int = 144
    num_classes: int = 100
    num_layers: int = 2
    dropout_rate: float = 0.25


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


class MSSTClassifier(nn.Module):
    """Multi-Scale Spatio-Temporal Transformer avec Attention Pooling apprenable."""

    def __init__(self, config: ModelArchitectureConfig):
        super().__init__()
        # 3 branches de convolution temporelle multi-échelles (k=3, 5, 7)
        branch_channels = config.hidden_dim // 3
        total_conv_channels = branch_channels * 3

        self.conv_k3 = nn.Conv1d(config.input_dim, branch_channels, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(config.input_dim, branch_channels, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(config.input_dim, branch_channels, kernel_size=7, padding=3)

        self.conv_norm = nn.BatchNorm1d(total_conv_channels)
        self.conv_act = nn.GELU()
        self.conv_drop = nn.Dropout(config.dropout_rate)

        # Plongement positionnel apprenable pour 40 frames
        self.pos_embedding = nn.Parameter(torch.randn(1, 40, total_conv_channels) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_conv_channels,
            nhead=4,
            dim_feedforward=total_conv_channels * 2,
            dropout=config.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Mécanisme d'attention temporelle pour pondérer les trames discriminantes
        self.attention_pool = nn.Sequential(
            nn.Linear(total_conv_channels, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(total_conv_channels),
            nn.Dropout(config.dropout_rate),
            nn.Linear(total_conv_channels, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, T, input_dim) -> transposé pour Conv1d : (Batch, input_dim, T)
        transposed = x.transpose(1, 2)
        c3 = self.conv_k3(transposed)
        c5 = self.conv_k5(transposed)
        c7 = self.conv_k7(transposed)

        multi_scale = torch.cat([c3, c5, c7], dim=1)  # (Batch, total_conv_channels, T)
        convolved = self.conv_drop(self.conv_act(self.conv_norm(multi_scale)))
        convolved = convolved.transpose(1, 2)  # (Batch, T, total_conv_channels)

        # Ajout du plongement positionnel
        sequence_length = convolved.size(1)
        encoded = convolved + self.pos_embedding[:, :sequence_length, :]
        transformed = self.transformer(encoded)  # (Batch, T, total_conv_channels)

        # Attention Pooling temporel apprenable
        attn_scores = self.attention_pool(transformed)  # (Batch, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (Batch, T, 1)
        pooled = torch.sum(transformed * attn_weights, dim=1)  # (Batch, total_conv_channels)

        return self.classifier(pooled)


