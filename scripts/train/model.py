import torch
import torch.nn as nn


class TorchLogReg(nn.Module):
    """
    Multiclass text classifier on top of TF-IDF features.

    NOTE:
    Historically this was a pure logistic regression (single Linear layer).
    We have upgraded it to a shallow MLP for higher capacity while keeping
    the same class name and constructor signature so existing training code
    continues to work.

    Architecture (by default):
        input_dim -> Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear -> num_classes
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = []

        # First projection into a smaller dense space
        layers.append(nn.Linear(input_dim, hidden_dim))

        # Optional BatchNorm to stabilise training on sparse-ish TF-IDF features
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))

        layers.append(nn.ReLU(inplace=True))

        # Dropout for regularisation
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Final classifier head
        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration module for logits.
    logT as a learnable parameter ensures numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.logT)
        return logits / T
