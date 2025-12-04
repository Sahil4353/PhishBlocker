import torch
import torch.nn as nn

class TorchLogReg(nn.Module):
    """
    Simple Linear Model:
    - Multiclass Logistic Regression in feature space (TF-IDF)
    - Works perfectly with sparse input -> dense batches
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration module for logits.
    logT as learnable param ensures stability.
    """
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        T = torch.exp(self.logT)
        return logits / T

