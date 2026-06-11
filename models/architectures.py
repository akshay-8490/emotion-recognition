"""
models/architectures.py — PyTorch model class definitions.
Zero Streamlit dependency — safe to import anywhere, test independently.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """Conv residual block with SE attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se    = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + identity)


class AudioCNN(nn.Module):
    """
    CNN for Speech Emotion Recognition from mel-spectrogram inputs.
    Input shape: (batch, 3, n_mels, time_frames)
      — 3 channels = mel-dB, delta, delta-delta
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.res1    = ResidualBlock(32)
        self.pool1   = nn.MaxPool2d(2)
        self.drop1   = nn.Dropout(0.2)

        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.res2    = ResidualBlock(64)
        self.pool2   = nn.MaxPool2d(2)
        self.drop2   = nn.Dropout(0.3)

        self.conv3   = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.pool3   = nn.MaxPool2d(2)
        self.drop3   = nn.Dropout(0.3)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.fc1     = nn.Linear(128, 64)
        self.drop_fc = nn.Dropout(0.4)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.res1(F.relu(self.bn1(self.conv1(x))))))
        x = self.drop2(self.pool2(self.res2(F.relu(self.bn2(self.conv2(x))))))
        x = self.drop3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.gap(x).view(x.size(0), -1)
        x = self.drop_fc(F.relu(self.fc1(x)))
        return self.fc2(x)


class FusionMLP(nn.Module):
    """
    Learned late-fusion MLP.
    Input: concatenation of [face_probs(3) | audio_probs(3) | ctx_prior(3) | reliabilities(2)]
           = 11-dimensional feature vector by default.
    """

    def __init__(self, input_dim: int = 11, hidden: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
