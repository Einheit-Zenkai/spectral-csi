"""
Bayesian Occupancy Network — ResNet-18 + MC Dropout
=====================================================
Implements a Bayesian Deep Learning approach to occupancy classification:

- **Backbone**: ResNet-18 (pretrained on ImageNet, fine-tuned on CSI spectrograms)
- **Bayesian inference**: Monte Carlo Dropout at test time
- **Output**: occupancy probability *mean* + *uncertainty* (epistemic)

Why Bayesian?
-------------
A standard CNN outputs a single softmax score — 0.92 could mean "very confident"
or "the model has never seen anything like this."  MC Dropout runs T stochastic
forward passes, producing a distribution of predictions:

    mean  → best-guess probability
    var   → model uncertainty (epistemic)

The decision module uses both:
    - High mean, low variance  → confident occupied → lights ON
    - Low mean, low variance   → confident empty   → lights OFF
    - Any high variance        → uncertain          → lights ON (fail-safe)

Reference
---------
Gal & Ghahramani, "Dropout as a Bayesian Approximation," ICML 2016.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class BayesianModelConfig:
    """Hyperparameters for the Bayesian occupancy network."""

    # MC Dropout
    dropout_rate: float = 0.2     # Applied before final FC layer
    mc_samples: int = 30          # Number of stochastic forward passes at inference

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 30

    # Input
    input_channels: int = 1        # Single-channel spectrogram
    input_size: tuple[int, int] = (224, 224)

    # Architecture
    num_classes: int = 2            # empty / occupied
    pretrained: bool = True

    # Decision thresholds
    occupancy_threshold: float = 0.5
    uncertainty_limit: float = 0.15  # Variance above this → fail-safe "occupied"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BayesianOccupancyNet(nn.Module):
    """
    ResNet-18 with MC Dropout for probabilistic occupancy detection.

    The key modification vs. a standard ResNet-18:
    1. First conv is changed from 3-channel to 1-channel (CSI spectrograms are mono).
    2. A Dropout layer is inserted before the final FC.
    3. At inference, dropout remains ON → stochastic predictions → uncertainty.

    Usage
    -----
    >>> model = BayesianOccupancyNet()
    >>> probs, uncertainty = model.predict_with_uncertainty(spectrogram_batch)
    """

    def __init__(self, config: BayesianModelConfig | None = None):
        super().__init__()
        self.cfg = config or BayesianModelConfig()

        # Load pretrained ResNet-18
        weights = ResNet18_Weights.IMAGENET1K_V1 if self.cfg.pretrained else None
        base = resnet18(weights=weights)

        # --- Modify first conv: 3 → 1 channel ---
        old_conv = base.conv1
        self.conv1 = nn.Conv2d(
            self.cfg.input_channels, 64,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Initialise from pretrained weights (average across RGB channels)
        if self.cfg.pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Copy the rest of ResNet-18
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # --- MC Dropout + final classifier ---
        self.dropout = nn.Dropout(p=self.cfg.dropout_rate)
        self.fc = nn.Linear(base.fc.in_features, self.cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.

        Parameters
        ----------
        x : (B, 1, H, W) float tensor — CSI spectrogram batch.

        Returns
        -------
        logits : (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)       # <── stays ON during inference for MC Dropout
        x = self.fc(x)
        return x

    # ------------------------------------------------------------------ #
    #  MC Dropout Inference                                                #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, T: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run T stochastic forward passes with dropout enabled.

        Parameters
        ----------
        x : (B, 1, H, W) input batch
        T : number of MC samples (default: self.cfg.mc_samples)

        Returns
        -------
        mean_probs   : (B,) mean predicted P(occupied)
        uncertainty  : (B,) variance of P(occupied) across T passes
        """
        T = T or self.cfg.mc_samples
        self.train()  # Keep dropout ON

        all_probs = []
        for _ in range(T):
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(occupied)
            all_probs.append(probs.cpu().numpy())

        all_probs = np.stack(all_probs, axis=0)  # (T, B)
        mean_probs = all_probs.mean(axis=0)       # (B,)
        uncertainty = all_probs.var(axis=0)        # (B,)

        return mean_probs, uncertainty

    def decide(
        self, x: torch.Tensor, T: int | None = None
    ) -> list[dict]:
        """
        Full decision pipeline: MC inference → threshold → verdict.

        Returns
        -------
        List of dicts (one per sample in batch):
            p_occupied  — mean occupancy probability
            uncertainty — epistemic uncertainty (variance)
            decision    — "occupied" or "empty"
            fail_safe   — True if uncertainty forced "occupied"
        """
        mean_probs, uncertainty = self.predict_with_uncertainty(x, T)
        cfg = self.cfg
        results = []
        for p, u in zip(mean_probs, uncertainty):
            fail_safe = bool(u > cfg.uncertainty_limit)
            if fail_safe:
                decision = "occupied"  # Don't trust uncertain predictions
            else:
                decision = "occupied" if p >= cfg.occupancy_threshold else "empty"

            results.append({
                "p_occupied": float(p),
                "uncertainty": float(u),
                "decision": decision,
                "fail_safe": fail_safe,
            })
        return results


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def create_trainer(
    model: BayesianOccupancyNet,
    device: str = "cpu",
) -> dict:
    """
    Create optimizer, loss, and scheduler for training.

    Returns dict with 'optimizer', 'criterion', 'scheduler'.
    """
    cfg = model.cfg
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )
    return {
        "optimizer": optimizer,
        "criterion": criterion,
        "scheduler": scheduler,
    }


def train_one_epoch(
    model: BayesianOccupancyNet,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class CSISpectrogramDataset(torch.utils.data.Dataset):
    """
    Wraps numpy arrays (from SyntheticCSIGenerator + SpectrumAnalyzer)
    into a PyTorch Dataset.

    Parameters
    ----------
    spectrograms : (N, 1, H, W) float32 array
    labels       : (N,) int array — 0=empty, 1=occupied
    """

    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray):
        self.spectrograms = torch.from_numpy(spectrograms).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.synthetic_csi import SyntheticCSIGenerator
    from core.spectrum_analyzer import SpectrumAnalyzer

    print("Building Bayesian ResNet-18 for CSI occupancy detection...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = BayesianOccupancyNet()
    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters : {n_params:,}")
    print(f"Trainable params : {n_train:,}")

    # Generate a small synthetic dataset
    print("\nGenerating synthetic spectrograms...")
    gen = SyntheticCSIGenerator()
    sa = SpectrumAnalyzer(sample_rate=100.0)

    X_windows, y_labels = gen.generate_labelled(n_occupied=8, n_empty=8, duration=10.0)

    # Convert each window to a spectrogram image
    spectrograms = []
    for i in range(len(y_labels)):
        img = sa.to_spectrogram_image(X_windows[i])  # (1, 224, 224)
        spectrograms.append(img)
    spectrograms = np.stack(spectrograms)  # (N, 1, 224, 224)
    print(f"Spectrogram batch shape: {spectrograms.shape}")
    print(f"Labels: {y_labels}")

    # Quick inference demo (untrained model — random predictions)
    print("\n--- MC Dropout Inference (untrained model) ---")
    X_tensor = torch.from_numpy(spectrograms[:4]).float().to(device)
    results = model.decide(X_tensor)
    for i, r in enumerate(results):
        label = "occupied" if y_labels[i] == 1 else "empty"
        print(f"  Sample {i} (true={label}): "
              f"P(occ)={r['p_occupied']:.3f}  "
              f"unc={r['uncertainty']:.4f}  "
              f"→ {r['decision']}"
              f"{'  [FAIL-SAFE]' if r['fail_safe'] else ''}")

    print("\n✓ Model architecture verified. Ready for training with real/larger synthetic data.")
