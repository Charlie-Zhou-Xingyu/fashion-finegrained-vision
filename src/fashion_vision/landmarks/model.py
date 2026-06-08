"""
Garment landmark prediction models.

Current baseline:
    ResNet18 coordinate regression model.

Input:
    image tensor [B, 3, H, W], range [0, 1]

Output:
    landmarks [B, max_landmarks, 2], normalized coordinates in [0, 1]
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetLandmarkPredictor(nn.Module):
    """
    ResNet-based garment landmark coordinate regressor.

    This is a simple baseline:
        ResNet18 backbone
            -> MLP head
            -> max_landmarks * 2 normalized coordinates

    Args:
        max_landmarks: Maximum number of landmarks.
        pretrained: Whether to use ImageNet pretrained ResNet weights.
        hidden_dim: Hidden dimension of MLP head.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        max_landmarks: int = 39,
        pretrained: bool = True,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.max_landmarks = int(max_landmarks)

        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        backbone = models.resnet18(weights=weights)
        in_features = int(backbone.fc.in_features)

        # Remove original classification head.
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, self.max_landmarks * 2),
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: [B, 3, H, W]

        Returns:
            landmarks: [B, max_landmarks, 2], normalized to [0, 1]
        """
        features = self.backbone(images)
        coords = self.head(features)
        coords = self.output_activation(coords)
        coords = coords.view(images.shape[0], self.max_landmarks, 2)
        return coords


def masked_smooth_l1_loss(
    pred_landmarks: torch.Tensor,
    target_landmarks: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """
    Smooth L1 loss only on valid landmarks.

    Args:
        pred_landmarks: [B, K, 2]
        target_landmarks: [B, K, 2]
        valid_mask: [B, K], 1 for valid landmark.
        beta: SmoothL1 beta.

    Returns:
        Scalar loss.
    """
    valid_mask = valid_mask.float()

    if valid_mask.sum() <= 0:
        return pred_landmarks.sum() * 0.0

    loss_fn = nn.SmoothL1Loss(reduction="none", beta=beta)
    loss = loss_fn(pred_landmarks, target_landmarks)  # [B, K, 2]

    loss = loss * valid_mask.unsqueeze(-1)
    loss = loss.sum() / (valid_mask.sum() * 2.0 + 1e-6)

    return loss


@torch.no_grad()
def compute_landmark_metrics(
    pred_landmarks: torch.Tensor,
    target_landmarks: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute simple normalized coordinate metrics.

    Args:
        pred_landmarks: [B, K, 2]
        target_landmarks: [B, K, 2]
        valid_mask: [B, K]

    Returns:
        Metrics dict.
    """
    valid_mask = valid_mask.float()

    if valid_mask.sum() <= 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "pck_005": 0.0,
            "pck_010": 0.0,
            "num_valid": 0.0,
        }

    diff = pred_landmarks - target_landmarks
    abs_diff = torch.abs(diff)

    mae = (abs_diff.sum(dim=-1) * valid_mask).sum() / (valid_mask.sum() * 2.0 + 1e-6)

    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # [B, K]
    rmse = torch.sqrt(((dist ** 2) * valid_mask).sum() / (valid_mask.sum() + 1e-6))

    pck_005 = (((dist <= 0.05).float() * valid_mask).sum() / (valid_mask.sum() + 1e-6))
    pck_010 = (((dist <= 0.10).float() * valid_mask).sum() / (valid_mask.sum() + 1e-6))

    return {
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "pck_005": float(pck_005.item()),
        "pck_010": float(pck_010.item()),
        "num_valid": float(valid_mask.sum().item()),
    }


def build_landmark_model(
    model_name: str = "resnet18",
    max_landmarks: int = 39,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build landmark model.

    Args:
        model_name: Model name.
        max_landmarks: Maximum landmark count.
        pretrained: Whether to use pretrained weights.

    Returns:
        Model.
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet18":
        return ResNetLandmarkPredictor(
            max_landmarks=max_landmarks,
            pretrained=pretrained,
        )

    raise ValueError(f"Unsupported landmark model: {model_name}")
