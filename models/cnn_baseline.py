import torch
import torch.nn as nn
from torchvision import models


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        self.feature_dim = model.fc.in_features
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_last_block(self):
        for p in self.backbone[-1].parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
