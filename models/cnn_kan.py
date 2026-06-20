import torch
import torch.nn as nn
from torchvision import models
from efficient_kan import KAN


class ResNet18KAN(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True,
                 kan_hidden_dims=None):
        super().__init__()
        if kan_hidden_dims is None:
            kan_hidden_dims = [256]
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        self.feature_dim = model.fc.in_features
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.kan = KAN([self.feature_dim] + list(kan_hidden_dims) + [num_classes])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_last_block(self):
        for p in self.backbone[-1].parameters():
            p.requires_grad = True

    def regularization_loss(self):
        if hasattr(self.kan, 'regularization_loss'):
            return self.kan.regularization_loss()
        return 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.kan(x)
