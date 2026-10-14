"""
model.py
========
Multi-scale feature-fusion CNN.

The reference paper's central idea is that a single CNN layer is not enough:
early/mid layers carry texture and edge information, deep layers carry
fault-specific semantics, so features from TWO depths should be combined.

The paper does that offline in MATLAB: run the CNN, pull the pooling-layer and
FC-layer activations, wavelet-compress one of them, concatenate, select 50
features with chi-square, then hand the vector to an LDA classifier.

This implementation does the same fusion INSIDE the network and trains it
end-to-end, which is simpler to reproduce, differentiable throughout, and lets
the fusion weights adapt during training:

    ResNet-18:     layer3 (256ch, 14x14)  +  layer4 (512ch, 7x7)   -> 768-d
    MobileNetV2:   features[:14] (96ch)   +  features[14:] (1280ch) -> 1376-d

Each branch is global-average-pooled, batch-normalised (so the two scales are on
comparable footing before concatenation), concatenated, then classified.

`extract_features()` returns the fused vector before the classifier, which is
what feature_pipeline.py feeds to the classical LDA/kNN/SVM classifiers.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

import config


class MultiScaleNet(nn.Module):
    def __init__(
        self,
        backbone: str = config.BACKBONE,
        num_classes: int = 9,
        pretrained: bool = config.PRETRAINED,
        dropout: float = config.DROPOUT,
        freeze_backbone: bool = config.FREEZE_BACKBONE,
    ):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.resnet18(weights=weights)
            # Stage A: everything up to and including layer3  -> 256 channels
            self.stage_a = nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3,
            )
            # Stage B: layer4                                  -> 512 channels
            self.stage_b = net.layer4
            dim_a, dim_b = 256, 512

        elif backbone == "mobilenet_v2":
            weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            net = tvm.mobilenet_v2(weights=weights)
            self.stage_a = net.features[:14]    # 96 channels, stride 16
            self.stage_b = net.features[14:]    # 1280 channels, stride 32
            dim_a, dim_b = 96, 1280

        else:
            raise ValueError(f"backbone must be 'resnet18' or 'mobilenet_v2', got '{backbone}'")

        self.pool = nn.AdaptiveAvgPool2d(1)
        # BatchNorm on each branch: without it the 1280-d MobileNet branch would
        # numerically dominate the 96-d branch in the concatenated vector.
        self.norm_a = nn.BatchNorm1d(dim_a)
        self.norm_b = nn.BatchNorm1d(dim_b)
        self.feature_dim = dim_a + dim_b

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        if freeze_backbone:
            for p in self.stage_a.parameters():
                p.requires_grad = False
            for p in self.stage_b.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ #
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Fused multi-scale descriptor, shape (B, feature_dim)."""
        fa = self.stage_a(x)
        fb = self.stage_b(fa)
        va = self.pool(fa).flatten(1)
        vb = self.pool(fb).flatten(1)
        return torch.cat([self.norm_a(va), self.norm_b(vb)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))

    # ------------------------------------------------------------------ #
    def param_groups(self, lr_backbone: float, lr_head: float):
        """Discriminative learning rates: pretrained weights move slowly."""
        backbone_params = [p for p in list(self.stage_a.parameters()) +
                           list(self.stage_b.parameters()) if p.requires_grad]
        head_params = (list(self.norm_a.parameters()) +
                       list(self.norm_b.parameters()) +
                       list(self.classifier.parameters()))
        groups = [{"params": head_params, "lr": lr_head}]
        if backbone_params:
            groups.insert(0, {"params": backbone_params, "lr": lr_backbone})
        return groups

    def last_conv_layer(self) -> nn.Module:
        """Target layer for Grad-CAM (deepest spatial feature map)."""
        return self.stage_b


def build_model(num_classes: int, **kwargs) -> MultiScaleNet:
    try:
        return MultiScaleNet(num_classes=num_classes, **kwargs)
    except Exception as e:
        # Almost always: no internet on the first run, so torchvision cannot
        # fetch the ImageNet checkpoint. It is cached after one success.
        if kwargs.get("pretrained", config.PRETRAINED):
            raise RuntimeError(
                "Could not load pretrained ImageNet weights.\n"
                f"  underlying error: {e}\n"
                "  Fix: connect to the internet for the first run (weights are cached\n"
                "  in ~/.cache/torch/hub afterwards), or set PRETRAINED = False in\n"
                "  config.py to train from scratch. Note that training from scratch on\n"
                "  255 images will perform far worse - transfer learning is doing most\n"
                "  of the work here."
            ) from e
        raise


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
