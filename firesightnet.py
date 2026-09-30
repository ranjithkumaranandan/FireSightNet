"""
FireSightNet: Multi-Class Forest Fire and Smoke Detection
Paper: Expert Systems with Applications
Author: Ranjith Kumar Anandan, SRM Institute of Science and Technology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


# ─────────────────────────────────────────────
# 1. CBAM — Channel + Spatial Attention
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        return self.sigmoid(avg + mx).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x


# ─────────────────────────────────────────────
# 2. Multi-Head Transformer Self-Attention
# ─────────────────────────────────────────────
class TransformerSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)          # (B, H*W, C)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm(tokens + self.dropout(attn_out))
        return tokens.permute(0, 2, 1).reshape(b, c, h, w)


# ─────────────────────────────────────────────
# 3. Feature Pyramid Network (FPN)
# ─────────────────────────────────────────────
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.output = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        # features: list of feature maps from shallow to deep
        laterals = [l(f) for l, f in zip(self.lateral, features)]
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='nearest'
            )
        return [o(l) for o, l in zip(self.output, laterals)]


# ─────────────────────────────────────────────
# 4. Generalized Mean Pooling (GeM)
# ─────────────────────────────────────────────
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), 1
        ).pow(1.0 / self.p)


# ─────────────────────────────────────────────
# 5. Focal Cross-Entropy Loss
# ─────────────────────────────────────────────
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, num_classes=4):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction='none'
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─────────────────────────────────────────────
# 6. FireSightNet — Full Model
# ─────────────────────────────────────────────
class FireSightNet(nn.Module):
    """
    FireSightNet v3: EfficientNet-B4 + Dual Attention + FPN + Triple Pooling
    4-class forest fire and smoke detection.
    """

    def __init__(
        self,
        num_classes=4,
        transformer_heads=8,
        use_transformer=True,
        use_cbam=True,
        use_fpn=True,
        use_triple_pooling=True,
        dropout1=0.4,
        dropout2=0.3,
        pretrained=True,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        self.use_cbam = use_cbam
        self.use_fpn = use_fpn
        self.use_triple_pooling = use_triple_pooling

        # ── Backbone ──────────────────────────────
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=pretrained, features_only=True
        )
        # EfficientNet-B4 feature channel sizes at stages 1-4
        self.fpn_channels = [24, 32, 56, 160, 1792]
        feat_dim = 1792   # deepest stage output channels

        # ── Dual Attention ─────────────────────────
        if use_transformer:
            self.transformer = TransformerSelfAttention(feat_dim, transformer_heads)
        if use_cbam:
            self.cbam = CBAM(feat_dim)

        # ── FPN ────────────────────────────────────
        if use_fpn:
            self.fpn = FPN(self.fpn_channels, out_channels=256)
            pool_in = 256   # FPN output channels (deepest level)
        else:
            pool_in = feat_dim

        # ── Triple Pooling ─────────────────────────
        if use_triple_pooling:
            self.gem = GeM(p=3.0)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.gmp = nn.AdaptiveMaxPool2d(1)
            pool_out_dim = pool_in * 3    # 768-d when pool_in=256
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
            pool_out_dim = pool_in

        # ── Classifier head ────────────────────────
        self.fc1 = nn.Linear(pool_out_dim, 512)
        self.bn1  = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(512, 256)
        self.bn2  = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout2)

        self.classifier = nn.Linear(256, num_classes)
        self.act = nn.GELU()

    def forward(self, x):
        # 1. Extract multi-scale features from backbone
        features = self.backbone(x)   # list of 5 feature maps

        feat = features[-1]           # deepest feature map

        # 2. Dual Attention
        if self.use_transformer:
            feat = self.transformer(feat)
        if self.use_cbam:
            feat = self.cbam(feat)

        # 3. FPN
        if self.use_fpn:
            fpn_outs = self.fpn(features)
            feat = fpn_outs[-1]       # use deepest FPN output

        # 4. Triple Pooling
        if self.use_triple_pooling:
            gem_out = self.gem(feat).flatten(1)
            gap_out = self.gap(feat).flatten(1)
            gmp_out = self.gmp(feat).flatten(1)
            f_pool  = torch.cat([gem_out, gap_out, gmp_out], dim=1)  # 768-d
        else:
            f_pool = self.gap(feat).flatten(1)

        # 5. Classifier
        h1 = self.drop1(self.act(self.bn1(self.fc1(f_pool))))   # h1 ∈ R^512
        h2 = self.drop2(self.act(self.bn2(self.fc2(h1))))       # h2 ∈ R^256
        return self.classifier(h2)


# ─────────────────────────────────────────────
# Factory helpers for ablation variants
# ─────────────────────────────────────────────
def build_variant(variant: str, num_classes=4, pretrained=True) -> FireSightNet:
    """
    Build a specific ablation variant.
    A: Backbone only
    B: + CBAM
    C: + Transformer
    D: + FPN
    E: + CBAM + FPN
    F: + Transformer + CBAM
    G: Full FireSightNet (all components)
    """
    configs = {
        'A': dict(use_transformer=False, use_cbam=False, use_fpn=False, use_triple_pooling=False),
        'B': dict(use_transformer=False, use_cbam=True,  use_fpn=False, use_triple_pooling=False),
        'C': dict(use_transformer=True,  use_cbam=False, use_fpn=False, use_triple_pooling=False),
        'D': dict(use_transformer=False, use_cbam=False, use_fpn=True,  use_triple_pooling=False),
        'E': dict(use_transformer=False, use_cbam=True,  use_fpn=True,  use_triple_pooling=False),
        'F': dict(use_transformer=True,  use_cbam=True,  use_fpn=False, use_triple_pooling=False),
        'G': dict(use_transformer=True,  use_cbam=True,  use_fpn=True,  use_triple_pooling=True),
    }
    cfg = configs[variant.upper()]
    return FireSightNet(num_classes=num_classes, pretrained=pretrained, **cfg)


if __name__ == '__main__':
    model = FireSightNet()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")   # (2, 4)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params / 1e6:.2f}M")
