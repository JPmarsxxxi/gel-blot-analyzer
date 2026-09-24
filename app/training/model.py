"""Small U-Net for band segmentation.

Deliberately small (few million params) so inference is fast on CPU in
production, not just on the training GPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BandUNet(nn.Module):
    """4-level U-Net, single input channel (grayscale), single output channel (band logit)."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c1, c2, c3, c4, cb = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )

        self.enc1 = _conv_block(1, c1)
        self.enc2 = _conv_block(c1, c2)
        self.enc3 = _conv_block(c2, c3)
        self.enc4 = _conv_block(c3, c4)
        self.bottleneck = _conv_block(c4, cb)

        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(cb, c4, 2, stride=2)
        self.dec4 = _conv_block(cb, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = _conv_block(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = _conv_block(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = _conv_block(c2, c1)

        self.out_conv = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # logits, shape (N, 1, H, W)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    target = target.flatten(1)
    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce + dice_loss(logits, target)
