# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvmodels

def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """Compute Gram matrix for a batch of feature maps."""
    B, C, H, W = feat.size()
    f = feat.view(B, C, H * W)
    return torch.bmm(f, f.transpose(1, 2)) / (C * H * W)

class VGGFeatures(nn.Module):
    """Extract specified layers from a frozen VGG19."""
    def __init__(self, content_layers: dict, style_layers: dict):
        super().__init__()
        vgg = tvmodels.vgg19(pretrained=True).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.target_layers = {**content_layers, **style_layers}

    def forward(self, x: torch.Tensor):
        feats = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.target_layers:
                feats[self.target_layers[i]] = x
        return feats

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in1   = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.in2   = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor):
        y = F.relu(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        return x + y

class TransformerNet(nn.Module):
    """Feed-forward style transfer network for spectrogram “images.”"""
    def __init__(self):
        super().__init__()
        # initial conv
        self.conv1 = nn.Conv2d(3, 32, 9, padding=4)
        self.in1   = nn.InstanceNorm2d(32, affine=True)
        # downsampling
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
        )
        # residual blocks
        self.res = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        # upsampling
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
        )
        # final conv
        self.conv2 = nn.Conv2d(32, 3, 9, padding=4)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.in1(self.conv1(x)))
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return self.conv2(x)
