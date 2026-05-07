import torch
import torch.nn as nn
from torchvision import models

def get_optical_classifier(num_classes=4):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model

class ResidualBlock(nn.Module):
    """Core building block of the CycleGAN generator."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-based encoder-decoder generator.
    G_AB: SAR (1ch) -> Optical (3ch)
    """
    def __init__(self, in_channels, out_channels, ngf=64, n_res_blocks=6):
        super().__init__()
        layers = [
            # Encoder
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf,   ngf*2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(inplace=True),
        ]

        # Transformer
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(ngf*4))

        layers += [
            # Decoder
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*2, ngf,   3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()  # output in [-1, 1]
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN: classifies overlapping image patches as real/fake.
    """
    def __init__(self, in_channels, ndf=64):
        super().__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, ndf,   normalize=False),
            *block(ndf,         ndf*2),
            *block(ndf*2,       ndf*4),
            *block(ndf*4,       ndf*8, stride=1),
             nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)
