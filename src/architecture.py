"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
    
    Stabile U-Net Architektur für Image Inpainting
    Fokus auf STABILITÄT und gute Ergebnisse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Convolutional Block - stabil und effektiv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    """Encoder: ConvBlock + MaxPool"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder: Upsample + Concat + ConvBlock"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResBlock(nn.Module):
    """Simple Residual Block"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.conv(x) + x)


class MyModel(nn.Module):
    """
    Stabiles U-Net für Image Inpainting
    Einfach, effektiv, keine NaN-Probleme
    """
    
    def __init__(self, n_in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        
        bc = base_channels
        
        # Encoder
        self.enc1 = EncoderBlock(n_in_channels, bc)       # 4 -> 64
        self.enc2 = EncoderBlock(bc, bc * 2)              # 64 -> 128
        self.enc3 = EncoderBlock(bc * 2, bc * 4)          # 128 -> 256
        self.enc4 = EncoderBlock(bc * 4, bc * 8)          # 256 -> 512
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(bc * 8, bc * 8),
            ResBlock(bc * 8),
            ResBlock(bc * 8),
        )
        
        # Decoder
        self.dec4 = DecoderBlock(bc * 8, bc * 8, bc * 4)
        self.dec3 = DecoderBlock(bc * 4, bc * 4, bc * 2)
        self.dec2 = DecoderBlock(bc * 2, bc * 2, bc)
        self.dec1 = DecoderBlock(bc, bc, bc)
        
        # Output
        self.output = nn.Sequential(
            nn.Conv2d(bc, bc // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bc // 2, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        input_img = x[:, :3, :, :]
        known_mask = x[:, 3:4, :, :]
        
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        output = self.output(x)
        
        # Merge: bekannte Pixel behalten, unbekannte rekonstruieren
        return input_img * known_mask + output * (1 - known_mask)


class CombinedLoss(nn.Module):
    """Einfache Loss-Funktion: MSE + L1"""
    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 1.0, 
                 perceptual_weight: float = 0.0, use_perceptual: bool = False):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, output, target):
        return self.mse_weight * self.mse(output, target) + self.l1_weight * self.l1(output, target)