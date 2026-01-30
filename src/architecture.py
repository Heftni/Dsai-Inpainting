"""
    Author: Niklas Heftberger
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
    
    ULTIMATE U-Net Architektur für Image Inpainting - RTX 5090 Optimiert
    Features:
    - Self-Attention für globale Zusammenhänge
    - CBAM (Channel + Spatial Attention)
    - Dilated Convolutions für großes Receptive Field
    - Multi-Scale Context Aggregation (ASPP)
    - Gated Convolutions für bessere Inpainting-Qualität
    - Deep Residual Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Sicherstellen dass reduced_channels mindestens 1 ist
        reduced_channels = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-Attention Module für globale Zusammenhänge
    Besonders wichtig für Inpainting um entfernte Kontextinformationen zu nutzen
    """
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Queries, Keys, Values
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B x HW x C'
        key = self.key(x).view(B, -1, H * W)  # B x C' x HW
        value = self.value(x).view(B, -1, H * W)  # B x C x HW
        
        # Attention
        attention = self.softmax(torch.bmm(query, key))  # B x HW x HW
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x


class GatedConv2d(nn.Module):
    """
    Gated Convolution - besser für Inpainting als normale Convolution
    Lernt automatisch welche Features wichtig sind
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        conv_out = self.conv(x)
        gate_out = torch.sigmoid(self.gate(x))
        return self.bn(conv_out * gate_out)


class ResidualBlock(nn.Module):
    """Verbesserter Residual Block mit Gated Conv und Attention"""
    def __init__(self, channels: int, use_attention: bool = True, use_gated: bool = True):
        super().__init__()
        if use_gated:
            self.conv1 = GatedConv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = GatedConv2d(channels, channels, kernel_size=3, padding=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels)
            )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.attention = CBAM(channels) if use_attention else nn.Identity()
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.attention(out)
        out = out + residual
        out = self.relu(out)
        return out


class DilatedResidualBlock(nn.Module):
    """Residual Block mit Dilated Convolutions für größeres receptive field"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = GatedConv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.conv2 = GatedConv2d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.conv3 = GatedConv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = out + residual
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):
    """Encoder Block mit Gated Conv und Attention"""
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = GatedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = GatedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = GatedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.attention(x)
        pooled = self.pool(x)
        return x, pooled


class DecoderBlock(nn.Module):
    """Decoder Block mit Attention und Gated Conv"""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = GatedConv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = GatedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = GatedConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
        
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.attention(x)
        return x


class ContextModule(nn.Module):
    """Multi-Scale Context Aggregation (ASPP-style) mit mehr Skalen"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=8, dilation=8)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.fusion = nn.Conv2d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.relu(self.bn(out))
        out = self.fusion(out)
        return out + x


class MyModel(nn.Module):
    """
    ULTIMATE U-Net für Image Inpainting - RTX 5090 Edition
    
    Features:
    - Gated Convolutions für bessere Inpainting-Qualität
    - Self-Attention für globale Zusammenhänge
    - CBAM Attention in Encoder und Decoder
    - Dilated Convolutions für großes Receptive Field
    - Multi-Scale Context Aggregation (ASPP)
    - Tiefes Residual Learning
    - Progressive Feature Refinement
    """
    
    def __init__(self, n_in_channels: int = 4, base_channels: int = 64):
        super().__init__()
        
        # Initial feature extraction mit Gated Conv
        self.initial = nn.Sequential(
            GatedConv2d(n_in_channels, base_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            GatedConv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder path - 5 Stufen für mehr Tiefe
        self.enc1 = EncoderBlock(base_channels, base_channels, use_attention=True)
        self.enc2 = EncoderBlock(base_channels, base_channels * 2, use_attention=True)
        self.enc3 = EncoderBlock(base_channels * 2, base_channels * 4, use_attention=True)
        self.enc4 = EncoderBlock(base_channels * 4, base_channels * 8, use_attention=True)
        
        # Bottleneck mit Self-Attention für globalen Kontext
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8, use_attention=True),
            DilatedResidualBlock(base_channels * 8),
            SelfAttention(base_channels * 8),  # Globale Attention
            ContextModule(base_channels * 8),
            DilatedResidualBlock(base_channels * 8),
            SelfAttention(base_channels * 8),  # Zweite globale Attention
            ResidualBlock(base_channels * 8, use_attention=True),
        )
        
        # Decoder path - Skip channels müssen zu Encoder output passen!
        # enc4 output: base_channels * 8, enc3 output: base_channels * 4, etc.
        self.dec4 = DecoderBlock(base_channels * 8, base_channels * 8, base_channels * 4, use_attention=True)  # skip von enc4
        self.dec3 = DecoderBlock(base_channels * 4, base_channels * 4, base_channels * 2, use_attention=True)  # skip von enc3
        self.dec2 = DecoderBlock(base_channels * 2, base_channels * 2, base_channels, use_attention=True)      # skip von enc2
        self.dec1 = DecoderBlock(base_channels, base_channels, base_channels, use_attention=True)              # skip von enc1
        
        # Multi-Stage Refinement für höchste Qualität
        self.refinement = nn.Sequential(
            ResidualBlock(base_channels, use_attention=True),
            ResidualBlock(base_channels, use_attention=True),
            GatedConv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layers
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Store input for global skip connection
        input_img = x[:, :3, :, :]
        known_mask = x[:, 3:4, :, :]
        
        # Initial feature extraction
        x = self.initial(x)
        
        # Encoder with skip connections
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        # Bottleneck with Self-Attention
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Refinement
        x = self.refinement(x)
        
        # Output
        output = self.output(x)
        
        # Kombiniere bekannte Pixel mit rekonstruierten
        final_output = input_img * known_mask + output * (1 - known_mask)
        
        return final_output


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss mit VGG16 Features
    Verbessert die visuelle Qualität erheblich
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False
            
        # Feature-Ebenen als einzelne Sequenzen speichern
        self.features = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2  
            nn.Sequential(*list(vgg.children())[9:16]), # relu3_3
        ])
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, output, target):
        # Normalize
        output = (output - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        # Effiziente sequentielle Feature-Extraktion
        out_feat, tar_feat = output, target
        for layer in self.features:
            out_feat = layer(out_feat)
            tar_feat = layer(tar_feat)
            loss += F.l1_loss(out_feat, tar_feat)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Ultimative Loss-Funktion für beste Ergebnisse:
    - MSE Loss für Pixel-Genauigkeit (wichtig für RMSE-Metrik)
    - L1 Loss für schärfere Details
    - Perceptual Loss für visuelle Qualität
    - SSIM Loss für strukturelle Ähnlichkeit
    """
    def __init__(self, mse_weight: float = 1.0, l1_weight: float = 1.0, 
                 perceptual_weight: float = 0.1, use_perceptual: bool = True):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.use_perceptual = use_perceptual
        
        if use_perceptual:
            self.perceptual = PerceptualLoss()
        
    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        l1_loss = self.l1(output, target)
        
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
        
        if self.use_perceptual:
            perceptual_loss = self.perceptual(output, target)
            total_loss += self.perceptual_weight * perceptual_loss
            
        return total_loss