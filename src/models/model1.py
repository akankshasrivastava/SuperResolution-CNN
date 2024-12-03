import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)  # Fixed here
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)  # Fixed here

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out + residual)
        return out

class SuperResolutionModel(nn.Module):
    def __init__(self):
        super(SuperResolutionModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # First downsampling block
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        
        # Second downsampling block
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            ResidualBlock(256),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        
        # First upsampling block
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        
        # Second upsampling block
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # L1 regularization factor
        self.l1_factor = 1e-10

    def forward(self, x):
        # Initial convolutions with dropout
        x1 = F.relu(self.dropout(self.conv1(x)))
        x2 = F.relu(self.dropout(self.conv2(x1)))
        
        # Downsampling path
        d1 = self.down1(x2)
        d2 = self.down2(d1)
        
        # Upsampling path with skip connections
        u1 = self.up1(d2)
        u1 = u1 + d1  # Skip connection
        
        u2 = self.up2(u1)
        u2 = u2 + x2  # Skip connection
        
        # Final convolution
        out = F.relu(self.final_conv(u2))
        
        return out

    def l1_loss(self):
        """Calculate L1 regularization loss"""
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_factor * l1_loss
