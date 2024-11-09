import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        """
        Domain discriminator that classifies whether an image is from source or target domain.
        
        Args:
            input_channels (int): Number of input channels (default: 3 for RGB images)
        """
        super(DomainDiscriminator, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Domain classification probability (0 = source, 1 = target)
        """
        features = self.features(x)
        domain_pred = self.classifier(features)
        return domain_pred 