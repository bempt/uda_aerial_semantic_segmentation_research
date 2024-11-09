"""Unsupervised Domain Adaptation (UDA) components for semantic segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DomainDiscriminator(nn.Module):
    """Domain discriminator network"""
    def __init__(self, num_channels=512):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(num_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        return self.discriminator(x)

class UDASegmentationModel(nn.Module):
    """UDA-enabled semantic segmentation model"""
    def __init__(
        self,
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=23,
        activation=None
    ):
        super().__init__()
        
        # Create base segmentation model
        self.segmentation_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=activation
        )
        
        # Create domain discriminator
        # Get the number of channels from encoder's last layer
        if hasattr(self.segmentation_model.encoder, 'out_channels'):
            encoder_channels = self.segmentation_model.encoder.out_channels[-1]
        else:
            encoder_channels = 512  # Default for ResNet50
            
        self.domain_discriminator = DomainDiscriminator(num_channels=encoder_channels)
        
        # Store encoder name for feature extraction
        self.encoder_name = encoder_name
        
    def forward(self, x, domain_adaptation=False):
        # Get encoder features
        features = self.segmentation_model.encoder(x)
        
        if domain_adaptation:
            # For domain adaptation, return both segmentation and domain predictions
            segmentation_output = self.segmentation_model.decoder(*features)
            domain_output = self.domain_discriminator(features[-1])
            domain_output = domain_output.squeeze(-1).squeeze(-1)
            return segmentation_output, domain_output
        else:
            # For normal inference, return only segmentation
            return self.segmentation_model(x)
    
    def get_encoder_features(self, x):
        """Extract features from encoder for domain adaptation"""
        return self.segmentation_model.encoder(x)[-1]

class UDALoss(nn.Module):
    """Combined loss for UDA training"""
    def __init__(self, lambda_adv=0.001):
        super().__init__()
        self.segmentation_loss = smp.losses.DiceLoss(mode='multiclass')
        self.domain_loss = nn.BCEWithLogitsLoss()
        self.lambda_adv = lambda_adv
        
    def forward(self, pred, target, domain_pred=None, domain_target=None):
        # Calculate segmentation loss
        seg_loss = self.segmentation_loss(pred, target)
        
        # If doing domain adaptation, add domain loss
        if domain_pred is not None and domain_target is not None:
            domain_loss = self.domain_loss(domain_pred, domain_target)
            return seg_loss + self.lambda_adv * domain_loss
        
        return seg_loss

def gradient_reverse_layer(x, alpha):
    """Gradient Reversal Layer for adversarial training"""
    return GradientReverseFunction.apply(x, alpha)

class GradientReverseFunction(torch.autograd.Function):
    """Gradient Reversal Layer function"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None 