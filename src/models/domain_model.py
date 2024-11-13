import torch
import torch.nn as nn

class DomainAdaptationModel(nn.Module):
    """Wrapper for segmentation model with domain adaptation support"""
    
    def __init__(self, segmentation_model, discriminator=None):
        """
        Initialize domain adaptation model.
        
        Args:
            segmentation_model: Base segmentation model
            discriminator: Optional domain discriminator
        """
        super().__init__()
        self.segmentation_model = segmentation_model
        self.discriminator = discriminator
        
    def forward(self, x, domain_adaptation=False):
        """
        Forward pass with optional domain adaptation.
        
        Args:
            x: Input tensor
            domain_adaptation: If True, return (segmentation, domain_pred)
                             If False, return segmentation only
                             
        Returns:
            Tuple of (segmentation_output, domain_prediction) if domain_adaptation=True
            segmentation_output only if domain_adaptation=False
        """
        # Get segmentation prediction
        seg_pred = self.segmentation_model(x)
        
        if domain_adaptation and self.discriminator is not None:
            # Get domain prediction
            domain_pred = self.discriminator(x)
            return seg_pred, domain_pred
        
        return seg_pred
    
    def get_features(self, x):
        """
        Get features from encoder for domain adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor from encoder
        """
        if hasattr(self.segmentation_model, 'encoder'):
            return self.segmentation_model.encoder(x)
        else:
            # If no encoder attribute, return None or raise error
            return None
    
    def train(self, mode=True):
        """Set training mode for both models."""
        self.segmentation_model.train(mode)
        if self.discriminator is not None:
            self.discriminator.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode for both models."""
        self.segmentation_model.eval()
        if self.discriminator is not None:
            self.discriminator.eval()
        return self
    
    def to(self, device):
        """Move both models to specified device."""
        self.segmentation_model = self.segmentation_model.to(device)
        if self.discriminator is not None:
            self.discriminator = self.discriminator.to(device)
        return self
    
    def parameters(self):
        """Get parameters from both models."""
        params = list(self.segmentation_model.parameters())
        if self.discriminator is not None:
            params.extend(list(self.discriminator.parameters()))
        return params 