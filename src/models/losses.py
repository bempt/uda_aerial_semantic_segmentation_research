import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialLoss:
    def __init__(self, lambda_adv=0.001):
        """
        Adversarial loss for domain adaptation.
        
        Args:
            lambda_adv (float): Weight for adversarial loss (default: 0.001)
        """
        self.lambda_adv = lambda_adv
        self.bce_loss = nn.BCELoss()
        
    def discriminator_loss(self, source_pred, target_pred):
        """
        Calculate discriminator loss.
        
        Args:
            source_pred (torch.Tensor): Discriminator predictions for source domain
            target_pred (torch.Tensor): Discriminator predictions for target domain
            
        Returns:
            torch.Tensor: Discriminator loss
        """
        # Source domain should be classified as 0
        source_label = torch.zeros_like(source_pred)
        source_loss = self.bce_loss(source_pred, source_label)
        
        # Target domain should be classified as 1
        target_label = torch.ones_like(target_pred)
        target_loss = self.bce_loss(target_pred, target_label)
        
        # Total discriminator loss
        d_loss = (source_loss + target_loss) / 2
        return d_loss
        
    def generator_loss(self, target_pred):
        """
        Calculate generator (segmentation model) adversarial loss.
        The generator tries to fool the discriminator by making target predictions close to 0.
        
        Args:
            target_pred (torch.Tensor): Discriminator predictions for target domain
            
        Returns:
            torch.Tensor: Generator adversarial loss
        """
        target_label = torch.zeros_like(target_pred)  # Try to fool discriminator
        g_loss = self.bce_loss(target_pred, target_label)
        return self.lambda_adv * g_loss 