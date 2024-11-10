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
        device = source_pred.device
        source_label = torch.ones_like(source_pred).to(device)
        target_label = torch.zeros_like(target_pred).to(device)
        
        source_loss = self.bce_loss(source_pred, source_label)
        target_loss = self.bce_loss(target_pred, target_label)
        
        return (source_loss + target_loss) / 2
        
    def generator_loss(self, target_pred):
        """
        Calculate generator (segmentation model) adversarial loss.
        The generator tries to fool the discriminator by making target predictions close to 0.
        
        Args:
            target_pred (torch.Tensor): Discriminator predictions for target domain
            
        Returns:
            torch.Tensor: Generator adversarial loss
        """
        device = target_pred.device
        target_label = torch.ones_like(target_pred).to(device)
        return self.lambda_adv * self.bce_loss(target_pred, target_label)

class ConsistencyLoss(nn.Module):
    """
    Consistency loss for unsupervised learning.
    Enforces consistent predictions between different augmented versions of the same image.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred1, pred2):
        """
        Calculate consistency loss between two predictions.
        
        Args:
            pred1: First prediction tensor (B, C, H, W)
            pred2: Second prediction tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Consistency loss value
        """
        # Apply softmax with temperature
        prob1 = F.softmax(pred1 / self.temperature, dim=1)
        prob2 = F.softmax(pred2 / self.temperature, dim=1)
        
        # Calculate KL divergence in both directions
        loss_kl_1 = F.kl_div(
            F.log_softmax(pred1 / self.temperature, dim=1),
            prob2,
            reduction='batchmean'
        )
        loss_kl_2 = F.kl_div(
            F.log_softmax(pred2 / self.temperature, dim=1),
            prob1,
            reduction='batchmean'
        )
        
        # Take the mean of both directions
        return (loss_kl_1 + loss_kl_2) / 2
    
    def get_similarity_matrix(self, pred1, pred2):
        """
        Calculate similarity matrix between predictions for visualization.
        
        Args:
            pred1: First prediction tensor (B, C, H, W)
            pred2: Second prediction tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Similarity matrix (B, H, W)
        """
        prob1 = F.softmax(pred1, dim=1)
        prob2 = F.softmax(pred2, dim=1)
        
        # Calculate cosine similarity across channel dimension
        similarity = F.cosine_similarity(prob1, prob2, dim=1)
        return similarity