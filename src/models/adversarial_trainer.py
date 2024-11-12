import torch
import torch.nn as nn
from tqdm import tqdm
from .train import SegmentationTrainer
from .discriminator import DomainDiscriminator
from .losses import AdversarialLoss
from .metrics import DomainAdaptationMetrics

class AdversarialTrainer(SegmentationTrainer):
    def __init__(self, model, device, lambda_adv=0.001):
        """
        Trainer for adversarial domain adaptation.
        
        Args:
            model: Segmentation model
            device: Device to use for training
            lambda_adv: Weight for adversarial loss
        """
        super().__init__(model, device)
        self.discriminator = DomainDiscriminator().to(device)
        self.adversarial_loss = AdversarialLoss(lambda_adv)
        self.discriminator_optimizer = None
        self.domain_metrics = DomainAdaptationMetrics()
        
    def calculate_iou(self, pred, target):
        """
        Calculate Intersection over Union (IoU) between prediction and target masks.
        
        Args:
            pred (torch.Tensor): Predicted mask
            target (torch.Tensor): Target mask
            
        Returns:
            float: IoU score
        """
        intersection = torch.logical_and(pred, target)
        union = torch.logical_or(pred, target)
        iou = torch.sum(intersection).float() / (torch.sum(union).float() + 1e-8)
        return iou.item()

    def train_epoch(self, source_dataloader, target_dataloader, optimizer, epoch):
        """
        Train for one epoch with both source and target domain data.
        
        Args:
            source_dataloader: DataLoader for source domain (with labels)
            target_dataloader: DataLoader for target domain (no labels)
            optimizer: Optimizer for segmentation model
            epoch: Current epoch number
        """
        self.model.train()
        self.discriminator.train()
        self.domain_metrics.reset()  # Reset metrics at start of epoch
        
        if self.discriminator_optimizer is None:
            self.discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), 
                lr=optimizer.param_groups[0]['lr']
            )
        
        total_loss = 0
        target_iter = iter(target_dataloader)
        
        # Progress bar
        pbar = tqdm(source_dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, (source_images, source_masks) in enumerate(pbar):
            # Get target images
            try:
                target_images = next(target_iter)
            except StopIteration:
                target_iter = iter(target_dataloader)
                target_images = next(target_iter)
            
            # Move data to device
            source_images = source_images.to(self.device)
            source_masks = source_masks.to(self.device)
            target_images = target_images.to(self.device)
            
            # Ensure source_masks has correct shape (B, H, W)
            if source_masks.dim() == 4 and source_masks.size(1) == 1:
                source_masks = source_masks.squeeze(1)
            
            # Train discriminator
            self.discriminator_optimizer.zero_grad()
            
            source_domain_pred = self.discriminator(source_images)
            target_domain_pred = self.discriminator(target_images)
            
            # Update domain adaptation metrics
            self.domain_metrics.update(source_domain_pred, target_domain_pred)
            
            d_loss = self.adversarial_loss.discriminator_loss(
                source_domain_pred, 
                target_domain_pred
            )
            d_loss.backward()
            self.discriminator_optimizer.step()
            
            # Train segmentation model
            optimizer.zero_grad()
            
            # Segmentation loss on source domain
            source_seg_pred = self.model(source_images)
            seg_loss = self.criterion(source_seg_pred, source_masks)
            
            # Adversarial loss on target domain
            target_domain_pred = self.discriminator(target_images)
            adv_loss = self.adversarial_loss.generator_loss(target_domain_pred)
            
            # Combined loss
            total_g_loss = seg_loss + adv_loss
            total_g_loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += total_g_loss.item()
            
            # Update progress bar with domain metrics
            metrics_dict = self.domain_metrics.get_metrics()
            pbar.set_postfix({
                'seg_loss': f'{seg_loss.item():.4f}',
                'd_loss': f'{d_loss.item():.4f}',
                'adv_loss': f'{adv_loss.item():.4f}',
                'domain_conf': metrics_dict['domain_confusion']
            })
        
        return total_loss / len(source_dataloader), self.domain_metrics.get_metrics()
    
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            tuple: (validation loss, validation metrics)
        """
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Ensure masks has correct shape (B, H, W)
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                pred_masks = outputs.argmax(dim=1)
                iou = self.calculate_iou(pred_masks, masks)
                accuracy = (pred_masks == masks).float().mean()
                
                total_loss += loss.item()
                total_iou += iou
                total_accuracy += accuracy
        
        # Calculate averages
        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        
        metrics = {
            'iou': f'{avg_iou:.4f}',
            'accuracy': f'{avg_accuracy:.4f}'
        }
        
        return avg_loss, metrics
    
    def train(self, source_dataloader, target_dataloader, valid_dataloader, 
              epochs, learning_rate, patience=3):
        """
        Train the model with domain adaptation.
        
        Args:
            source_dataloader: DataLoader for source domain (with labels)
            target_dataloader: DataLoader for target domain (no labels)
            valid_dataloader: DataLoader for validation
            epochs: Number of epochs to train
            learning_rate: Learning rate
            patience: Early stopping patience
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss, domain_metrics = self.train_epoch(
                source_dataloader, 
                target_dataloader, 
                optimizer, 
                epoch
            )
            valid_loss, valid_metrics = self.validate(valid_dataloader)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}')
            print(f'Valid Metrics: {valid_metrics}')
            print(f'Domain Metrics: {domain_metrics}')
            
            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping after {epoch} epochs')
                    break