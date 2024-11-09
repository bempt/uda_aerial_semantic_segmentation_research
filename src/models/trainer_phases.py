"""Multi-phase training implementation for UDA semantic segmentation"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from .uda import UDASegmentationModel, UDALoss, gradient_reverse_layer
import torchmetrics
from .augmentation import get_strong_augmentation
from .train import EarlyStopping

class MultiPhaseTrainer:
    def __init__(
        self,
        source_train_loader,
        source_val_loader,
        target_train_loader,
        target_val_loader,
        device='cuda',
        num_classes=23,
        log_dir='logs'
    ):
        self.source_train_loader = source_train_loader
        self.source_val_loader = source_val_loader
        self.target_train_loader = target_train_loader
        self.target_val_loader = target_val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Initialize model
        self.model = UDASegmentationModel(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_classes
        ).to(device)
        
        # Initialize metrics with binary classification for domain
        self.metrics = {
            'iou': torchmetrics.JaccardIndex(task='multiclass', num_classes=num_classes).to(device),
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device),
            'domain_acc': torchmetrics.Accuracy(task='binary').to(device)
        }
        
        # Setup logging
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, self.timestamp))
        
    def phase1_train(self, epochs=50, learning_rate=1e-4, patience=7):
        """Phase 1: Basic semantic segmentation training"""
        print("Starting Phase 1: Basic Semantic Segmentation")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = UDALoss()
        early_stopping = EarlyStopping(patience=patience)
        
        best_iou = 0.0
        phase1_dir = f'checkpoints/phase1_{self.timestamp}'
        os.makedirs(phase1_dir, exist_ok=True)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_idx, (images, masks) in enumerate(self.source_train_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Update metrics
                self.metrics['iou'].update(outputs.argmax(dim=1), masks)
                self.metrics['accuracy'].update(outputs.argmax(dim=1), masks)
            
            # Validation
            val_loss, val_metrics = self._validate_phase1(self.source_val_loader, criterion)
            
            # Log metrics
            self._log_metrics('phase1', train_loss/len(self.source_train_loader), 
                            val_loss, val_metrics, epoch)
            
            # Save best model
            if val_metrics['iou'] > best_iou:
                best_iou = val_metrics['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iou': best_iou,
                }, os.path.join(phase1_dir, 'best_model.pth'))
            
            # Early stopping
            if early_stopping(val_loss):
                print("Early stopping triggered")
                break
    
    def phase2_train(self, epochs=30, learning_rate=5e-5, patience=5):
        """Phase 2: Supervised Adversarial Training"""
        print("Starting Phase 2: Supervised Adversarial Training")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = UDALoss(lambda_adv=0.001)
        early_stopping = EarlyStopping(patience=patience)
        
        phase2_dir = f'checkpoints/phase2_{self.timestamp}'
        os.makedirs(phase2_dir, exist_ok=True)
        
        best_combined_score = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            # Create iterator for target loader
            target_iter = iter(self.target_train_loader)
            
            for batch_idx, (source_images, source_masks) in enumerate(self.source_train_loader):
                try:
                    target_images, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_train_loader)
                    target_images, _ = next(target_iter)
                
                # Move to device
                source_images = source_images.to(self.device)
                source_masks = source_masks.long().to(self.device)
                target_images = target_images.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with domain adaptation
                source_seg, source_domain = self.model(source_images, domain_adaptation=True)
                _, target_domain = self.model(target_images, domain_adaptation=True)
                
                # Create domain labels (batch_size,)
                batch_size = source_images.size(0)
                source_domain_label = torch.ones(batch_size).to(self.device)
                target_domain_label = torch.zeros(batch_size).to(self.device)
                
                # Calculate losses
                seg_loss = criterion(source_seg, source_masks)
                
                # Ensure domain predictions are properly shaped
                source_domain = source_domain.view(batch_size)
                target_domain = target_domain.view(batch_size)
                
                domain_loss = (criterion.domain_loss(source_domain, source_domain_label) + 
                             criterion.domain_loss(target_domain, target_domain_label)) / 2
                
                total_loss = seg_loss + criterion.lambda_adv * domain_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                batch_count += 1
                
                # Update metrics
                with torch.no_grad():
                    self.metrics['iou'].update(source_seg.argmax(dim=1), source_masks)
                    self.metrics['accuracy'].update(source_seg.argmax(dim=1), source_masks)
                    
                    # Update domain accuracy with binary predictions
                    source_domain_pred = (source_domain > 0).float()
                    target_domain_pred = (target_domain > 0).float()
                    
                    self.metrics['domain_acc'].update(source_domain_pred, source_domain_label)
                    self.metrics['domain_acc'].update(target_domain_pred, target_domain_label)
            
            # Compute epoch metrics
            train_metrics = {
                'iou': self.metrics['iou'].compute(),
                'accuracy': self.metrics['accuracy'].compute(),
                'domain_acc': self.metrics['domain_acc'].compute()
            }
            
            # Reset metrics
            for metric in self.metrics.values():
                metric.reset()
            
            # Validation
            val_loss, val_metrics = self._validate_phase2()
            
            # Log metrics
            self._log_metrics('phase2', train_loss/batch_count, val_loss, val_metrics, epoch)
            
            # Save best model based on combined score
            combined_score = val_metrics['iou'] * val_metrics['domain_acc']
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'score': best_combined_score,
                }, os.path.join(phase2_dir, 'best_model.pth'))
            
            if early_stopping(val_loss):
                print("Early stopping triggered")
                break
    
    def phase3_train(self, epochs=20, learning_rate=1e-5, patience=5):
        """Phase 3: Unsupervised Fine-tuning"""
        print("Starting Phase 3: Unsupervised Fine-tuning")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = UDALoss(lambda_adv=0.0001)  # Reduced adversarial weight
        early_stopping = EarlyStopping(patience=patience)
        
        # Get strong augmentation for consistency regularization
        strong_aug = get_strong_augmentation()
        
        phase3_dir = f'checkpoints/phase3_{self.timestamp}'
        os.makedirs(phase3_dir, exist_ok=True)
        
        best_target_iou = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_idx, (target_images, _) in enumerate(self.target_train_loader):
                target_images = target_images.to(self.device)
                
                # Generate two different augmentations
                aug1 = strong_aug(image=target_images.cpu().numpy())['image'].to(self.device)
                aug2 = strong_aug(image=target_images.cpu().numpy())['image'].to(self.device)
                
                optimizer.zero_grad()
                
                # Get predictions for both augmentations
                pred1 = self.model(aug1)
                pred2 = self.model(aug2)
                
                # Consistency loss
                consistency_loss = torch.mean((pred1 - pred2) ** 2)
                
                # Domain confusion loss (optional)
                _, domain_pred = self.model(target_images, domain_adaptation=True)
                domain_confusion_loss = -torch.mean(torch.abs(domain_pred - 0.5))
                
                # Total loss
                total_loss = consistency_loss + 0.1 * domain_confusion_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # Validation on target dataset
            val_loss, val_metrics = self._validate_phase3()
            
            # Log metrics
            self._log_metrics('phase3', train_loss/len(self.target_train_loader), 
                            val_loss, val_metrics, epoch)
            
            # Save best model
            if val_metrics['iou'] > best_target_iou:
                best_target_iou = val_metrics['iou']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'target_iou': best_target_iou,
                }, os.path.join(phase3_dir, 'best_model.pth'))
            
            if early_stopping(val_loss):
                print("Early stopping triggered")
                break
    
    def _validate_phase1(self, dataloader, criterion):
        """Validation for Phase 1"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                self.metrics['iou'].update(outputs.argmax(dim=1), masks)
                self.metrics['accuracy'].update(outputs.argmax(dim=1), masks)
        
        metrics = {
            'iou': self.metrics['iou'].compute(),
            'accuracy': self.metrics['accuracy'].compute()
        }
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        return val_loss/len(dataloader), metrics
    
    def _validate_phase2(self):
        """Validation for Phase 2"""
        self.model.eval()
        val_loss = 0
        batch_count = 0
        criterion = UDALoss(lambda_adv=0.001)
        
        with torch.no_grad():
            target_iter = iter(self.target_val_loader)
            
            for source_images, source_masks in self.source_val_loader:
                try:
                    target_images, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_val_loader)
                    target_images, _ = next(target_iter)
                
                # Move to device
                source_images = source_images.to(self.device)
                source_masks = source_masks.long().to(self.device)
                target_images = target_images.to(self.device)
                
                # Forward pass
                source_seg, source_domain = self.model(source_images, domain_adaptation=True)
                _, target_domain = self.model(target_images, domain_adaptation=True)
                
                # Create domain labels
                batch_size = source_images.size(0)
                source_domain_label = torch.ones(batch_size).to(self.device)
                target_domain_label = torch.zeros(batch_size).to(self.device)
                
                # Ensure domain predictions are properly shaped
                source_domain = source_domain.view(batch_size)
                target_domain = target_domain.view(batch_size)
                
                # Calculate losses
                seg_loss = criterion(source_seg, source_masks)
                domain_loss = (criterion.domain_loss(source_domain, source_domain_label) + 
                             criterion.domain_loss(target_domain, target_domain_label)) / 2
                
                val_loss += seg_loss.item() + criterion.lambda_adv * domain_loss.item()
                batch_count += 1
                
                # Update metrics
                self.metrics['iou'].update(source_seg.argmax(dim=1), source_masks)
                self.metrics['accuracy'].update(source_seg.argmax(dim=1), source_masks)
                
                source_domain_pred = (source_domain > 0).float()
                target_domain_pred = (target_domain > 0).float()
                
                self.metrics['domain_acc'].update(source_domain_pred, source_domain_label)
                self.metrics['domain_acc'].update(target_domain_pred, target_domain_label)
        
        # Compute metrics
        metrics = {
            'iou': self.metrics['iou'].compute(),
            'accuracy': self.metrics['accuracy'].compute(),
            'domain_acc': self.metrics['domain_acc'].compute()
        }
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        return val_loss/batch_count, metrics
    
    def _validate_phase3(self):
        """Validation for Phase 3"""
        # Similar to _validate_phase1 but on target dataset
        pass
    
    def _log_metrics(self, phase, train_loss, val_loss, metrics, epoch):
        """Log metrics to TensorBoard"""
        self.writer.add_scalar(f'{phase}/train_loss', train_loss, epoch)
        self.writer.add_scalar(f'{phase}/val_loss', val_loss, epoch)
        
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{metric_name}', value, epoch)
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print('Metrics:', {k: f'{v:.4f}' for k, v in metrics.items()}) 