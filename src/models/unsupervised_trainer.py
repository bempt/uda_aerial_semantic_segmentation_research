import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Optional

from .train import SegmentationTrainer
from .losses import FineTuningLoss
from .metrics import DomainAdaptationMetrics
from .augmentation import get_strong_augmentation
from .domain_model import DomainAdaptationModel
from .discriminator import DomainDiscriminator

class UnsupervisedTrainer(SegmentationTrainer):
    """Trainer for Phase 3 unsupervised fine-tuning"""
    
    def __init__(
        self,
        model,
        device,
        consistency_weight: float = 1.0,
        domain_weight: float = 0.1,
        supervised_weight: float = 0.1,
        rampup_length: int = 40,
        log_interval: int = 10,
        patience: int = 7
    ):
        # Create discriminator
        discriminator = DomainDiscriminator().to(device)
        
        # Wrap model with domain adaptation support
        domain_model = DomainAdaptationModel(model, discriminator).to(device)
        
        super().__init__(domain_model, device)
        
        self.fine_tuning_loss = FineTuningLoss(
            consistency_weight=consistency_weight,
            domain_weight=domain_weight,
            supervised_weight=supervised_weight,
            rampup_length=rampup_length
        )
        
        self.domain_metrics = DomainAdaptationMetrics()
        self.strong_augmentation = get_strong_augmentation()
        self.log_interval = log_interval
        self.patience = patience
        
        # Initialize early stopping variables
        self.best_score = float('-inf')
        self.best_epoch = 0
        self.counter = 0
        
    def train_epoch(
        self,
        target_dataloader,
        optimizer,
        epoch: int,
        supervised_dataloader=None
    ):
        """
        Train for one epoch with unsupervised fine-tuning.
        
        Args:
            target_dataloader: DataLoader for target domain (unlabeled)
            optimizer: Optimizer for model parameters
            epoch: Current epoch number
            supervised_dataloader: Optional dataloader for labeled data
        """
        self.model.train()
        self.domain_metrics.reset()
        total_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(target_dataloader, desc=f'Epoch {epoch}')
        
        # Get supervised data iterator if available
        supervised_iter = iter(supervised_dataloader) if supervised_dataloader else None
        
        for batch_idx, target_images in enumerate(pbar):
            try:
                # Clear GPU cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
                # Move target images to device
                if isinstance(target_images, (list, tuple)):
                    target_images = target_images[0]  # Handle case where dataloader returns tuple/list
                target_images = target_images.to(self.device)
                
                # Get supervised batch if available
                supervised_batch = None
                if supervised_iter is not None:
                    try:
                        supervised_batch = next(supervised_iter)
                    except StopIteration:
                        supervised_iter = iter(supervised_dataloader)
                        supervised_batch = next(supervised_iter)
                        
                # Convert tensor to numpy for augmentation
                target_np = target_images.cpu().numpy().transpose(0, 2, 3, 1)  # B,C,H,W -> B,H,W,C
                
                # Apply strong augmentations to target images
                aug1_list = []
                aug2_list = []
                for img in target_np:
                    # Apply augmentation to each image in batch
                    aug1 = self.strong_augmentation(image=img)['image']
                    aug2 = self.strong_augmentation(image=img)['image']
                    aug1_list.append(aug1)
                    aug2_list.append(aug2)
                    
                # Stack augmented images and move to device
                aug1 = torch.stack(aug1_list).to(self.device)
                aug2 = torch.stack(aug2_list).to(self.device)
                
                # Get predictions for augmented images
                with torch.amp.autocast('cuda'):  # Updated autocast usage
                    pred1 = self.model(aug1)
                    pred2 = self.model(aug2)
                    
                    # Get domain predictions
                    _, domain_pred = self.model(target_images, domain_adaptation=True)
                    
                    # Calculate fine-tuning loss
                    loss_dict = self.fine_tuning_loss(
                        pred1=pred1,
                        pred2=pred2,
                        domain_pred=domain_pred,
                        epoch=epoch,
                        supervised_pred=supervised_batch[0] if supervised_batch else None,
                        supervised_target=supervised_batch[1] if supervised_batch else None
                    )
                
                # Check for invalid loss values
                if not torch.isfinite(loss_dict['total']):
                    print(f"Warning: Invalid loss value encountered: {loss_dict}")
                    continue
                
                # Optimization step
                optimizer.zero_grad()
                loss_dict['total'].backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Update metrics with sigmoid for probabilities
                self.domain_metrics.update(
                    source_pred=torch.sigmoid(domain_pred),
                    target_pred=torch.sigmoid(domain_pred)
                )
                
                # Update running loss (safely)
                if torch.isfinite(loss_dict['total']):
                    total_loss += loss_dict['total'].item()
                    num_batches += 1
                
                # Update progress bar
                metrics = self.domain_metrics.get_metrics()
                pbar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}" if torch.isfinite(loss_dict['total']) else 'NaN',
                    'cons_loss': f"{loss_dict['consistency'].item():.4f}" if torch.isfinite(loss_dict['consistency']) else 'NaN',
                    'domain_conf': metrics['domain_confusion'],
                    'rampup': f"{loss_dict['rampup_weight'].item():.2f}"
                })
                
                # Log to tensorboard
                if batch_idx % self.log_interval == 0:
                    step = epoch * len(target_dataloader) + batch_idx
                    self._log_training_step(loss_dict, metrics, step)
                    
                # Clear some memory
                del pred1, pred2, domain_pred, loss_dict
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
                
        # Safely calculate average loss
        avg_loss = total_loss / max(num_batches, 1)  # Prevent division by zero
        return avg_loss, self.domain_metrics.get_metrics()
        
    def _log_training_step(self, loss_dict: Dict[str, torch.Tensor], metrics: Dict[str, float], step: int):
        """Log training metrics to tensorboard"""
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.logger.log_scalar(f'train/loss_{loss_name}', loss_value.item(), step)
            
        # Log domain metrics
        for metric_name, metric_value in metrics.items():
            self.logger.log_scalar(f'train/{metric_name}', float(metric_value), step)
            
    def _log_validation_step(self, metrics: Dict[str, float], step: int):
        """
        Log validation metrics to tensorboard.
        
        Args:
            metrics: Dictionary of validation metrics
            step: Current global step
        """
        # Log base segmentation metrics
        for metric_name, metric_value in metrics.items():
            # Convert string metrics to float if needed
            if isinstance(metric_value, str):
                try:
                    metric_value = float(metric_value)
                except ValueError:
                    continue
            self.logger.log_scalar(f'val/{metric_name}', metric_value, step)
        
        # Log domain adaptation metrics
        domain_metrics = self.domain_metrics.get_metrics()
        for metric_name, metric_value in domain_metrics.items():
            if isinstance(metric_value, str):
                try:
                    metric_value = float(metric_value)
                except ValueError:
                    continue
            self.logger.log_scalar(f'val/domain_{metric_name}', metric_value, step)
        
        # Log validation images periodically
        if step % (self.log_interval * 10) == 0:  # Less frequent than training logs
            try:
                # Get a validation batch
                val_batch = next(iter(self.valid_dataloader))
                images, masks = val_batch
                images = images.to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    predictions = self.model(images)
                
                # Log sample predictions
                for i in range(min(2, len(images))):  # Log up to 2 samples
                    self.logger.log_image(
                        f'val/sample_{i}/image',
                        images[i],
                        step
                    )
                    self.logger.log_image(
                        f'val/sample_{i}/mask',
                        masks[i],
                        step
                    )
                    self.logger.log_image(
                        f'val/sample_{i}/prediction',
                        torch.argmax(predictions[i], dim=0),
                        step
                    )
                    
                # Log domain confusion visualization if available
                if hasattr(self.model, 'discriminator') and self.model.discriminator is not None:
                    with torch.no_grad():
                        _, domain_pred = self.model(images, domain_adaptation=True)
                        domain_scores = torch.sigmoid(domain_pred)
                        
                        # Create domain confusion heatmap
                        for i in range(min(2, len(images))):
                            self.logger.log_scalar(
                                f'val/sample_{i}/domain_score',
                                domain_scores[i].mean().item(),
                                step
                            )
                            
            except Exception as e:
                print(f"Warning: Failed to log validation images: {str(e)}")
        
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: DataLoader for validation data
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        total_iou = 0
        
        # Store dataloader for visualization
        self.valid_dataloader = dataloader
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                
                # Calculate validation metrics
                metrics = self.calculate_metrics(outputs, masks)
                
                # Accumulate IoU
                total_iou += float(metrics.get('iou', 0))
                
                # Log validation metrics
                if num_batches % self.log_interval == 0:
                    self._log_validation_step(metrics, self.current_epoch * len(dataloader) + num_batches)
                
                num_batches += 1
        
        # Ensure IoU is included in metrics
        metrics['iou'] = total_iou / max(num_batches, 1)
        
        return metrics
        
    def train(
        self,
        target_dataloader,
        valid_dataloader,
        epochs: int,
        learning_rate: float,
        supervised_dataloader=None,
        patience: int = 7
    ):
        """
        Train the model with unsupervised fine-tuning.
        
        Args:
            target_dataloader: DataLoader for target domain (unlabeled)
            valid_dataloader: DataLoader for validation
            epochs: Number of epochs to train
            learning_rate: Learning rate
            supervised_dataloader: Optional dataloader for labeled data
            patience: Early stopping patience
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_metrics = self.train_epoch(
                target_dataloader=target_dataloader,
                optimizer=optimizer,
                epoch=epoch,
                supervised_dataloader=supervised_dataloader
            )
            
            # Validation
            valid_metrics = self.validate(valid_dataloader)
            
            # Log epoch metrics
            print(f'\nEpoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Train Metrics: {train_metrics}')
            print(f'Valid Metrics: {valid_metrics}')
            
            # Early stopping check
            if self.early_stopping(epoch, valid_metrics):
                print("Early stopping triggered")
                break 
        
    def early_stopping(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of validation metrics
            
        Returns:
            bool: True if training should stop
        """
        # Get validation score (use IoU as primary metric)
        current_score = float(metrics.get('iou', 0))
        
        # Check if score improved
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        # Log early stopping metrics
        if hasattr(self, 'logger'):
            self.logger.log_scalar('early_stopping/score', current_score, epoch)
            self.logger.log_scalar('early_stopping/counter', self.counter, epoch)
        
        # Stop if no improvement for patience epochs
        if self.counter >= self.patience:
            print(f"\nEarly stopping triggered. Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
            return True
        
        return False