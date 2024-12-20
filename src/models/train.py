"""Training script for semantic segmentation model"""

import os
import datetime
import subprocess
import webbrowser
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from src.data.dataset import DroneDataset
from src.models.config import Config
from src.models.augmentation import get_training_augmentation
import segmentation_models_pytorch as smp
import torchmetrics
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from src.visualization.tensorboard_logger import TensorboardLogger
import numpy as np
import torchvision.utils as utils
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from src.visualization import utils as viz_utils

def load_class_dict():
    """Load class dictionary from CSV file"""
    csv_path = os.path.join(Config.DATA_DIR, 'class_dict_seg.csv')
    try:
        # Read CSV with correct column names
        df = pd.read_csv(csv_path, skipinitialspace=True)  # Added skipinitialspace to handle spaces after commas
        print("\nLoaded class mapping:")
        print(df)
        return df
    except Exception as e:
        print(f"Error loading class dictionary: {str(e)}")
        return None

def launch_tensorboard(logdir, port=6006):
    """Launch TensorBoard server and open in browser"""
    # Create logs directory if it doesn't exist
    os.makedirs(logdir, exist_ok=True)
    
    # Kill any existing TensorBoard instances
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'tensorboard.exe'], 
                         stderr=subprocess.DEVNULL)
        else:  # Linux/Mac
            subprocess.run(['pkill', '-f', 'tensorboard'], 
                         stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # Start TensorBoard server
    cmd = ['tensorboard', '--logdir', logdir, '--port', str(port)]
    try:
        tensorboard = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open(f'http://localhost:{port}')
        
        return tensorboard
    except Exception as e:
        print(f"Warning: Could not start TensorBoard: {str(e)}")
        print(f"You can manually start TensorBoard with: tensorboard --logdir {logdir}")
        return None

class EarlyStopping:
    """Enhanced early stopping with multiple metrics and mode support"""
    def __init__(
        self, 
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min',
        min_epochs: int = 10,
        metrics_to_track: List[str] = None,
        weights: Dict[str, float] = None,
        verbose: bool = False
    ):
        """
        Initialize early stopping handler.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as an improvement
            mode: 'min' or 'max' - whether to look for metric minimization or maximization
            min_epochs: Minimum number of epochs to train before allowing early stopping
            metrics_to_track: List of metric names to monitor
            weights: Dictionary of metric names and their weights for combined score
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.metrics_to_track = metrics_to_track or ['loss']
        self.weights = weights or {'loss': 1.0}
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metrics = {}
        self.val_loss_min = float('inf')
        
        # Initialize metric history
        self.metric_history = {metric: [] for metric in self.metrics_to_track}
        
    def _calculate_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted combined score from multiple metrics"""
        score = 0.0
        for metric, value in metrics.items():
            if metric in self.weights:
                score += self.weights[metric] * value
        return score
        
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score"""
        if self.mode == 'min':
            return current < best - self.min_delta
        return current > best + self.min_delta
        
    def __call__(self, epoch: int, metrics: Dict[str, float], logger: TensorboardLogger = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of current metrics
            logger: Optional TensorboardLogger instance
            
        Returns:
            bool: True if training should stop
        """
        # Update metric history
        for metric, value in metrics.items():
            if metric in self.metric_history:
                self.metric_history[metric].append(value)
                
        # Calculate combined score
        current_score = self._calculate_score(metrics)
        
        # Log metrics if logger provided
        if logger:
            logger.log_scalar('early_stopping/score', current_score, epoch)
            logger.log_scalar('early_stopping/counter', self.counter, epoch)
        
        # Don't stop before minimum epochs
        if epoch < self.min_epochs:
            return False
            
        if self.best_score is None:
            self.best_score = current_score
            self.best_metrics = metrics.copy()
        elif self._is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_metrics = metrics.copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered after {epoch} epochs')
                return True
                
        return False
        
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best metrics seen so far"""
        return self.best_metrics
        
    def get_improvement_rate(self) -> Dict[str, float]:
        """Calculate improvement rate for each metric"""
        rates = {}
        for metric, history in self.metric_history.items():
            if len(history) > 1:
                total_change = history[-1] - history[0]
                num_epochs = len(history)
                rates[metric] = total_change / num_epochs
        return rates

class SegmentationTrainer:
    def __init__(self, model, device):
        """
        Initialize trainer.
        
        Args:
            model: Segmentation model
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.iou_metric = torchmetrics.JaccardIndex(
            task='multiclass',
            num_classes=Config.NUM_CLASSES
        ).to(device)
        
        # Initialize tensorboard logger
        self.logger = TensorboardLogger(log_dir=Config.LOGS_DIR)
        
        # Add per-class IoU metrics
        self.per_class_iou_metrics = nn.ModuleList([
            torchmetrics.JaccardIndex(
                task='binary',
                num_classes=2
            ).to(device) for _ in range(Config.NUM_CLASSES)
        ])
        
    def calculate_metrics(self, outputs, masks):
        """Calculate training metrics."""
        pred_masks = outputs.argmax(dim=1)
        
        # Calculate overall IoU and accuracy
        metrics = {
            'iou': self.iou_metric(pred_masks, masks).item(),
            'accuracy': (pred_masks == masks).float().mean().item()
        }
        
        # Calculate per-class IoU
        for class_idx in range(Config.NUM_CLASSES):
            class_pred = (pred_masks == class_idx)
            class_true = (masks == class_idx)
            metrics[f'iou_class_{class_idx}'] = self.per_class_iou_metrics[class_idx](
                class_pred, class_true
            ).item()
            
        return metrics
        
    def _log_confusion_matrix(self, outputs, masks, step: int, prefix: str = 'train'):
        """Log confusion matrix visualization."""
        pred_masks = outputs.argmax(dim=1).cpu().numpy()
        true_masks = masks.cpu().numpy()
        
        # Calculate confusion matrix
        cm = skmetrics.confusion_matrix(
            true_masks.flatten(),
            pred_masks.flatten(),
            labels=range(Config.NUM_CLASSES)
        )
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(Config.NUM_CLASSES),
            yticklabels=range(Config.NUM_CLASSES)
        )
        plt.title(f'{prefix.capitalize()} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Log to tensorboard
        self.logger.log_figure(f'{prefix}/confusion_matrix', plt.gcf(), step)
        plt.close()
        
    def _log_roc_curves(self, outputs: torch.Tensor, masks: torch.Tensor, 
                       step: int, prefix: str = 'train'):
        """Log ROC curves for each class."""
        # Convert to probabilities
        probs = torch.softmax(outputs, dim=1)
        
        plt.figure(figsize=(10, 8))
        
        # Calculate ROC curve for each class
        for class_idx in range(Config.NUM_CLASSES):
            class_probs = probs[:, class_idx].cpu().numpy().flatten()
            class_true = (masks == class_idx).cpu().numpy().flatten()
            
            fpr, tpr, _ = skmetrics.roc_curve(class_true, class_probs)
            auc = skmetrics.auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{prefix.capitalize()} ROC Curves')
        plt.legend()
        
        # Log to tensorboard
        self.logger.log_figure(f'{prefix}/roc_curves', plt.gcf(), step)
        plt.close()
        
    def _log_pr_curves(self, outputs: torch.Tensor, masks: torch.Tensor, 
                      step: int, prefix: str = 'train'):
        """Log precision-recall curves for each class."""
        # Convert to probabilities
        probs = torch.softmax(outputs, dim=1)
        
        plt.figure(figsize=(10, 8))
        
        # Calculate PR curve for each class
        for class_idx in range(Config.NUM_CLASSES):
            class_probs = probs[:, class_idx].cpu().numpy().flatten()
            class_true = (masks == class_idx).cpu().numpy().flatten()
            
            precision, recall, _ = skmetrics.precision_recall_curve(class_true, class_probs)
            ap = skmetrics.average_precision_score(class_true, class_probs)
            
            plt.plot(recall, precision, label=f'Class {class_idx} (AP = {ap:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{prefix.capitalize()} Precision-Recall Curves')
        plt.legend()
        
        # Log to tensorboard
        self.logger.log_figure(f'{prefix}/pr_curves', plt.gcf(), step)
        plt.close()

    def train_epoch(self, dataloader, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device).long()
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate and log training metrics
            with torch.no_grad():
                metrics = self.calculate_metrics(outputs.detach(), masks)
                
                # Log to tensorboard - Use global step
                global_step = (epoch - 1) * len(dataloader) + batch_idx
                
                # Log scalar metrics directly
                self.logger.log_scalar('train/loss', loss.item(), global_step)
                self.logger.log_scalar('train/iou', metrics['iou'], global_step)
                self.logger.log_scalar('train/accuracy', metrics['accuracy'], global_step)
                
                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.log_scalar('train/learning_rate', current_lr, global_step)
                
                # Log sample predictions periodically
                if batch_idx % Config.LOG_INTERVAL == 0:
                    self._log_predictions(images[0], masks[0], outputs[0], global_step, prefix='train')
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{metrics["iou"]:.4f}',
                    'acc': f'{metrics["accuracy"]:.4f}'
                })
                
                # Log advanced metrics periodically
                if batch_idx % Config.LOG_INTERVAL == 0:
                    with torch.no_grad():
                        self._log_confusion_matrix(outputs, masks, global_step, prefix='train')
                        self._log_roc_curves(outputs, masks, global_step, prefix='train')
                        self._log_pr_curves(outputs, masks, global_step, prefix='train')
                        
                        # Log per-class IoU
                        for class_idx in range(Config.NUM_CLASSES):
                            self.logger.log_scalar(
                                f'train/iou_class_{class_idx}',
                                metrics[f'iou_class_{class_idx}'],
                                global_step
                            )
            
        return total_loss / len(dataloader)
        
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                metrics = self.calculate_metrics(outputs, masks)
                all_metrics.append(metrics)
                
                # Log sample predictions periodically
                if batch_idx % Config.LOG_INTERVAL == 0:
                    self._log_predictions(images[0], masks[0], outputs[0], self.current_epoch, prefix='val')
                
                # Log advanced metrics periodically
                if batch_idx % Config.LOG_INTERVAL == 0:
                    self._log_confusion_matrix(outputs, masks, self.current_epoch, prefix='val')
                    self._log_roc_curves(outputs, masks, self.current_epoch, prefix='val')
                    self._log_pr_curves(outputs, masks, self.current_epoch, prefix='val')
                    
                    # Log per-class IoU
                    for class_idx in range(Config.NUM_CLASSES):
                        self.logger.log_scalar(
                            f'val/iou_class_{class_idx}',
                            metrics[f'iou_class_{class_idx}'],
                            self.current_epoch
                        )
            
            # Calculate average metrics
            avg_metrics = {
                'loss': total_loss / len(dataloader),
                'iou': np.mean([m['iou'] for m in all_metrics]),
                'accuracy': np.mean([m['accuracy'] for m in all_metrics])
            }
            
            # Log validation metrics directly
            for metric_name, value in avg_metrics.items():
                self.logger.log_scalar(f'val/{metric_name}', value, self.current_epoch)
            
        return avg_metrics
        
    def _log_predictions(self, image, mask, output, step, prefix='train'):
        """Log prediction visualizations to tensorboard."""
        # Convert output to prediction mask
        pred_mask = output.argmax(dim=0).cpu()
        
        # Convert tensors to float32 for visualization
        image = image.float().cpu()
        mask = mask.float().cpu()
        pred_mask = pred_mask.float()
        
        # Create visualization grid
        self.logger.log_image(f'{prefix}/image', image, step)
        self.logger.log_image(f'{prefix}/ground_truth', mask, step)
        self.logger.log_image(f'{prefix}/prediction', pred_mask, step)
        
        # Create overlay visualization using our custom utils
        overlay = viz_utils.create_overlay(image, pred_mask)
        self.logger.log_image(f'{prefix}/overlay', overlay, step)
        
    def train(self, train_dataloader, valid_dataloader, epochs, learning_rate, patience=7):
        """Train the model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize early stopping with multiple metrics
        early_stopping = EarlyStopping(
            patience=patience,
            mode='max',  # We want to maximize validation metrics
            min_epochs=10,
            metrics_to_track=['loss', 'iou', 'accuracy'],
            weights={'loss': -1.0, 'iou': 1.0, 'accuracy': 0.5},  # Negative weight for loss since we want to minimize it
            verbose=True
        )
        
        self.current_epoch = 0
        
        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch(train_dataloader, optimizer, epoch)
            valid_metrics = self.validate(valid_dataloader)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_metrics["loss"]:.4f}')
            print(f'Valid Metrics: {valid_metrics}')
            
            # Check early stopping
            if early_stopping(epoch, valid_metrics, self.logger):
                print(f"Early stopping triggered. Best metrics: {early_stopping.get_best_metrics()}")
                break
                
            # Save best model based on early stopping's best metrics
            if valid_metrics == early_stopping.get_best_metrics():
                model_path = Path(Config.CHECKPOINTS_DIR) / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': valid_metrics,
                    'improvement_rates': early_stopping.get_improvement_rate()
                }, model_path)
                print("Saved new best model!")
                    
        # Close tensorboard logger
        self.logger.close()

def train_model(
    data_dir=Config.DATA_DIR,
    model_name=Config.MODEL_NAME,
    encoder_name=Config.ENCODER_NAME,
    batch_size=Config.BATCH_SIZE,
    num_epochs=Config.NUM_EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    device=None
):
    print("Starting training...")
    print(f"Using device: {device if device else Config.get_device()}")
    print(f"Data directory: {data_dir}")
    
    if device is None:
        device = Config.get_device()
    
    # Load class dictionary
    class_df = load_class_dict()
    if class_df is None:
        raise ValueError("Could not load class dictionary")
    
    num_classes = len(class_df)
    print(f"\nNumber of classes: {num_classes}")
    
    # Create logs directory and setup TensorBoard
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(Config.LOGS_DIR, current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Launch TensorBoard
    tensorboard_process = launch_tensorboard(Config.LOGS_DIR)
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"View TensorBoard at: http://localhost:6006")
    
    try:
        # Setup paths
        train_images_dir = os.path.join(data_dir, 'original_images')
        train_masks_dir = os.path.join(data_dir, 'label_images_semantic')
        
        print(f"Loading dataset from:")
        print(f"  Images: {train_images_dir}")
        print(f"  Masks: {train_masks_dir}")
        
        # Create datasets
        train_dataset = DroneDataset(
            train_images_dir,
            train_masks_dir,
            transform=get_training_augmentation()
        )
        
        print(f"Dataset size: {len(train_dataset)} images")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True
        )
        
        print(f"Creating {model_name} model with {encoder_name} encoder...")
        
        # Create model with correct number of classes
        model = getattr(smp, model_name)(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
        )
        
        # Move model to device
        model = model.to(device)
        print("Model created and moved to device")
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create loss function
        loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        
        # Training loop
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                # Print shapes for debugging
                if batch_idx == 0:
                    print(f"\nInput shapes - Images: {images.shape}, Masks: {masks.shape}")
                
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Print value ranges for debugging
                if batch_idx == 0:
                    print(f"Value ranges - Images: [{images.min():.3f}, {images.max():.3f}]")
                    print(f"Masks: [{masks.min():.3f}, {masks.max():.3f}]")
                    print(f"Outputs: [{outputs.min():.3f}, {outputs.max():.3f}]")
                    print(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}")
                
                # Calculate loss
                loss = loss_fn(outputs, masks.long())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                batch_count += 1
                global_step += 1
                
                # Log to TensorBoard
                if batch_idx % Config.LOG_INTERVAL == 0:
                    writer.add_scalar('Loss/batch', loss.item(), global_step)
                    
                    # Log example predictions
                    if batch_idx == 0:
                        with torch.no_grad():
                            # Get predicted class indices
                            pred_mask = outputs.argmax(dim=1)[0]
                            true_mask = masks[0]
                            
                            # Create colored visualizations
                            pred_colored = torch.zeros(3, *pred_mask.shape, device=device)
                            true_colored = torch.zeros(3, *true_mask.shape, device=device)
                            
                            # Convert RGB values from string to integers if needed
                            for class_idx, row in class_df.iterrows():
                                mask_pred = pred_mask == class_idx
                                mask_true = true_mask == class_idx
                                
                                # Get RGB values from the correct column names
                                rgb = [int(x) for x in [row[1], row[2], row[3]]]  # Using column indices instead of names
                                
                                for c, color in enumerate(rgb):
                                    pred_colored[c][mask_pred] = color / 255.0
                                    true_colored[c][mask_true] = color / 255.0
                            
                            writer.add_images('Images/input', images[0:1], global_step)
                            writer.add_images('Images/true_mask', true_colored.unsqueeze(0), global_step)
                            writer.add_images('Images/pred_mask', pred_colored.unsqueeze(0), global_step)
                            
                            # Log prediction statistics
                            writer.add_histogram('Predictions', outputs[0], global_step)
                    
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')
            
            # Calculate and log epoch metrics
            avg_loss = epoch_loss / batch_count
            writer.add_scalar('Loss/epoch', avg_loss, epoch)
            print(f'Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}')
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
                print(f"Saved new best model with loss: {best_loss:.4f}")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        writer.close()
        if tensorboard_process:
            tensorboard_process.terminate()
            tensorboard_process.wait()
    
    return model

if __name__ == '__main__':
    print("Starting training script...")
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Train model
        model = train_model(device=device)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise