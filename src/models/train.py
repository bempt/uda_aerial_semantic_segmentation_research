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
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

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
        self.criterion = nn.CrossEntropyLoss()  # Changed from BCEWithLogitsLoss
        
        # Initialize metrics for multi-class segmentation
        self.iou_metric = torchmetrics.JaccardIndex(
            task='multiclass',
            num_classes=Config.NUM_CLASSES
        ).to(device)
        
    def calculate_metrics(self, outputs, masks):
        """
        Calculate training metrics.
        
        Args:
            outputs: Model predictions (B, C, H, W)
            masks: Ground truth masks (B, H, W)
            
        Returns:
            dict: Dictionary of metric values
        """
        # Get predicted class indices
        pred_masks = outputs.argmax(dim=1)
        
        # Calculate IoU
        iou = self.iou_metric(pred_masks, masks)
        
        # Calculate accuracy
        accuracy = (pred_masks == masks).float().mean()
        
        return {
            'iou': f'{iou.item():.4f}',
            'accuracy': f'{accuracy.item():.4f}'
        }
        
    def train_epoch(self, dataloader, optimizer, epoch):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).long()  # Convert masks to long type
            
            optimizer.zero_grad()
            outputs = self.model(images)
            
            # CrossEntropyLoss expects (B, C, H, W) for outputs and (B, H, W) for targets
            loss = self.criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate and display metrics
            with torch.no_grad():
                metrics = self.calculate_metrics(outputs.detach(), masks)
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': metrics['iou'],
                    'acc': metrics['accuracy']
                })
            
        return total_loss / len(dataloader)
        
    def validate(self, dataloader):
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            tuple: (validation loss, validation metrics)
        """
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device).long()  # Convert masks to long type
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                all_outputs.append(outputs)
                all_masks.append(masks)
            
            # Concatenate all batches
            all_outputs = torch.cat(all_outputs)
            all_masks = torch.cat(all_masks)
            
            # Calculate metrics
            metrics = self.calculate_metrics(all_outputs, all_masks)
            
        return total_loss / len(dataloader), metrics
        
    def train(self, train_dataloader, valid_dataloader, epochs, learning_rate, patience=7):
        """
        Train the model.
        
        Args:
            train_dataloader: Training data loader
            valid_dataloader: Validation data loader
            epochs: Number of epochs to train
            learning_rate: Learning rate
            patience: Early stopping patience
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_dataloader, optimizer, epoch)
            valid_loss, valid_metrics = self.validate(valid_dataloader)
            
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Valid Loss: {valid_loss:.4f}')
            print(f'Valid Metrics: {valid_metrics}')
            
            # Early stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'EarlyStopping counter: {patience_counter} out of {patience}')
                if patience_counter >= patience:
                    print(f'Early stopping after {epoch} epochs')
                    break

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