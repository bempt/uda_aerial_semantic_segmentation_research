import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from src.models.config import Config

def load_class_dict():
    """Load class dictionary from CSV file"""
    csv_path = os.path.join(Config.DATA_DIR, 'class_dict_seg.csv')
    try:
        df = pd.read_csv(csv_path)
        print("\nLoaded class mapping:")
        print(df)
        return df
    except Exception as e:
        print(f"Error loading class dictionary: {str(e)}")
        return None

def create_colored_mask(prediction, class_df):
    """Convert prediction to colored mask using class colors"""
    height, width = prediction.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Iterate through each class
    for idx, row in class_df.iterrows():
        # Create mask for current class
        mask = prediction == idx
        # Get RGB values from DataFrame - access by column index to avoid KeyError
        rgb = [int(row[1]), int(row[2]), int(row[3])]  # Using column indices 1,2,3 for r,g,b
        # Apply color to mask
        colored_mask[mask] = rgb
    
    return colored_mask

def create_overlay(image, mask, alpha=0.5):
    """
    Create an overlay of the original image and predicted mask.
    
    Args:
        image: Original image (PIL Image or numpy array)
        mask: Predicted mask (numpy array)
        alpha: Transparency of the overlay (0-1)
        
    Returns:
        PIL.Image: Overlay image
    """
    if isinstance(image, torch.Tensor):
        # Denormalize and convert to PIL Image
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * np.array(Config.NORMALIZE_STD) + 
                np.array(Config.NORMALIZE_MEAN))
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Convert mask to RGB
    mask = (mask * 255).astype(np.uint8)
    mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_colored[mask > 0] = [255, 0, 0]  # Red for positive predictions
    mask_colored = Image.fromarray(mask_colored)
    
    # Create overlay
    overlay = Image.blend(image, mask_colored, alpha)
    return overlay

def predict_mask(model, img, device=None):
    """
    Predict segmentation mask for a given image.
    
    Args:
        model: Trained segmentation model
        img: Input image (PIL Image, numpy array, or tensor)
        device: Device to use for prediction
        
    Returns:
        numpy.ndarray: Predicted segmentation mask
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Convert input to tensor if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    if isinstance(img, Image.Image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.NORMALIZE_MEAN,
                std=Config.NORMALIZE_STD
            ),
            transforms.Resize(Config.IMAGE_SIZE)
        ])
        img = transform(img)
    
    if img.dim() == 3:
        img = img.unsqueeze(0)  # Add batch dimension
    
    img = img.to(device)
    
    with torch.no_grad():
        mask = model(img)
        mask = torch.sigmoid(mask)
        mask = (mask > 0.5).float()
        
    return mask.squeeze().cpu().numpy()

def predict_batch(model, images, device='cuda'):
    """
    Predict segmentation masks for a batch of images
    
    Args:
        model: PyTorch model
        images: Batch of input images [B, C, H, W]
        device: Device to run prediction on
        
    Returns:
        Batch of predicted masks as numpy array
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        pred_masks = outputs.argmax(dim=1)
        return pred_masks.cpu().numpy()

def test_model(
    model_path,
    test_dir,
    output_dir,
    model_name='Unet',
    encoder_name='resnet34',
    device='cuda'
):
    print(f"Loading model from {model_path}")
    print(f"Testing on images from {test_dir}")
    
    # Load class dictionary
    class_df = load_class_dict()
    if class_df is None:
        raise ValueError("Could not load class dictionary")
    
    num_classes = len(class_df)
    print(f"\nNumber of classes: {num_classes}")
    
    # Create timestamp for this prediction run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create organized output directories
    model_output_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    predictions_dir = os.path.join(model_output_dir, "predictions")
    overlays_dir = os.path.join(model_output_dir, "overlays")
    colored_masks_dir = os.path.join(model_output_dir, "colored_masks")
    
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    os.makedirs(colored_masks_dir, exist_ok=True)
    
    print(f"\nSaving outputs to: {model_output_dir}")
    
    # Load model
    model = getattr(smp, model_name)(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )
    
    # Load checkpoint
    try:
        checkpoint = torch.load(
            model_path,
            weights_only=True
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model weights")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    model = model.to(device)
    model.eval()
    
    # Get test images
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    print(f"\nFound {len(test_images)} test images\n")
    
    # Create a new stats file for this prediction run
    stats_file = os.path.join(model_output_dir, "prediction_stats.txt")
    
    with open(stats_file, 'w') as f:
        # Write model info
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {model_path}\n")
        f.write(f"Loss: {checkpoint.get('loss', 'N/A')}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'N/A')}\n\n")
        
        # Write class mapping
        f.write("Class Mapping:\n")
        f.write(str(class_df) + "\n\n")
        
        f.write("Per-Image Class Distribution:\n")
        f.write("-" * 50 + "\n")
    
    # Process each image
    for i, image_name in enumerate(test_images, 1):
        try:
            image_path = os.path.join(test_dir, image_name)
            print(f"Processing image {i}/{len(test_images)}: {image_name}")
            
            # Get predictions and original image
            predicted_mask, original_image = predict_mask(model, image_path, device)
            
            # Create colored mask
            colored_mask = create_colored_mask(predicted_mask, class_df)
            
            # Create overlay
            overlay = create_overlay(original_image, colored_mask)
            
            # Save outputs
            base_name = os.path.splitext(image_name)[0]
            
            # Save raw prediction
            cv2.imwrite(
                os.path.join(predictions_dir, f'{base_name}_pred.png'),
                predicted_mask.astype(np.uint8)
            )
            
            # Save colored mask
            cv2.imwrite(
                os.path.join(colored_masks_dir, f'{base_name}_colored.png'),
                cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
            )
            
            # Save overlay
            cv2.imwrite(
                os.path.join(overlays_dir, f'{base_name}_overlay.png'),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            )
            
            # Calculate and save class distribution
            unique, counts = np.unique(predicted_mask, return_counts=True)
            with open(stats_file, 'a') as f:
                f.write(f"\nImage {i}/{len(test_images)}: {image_name}\n")
                for cls, cnt in zip(unique, counts):
                    class_name = class_df.iloc[cls]['name']
                    percentage = (cnt / predicted_mask.size) * 100
                    f.write(f"  {class_name}: {percentage:.2f}%\n")
                f.write("\n")
            
            print(f"Class distribution saved to {stats_file}")
            
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}\n")
            # Log errors to stats file as well
            with open(stats_file, 'a') as f:
                f.write(f"Error processing {image_name}: {str(e)}\n\n")
            continue

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test model
    test_model(
        model_path='./checkpoints/best_model.pth',
        test_dir='./data/sample/semantic_drone/original_images',
        output_dir='./results',
        model_name='Unet',
        device=device
    ) 