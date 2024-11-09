import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class TargetDataset(Dataset):
    """
    Dataset class for target domain images (no labels required).
    """
    def __init__(self, images_dir, transform=None):
        """
        Args:
            images_dir (str): Path to directory containing target domain images
            transform (callable, optional): Optional transform to be applied on images
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {images_dir}")
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Convert to numpy array
        image = np.array(image)
        
        # Apply transforms if specified
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        # Convert to tensor if not already done by transform
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            image = image / 255.0  # Normalize to [0, 1]
            
        return image 