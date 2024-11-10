import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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
        
        # Apply transforms if specified
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default transform if none provided
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((320, 320))
            ])
            image = transform(image)
            
        return image