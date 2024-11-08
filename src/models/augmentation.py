"""Data augmentation functions"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.config import Config

def get_training_augmentation():
    """Get training augmentation pipeline"""
    train_transform = [
        A.Resize(height=320, width=320),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    return A.Compose([
        A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ]) 