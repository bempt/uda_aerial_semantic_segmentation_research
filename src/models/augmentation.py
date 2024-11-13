"""Data augmentation functions"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

def get_training_augmentation():
    """Get basic augmentation pipeline for training."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_strong_augmentation():
    """Get strong augmentation pipeline for unsupervised learning."""
    return A.Compose([
        A.RandomRotate90(p=0.7),
        A.Flip(p=0.7),
        A.Transpose(p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(30.0, 80.0)),
            A.GaussNoise(var_limit=(20.0, 60.0)),
        ], p=0.4),
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.4),
            A.MedianBlur(blur_limit=5, p=0.3),
            A.Blur(blur_limit=5, p=0.3),
        ], p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.3,
            rotate_limit=60,
            p=0.5
        ),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3, p=0.4),
            A.GridDistortion(distort_limit=0.3, p=0.4),
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=0.4
            ),
        ], p=0.4),
        A.OneOf([
            A.CLAHE(clip_limit=4, p=0.4),
            A.Sharpen(p=0.4),
            A.Emboss(p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.4
            ),
        ], p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.4
        ),
        A.Normalize(),
        ToTensorV2(always_apply=True),
    ])

def get_validation_augmentation():
    """Get validation augmentation pipeline."""
    return A.Compose([
        A.Normalize(),
        ToTensorV2(always_apply=True),
    ])

def apply_augmentation(image, augmentation):
    """Apply augmentation and ensure numpy array output."""
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    augmented = augmentation(image=image)
    return augmented['image']