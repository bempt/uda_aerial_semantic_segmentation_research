import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split

# Use relative imports
from .data.dataset import DroneDataset
from .models.train import SegmentationTrainer
from .models.predict import predict_mask
from .models.augmentation import get_training_augmentation
from .models.config import Config

def test_system():
    print("Starting system test...")
    
    # 1. Test data loading
    print("\n1. Testing data loading...")
    try:
        sample_data_dir = os.path.join('data', 'sample', 'semantic_drone')
        images_dir = os.path.join(sample_data_dir, 'original_images')
        masks_dir = os.path.join(sample_data_dir, 'label_images_semantic')
        
        dataset = DroneDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=get_training_augmentation()
        )
        print(f"✓ Dataset loaded successfully with {len(dataset)} images")
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        print("✓ DataLoaders created successfully")
        
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        return False

    # 2. Test model creation
    print("\n2. Testing model creation...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=23,  # Number of classes in the dataset
        )
        print("✓ Model created successfully")
        
    except Exception as e:
        print(f"✗ Model creation failed: {str(e)}")
        return False

    # 3. Test training loop
    print("\n3. Testing training loop...")
    try:
        trainer = SegmentationTrainer(
            model=model,
            device=device
        )
        
        # Run a mini training session (2 epochs)
        trainer.train(
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            epochs=2,
            learning_rate=0.001,
            patience=3
        )
        print("✓ Training loop completed successfully")
        
    except Exception as e:
        print(f"✗ Training loop failed: {str(e)}")
        return False

    # 4. Test model saving and loading
    print("\n4. Testing model saving and loading...")
    try:
        # Get the latest checkpoint directory
        checkpoint_dirs = [d for d in os.listdir('checkpoints') if os.path.isdir(os.path.join('checkpoints', d))]
        latest_dir = sorted(checkpoint_dirs)[-1]
        checkpoint_path = os.path.join('checkpoints', latest_dir, 'best_model.pth')
        
        # Load the checkpoint
        trainer.load_checkpoint(checkpoint_path)
        print("✓ Model checkpoint loaded successfully")
        
    except Exception as e:
        print(f"✗ Model saving/loading failed: {str(e)}")
        return False

    # 5. Test prediction
    print("\n5. Testing prediction...")
    try:
        # Get a sample image from the validation set
        sample_image, _ = val_dataset[0]
        sample_image = sample_image.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        prediction = predict_mask(
            model=model,
            img=sample_image,
            device=device
        )
        print("✓ Prediction completed successfully")
        print(f"Prediction shape: {prediction.shape}")
        
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        return False

    print("\nAll system tests completed successfully! ✓")
    return True

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    success = test_system()
    if success:
        print("\nSystem is ready for training!")
    else:
        print("\nSystem test failed. Please check the errors above.") 