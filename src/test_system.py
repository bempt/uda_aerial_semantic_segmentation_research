import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split

from src.data.dataset import DroneDataset
from src.models.train import SegmentationTrainer
from src.models.predict import predict_mask
from src.models.augmentation import get_training_augmentation
from src.models.config import Config

def test_system():
    print("Starting system test...")
    
    # Setup directories
    Config.setup_directories()
    
    # 1. Test data loading
    print("\n1. Testing data loading...")
    try:
        images_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'original_images')
        masks_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic')
        
        dataset = DroneDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            transform=get_training_augmentation()
        )
        print(f"✓ Dataset loaded successfully with {len(dataset)} images")
        
        # Split dataset using config
        train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
        )
        print("✓ DataLoaders created successfully")
        
    except Exception as e:
        print(f"✗ Data loading failed: {str(e)}")
        return False

    # 2. Test model creation
    print("\n2. Testing model creation...")
    try:
        model = smp.Unet(
            encoder_name=Config.ENCODER_NAME,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.IN_CHANNELS,
            classes=Config.NUM_CLASSES,
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
            device=Config.DEVICE
        )
        
        # Run a mini training session (2 epochs)
        trainer.train(
            train_dataloader=train_loader,
            valid_dataloader=val_loader,
            epochs=2,  # Override config epochs for testing
            learning_rate=Config.LEARNING_RATE,
            patience=Config.PATIENCE
        )
        print("✓ Training loop completed successfully")
        
    except Exception as e:
        print(f"✗ Training loop failed: {str(e)}")
        return False

    # 4. Test model saving and loading
    print("\n4. Testing model saving and loading...")
    try:
        test_checkpoint_dir = os.path.join(Config.CHECKPOINTS_DIR, 'test_checkpoint')
        os.makedirs(test_checkpoint_dir, exist_ok=True)
        test_checkpoint_path = os.path.join(test_checkpoint_dir, 'test_model.pth')
        
        torch.save(model.state_dict(), test_checkpoint_path)
        model.load_state_dict(torch.load(test_checkpoint_path))
        print("✓ Model checkpoint saved and loaded successfully")
        
    except Exception as e:
        print(f"✗ Model saving/loading failed: {str(e)}")
        return False

    # 5. Test prediction
    print("\n5. Testing prediction...")
    try:
        sample_image, _ = val_dataset[0]
        sample_image = sample_image.unsqueeze(0)
        
        prediction = predict_mask(
            model=model,
            img=sample_image,
            device=Config.DEVICE
        )
        print("✓ Prediction completed successfully")
        print(f"Prediction shape: {prediction.shape}")
        
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        return False

    print("\nAll system tests completed successfully! ✓")
    return True

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\nSystem is ready for training!")
    else:
        print("\nSystem test failed. Please check the errors above.")