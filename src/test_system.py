import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from src.data.dataset import DroneDataset
from src.data.target_dataset import TargetDataset
from src.models.train import SegmentationTrainer
from src.models.predict import predict_mask
from src.models.augmentation import get_training_augmentation
from src.models.config import Config
from src.models.discriminator import DomainDiscriminator
from src.models.losses import AdversarialLoss
from src.data.prepare_holyrood import prepare_holyrood_dataset

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

    # 6. Test domain discriminator
    print("\n6. Testing domain discriminator...")
    try:
        discriminator = DomainDiscriminator(input_channels=3)
        
        # Test with random input
        batch_size = 4
        test_input = torch.randn(batch_size, 3, 256, 256)  # Assuming 256x256 images
        
        # Forward pass
        domain_predictions = discriminator(test_input)
        
        # Check output shape and values
        assert domain_predictions.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {domain_predictions.shape}"
        assert torch.all((domain_predictions >= 0) & (domain_predictions <= 1)), "Predictions should be between 0 and 1"
        
        print("✓ Domain discriminator tested successfully")
        print(f"Sample predictions shape: {domain_predictions.shape}")
        print(f"Sample prediction values: {domain_predictions.squeeze().detach().numpy()}")
        
    except Exception as e:
        print(f"✗ Domain discriminator test failed: {str(e)}")
        return False

    # 7. Testing adversarial losses
    print("\n7. Testing adversarial losses...")
    try:
        adv_loss = AdversarialLoss(lambda_adv=0.001)
        
        # Create dummy predictions
        batch_size = 4
        source_pred = torch.rand(batch_size, 1)  # Random predictions between 0 and 1
        target_pred = torch.rand(batch_size, 1)
        
        # Test discriminator loss
        d_loss = adv_loss.discriminator_loss(source_pred, target_pred)
        assert isinstance(d_loss, torch.Tensor), "Discriminator loss should be a tensor"
        assert d_loss.shape == torch.Size([]), "Discriminator loss should be a scalar"
        
        # Test generator loss
        g_loss = adv_loss.generator_loss(target_pred)
        assert isinstance(g_loss, torch.Tensor), "Generator loss should be a tensor"
        assert g_loss.shape == torch.Size([]), "Generator loss should be a scalar"
        
        print("✓ Adversarial losses tested successfully")
        print(f"Sample discriminator loss: {d_loss.item():.4f}")
        print(f"Sample generator loss: {g_loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Adversarial losses test failed: {str(e)}")
        return False

    # 8. Test target domain dataset
    print("\n8. Testing target domain dataset...")
    try:
        # Use the same sample directory for testing
        # In practice, this would be a different directory with target domain images
        target_images_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'original_images')
        
        target_dataset = TargetDataset(
            images_dir=target_images_dir,
            transform=get_training_augmentation()
        )
        
        # Test dataset size
        assert len(target_dataset) > 0, "Target dataset is empty"
        
        # Test loading an image
        sample_image = target_dataset[0]
        assert isinstance(sample_image, torch.Tensor), "Dataset should return a tensor"
        assert sample_image.dim() == 3, "Image should have 3 dimensions (C, H, W)"
        assert sample_image.shape[0] == 3, "Image should have 3 channels"
        
        # Test dataloader
        target_loader = DataLoader(
            target_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
        )
        
        # Test batch loading
        sample_batch = next(iter(target_loader))
        assert sample_batch.dim() == 4, "Batch should have 4 dimensions (B, C, H, W)"
        
        print("✓ Target domain dataset tested successfully")
        print(f"Dataset size: {len(target_dataset)}")
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample batch shape: {sample_batch.shape}")
        
    except Exception as e:
        print(f"✗ Target domain dataset test failed: {str(e)}")
        return False

    # 9. Test Holyrood dataset preparation
    print("\n9. Testing Holyrood dataset preparation...")
    try:
        # Use sample Holyrood dataset
        holyrood_sample_dir = os.path.join('data', 'sample', 'holyrood')  # Changed path
        
        # Test loading Holyrood dataset
        holyrood_dataset = TargetDataset(
            images_dir=str(holyrood_sample_dir),
            transform=get_training_augmentation()
        )
        
        # Test dataloader
        holyrood_loader = DataLoader(
            holyrood_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
        )
        
        # Test batch loading
        sample_batch = next(iter(holyrood_loader))
        assert sample_batch.dim() == 4, "Batch should have 4 dimensions (B, C, H, W)"
        
        print("✓ Holyrood sample dataset tested successfully")
        print(f"Total sample images: {len(holyrood_dataset)}")
        print(f"Sample batch shape: {sample_batch.shape}")
        
    except Exception as e:
        print(f"✗ Holyrood sample dataset test failed: {str(e)}")
        return False

    print("\nAll system tests completed successfully! ✓")
    return True

if __name__ == "__main__":
    success = test_system()
    if success:
        print("\nSystem is ready for training!")
    else:
        print("\nSystem test failed. Please check the errors above.")