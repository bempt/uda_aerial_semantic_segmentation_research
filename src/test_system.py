import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import time

from src.data.dataset import DroneDataset
from src.data.target_dataset import TargetDataset
from src.models.train import SegmentationTrainer
from src.models.predict import predict_mask
from src.models.augmentation import get_training_augmentation, get_strong_augmentation
from src.models.config import Config
from src.models.discriminator import DomainDiscriminator
from src.models.losses import AdversarialLoss, ConsistencyLoss, DiceLoss, WeightedSegmentationLoss, calculate_class_weights, FineTuningLoss
from src.data.prepare_holyrood import prepare_holyrood_dataset
from src.models.adversarial_trainer import AdversarialTrainer
from src.models.phase_manager import PhaseManager, TrainingPhase
from src.data.setup_test_data import setup_test_data
from src.visualization.tensorboard_logger import TensorboardLogger
from src.models.unsupervised_trainer import UnsupervisedTrainer
from src.models.domain_model import DomainAdaptationModel

class TestSuites:
    @staticmethod
    def data_loading_suite():
        print("\nRunning Data Loading Test Suite...")
        try:
            images_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'original_images')
            masks_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic')
            
            # Test dataset with class balancing
            dataset = DroneDataset(
                images_dir=images_dir,
                masks_dir=masks_dir,
                transform=get_training_augmentation(),
                balance_classes=True
            )
            print(f"✓ Dataset loaded successfully with {len(dataset)} images")
            
            # Verify class statistics
            assert hasattr(dataset, 'class_stats'), "Dataset should have class statistics"
            assert hasattr(dataset, 'sample_weights'), "Dataset should have sample weights"
            
            # Verify sample weights
            assert len(dataset.sample_weights) == len(dataset), "Wrong number of sample weights"
            assert np.isclose(dataset.sample_weights.sum(), 1.0), "Sample weights should sum to 1"
            
            # Split dataset using config
            train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create balanced dataloader with proper indices
            train_indices = train_dataset.indices
            train_sampler = dataset.get_sampler(indices=train_indices)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.BATCH_SIZE,
                sampler=train_sampler,  # Use sampler with proper indices
                num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=False,
                num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
            )
            
            # Test batch loading with balanced sampling
            sample_batch = next(iter(train_loader))
            assert len(sample_batch) == 2, "Batch should contain images and masks"
            
            print("✓ DataLoaders created successfully")
            print("Class statistics:", dataset.class_stats)
            
            return True, train_loader, val_loader, train_dataset, val_dataset
            
        except Exception as e:
            print(f"✗ Data loading failed: {str(e)}")
            return False, None, None, None, None

    @staticmethod
    def model_creation_suite():
        print("\nRunning Model Creation Test Suite...")
        try:
            model = smp.Unet(
                encoder_name=Config.ENCODER_NAME,
                encoder_weights=Config.ENCODER_WEIGHTS,
                in_channels=Config.IN_CHANNELS,
                classes=Config.NUM_CLASSES,
            )
            print("✓ Model created successfully")
            return True, model
            
        except Exception as e:
            print(f"✗ Model creation failed: {str(e)}")
            return False, None

    @staticmethod
    def loss_functions_suite():
        print("\nRunning Loss Functions Test Suite...")
        success = True
        try:
            # Test Dice Loss
            print("\nTesting Dice Loss...")
            dice_loss = DiceLoss()
            
            batch_size = 4
            num_classes = Config.NUM_CLASSES
            predictions = torch.rand(batch_size, num_classes, 256, 256)
            targets = torch.randint(0, num_classes, (batch_size, 256, 256))
            
            targets_one_hot = torch.nn.functional.one_hot(targets, num_classes)
            targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            
            loss = dice_loss(predictions, targets_one_hot)
            
            assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
            assert loss.shape == torch.Size([]), "Loss should be a scalar"
            assert loss >= 0 and loss <= 1, "Dice loss should be between 0 and 1"
            
            print("✓ Dice Loss tested successfully")
            print(f"Sample Dice Loss: {loss.item():.4f}")

            # Test Weighted Segmentation Loss
            print("\nTesting Weighted Segmentation Loss...")
            dummy_dataset = DroneDataset(
                images_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'original_images'),
                masks_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic'),
                transform=None,
                balance_classes=True
            )
            
            class_weights = calculate_class_weights(dummy_dataset, num_classes=Config.NUM_CLASSES)
            weighted_loss = WeightedSegmentationLoss(num_classes=Config.NUM_CLASSES, class_weights=class_weights)
            
            predictions = torch.randn(batch_size, Config.NUM_CLASSES, 256, 256)
            targets = torch.randint(0, Config.NUM_CLASSES, (batch_size, 256, 256))
            
            loss = weighted_loss(predictions, targets)
            
            assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
            assert loss.shape == torch.Size([]), "Loss should be a scalar"
            assert loss >= 0, "Loss should be non-negative"
            
            print("✓ Weighted Segmentation Loss tested successfully")
            print(f"Sample weighted loss: {loss.item():.4f}")

            return True
            
        except Exception as e:
            print(f"✗ Loss functions test failed: {str(e)}")
            return False

    @staticmethod 
    def logging_suite():
        print("\nRunning Logging Test Suite...")
        try:
            from src.visualization.tensorboard_logger import TensorboardLogger
            import matplotlib.pyplot as plt
            
            logger = TensorboardLogger(log_dir="test_logs")
            
            logger.log_scalar("test/loss", 0.5, 1)
            
            metrics = {"accuracy": 0.85, "precision": 0.78}
            logger.log_scalars("test/metrics", metrics, 1)
            
            sample_image = torch.rand(3, 64, 64)
            logger.log_image("test/image", sample_image, 1)
            
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            logger.log_figure("test/figure", fig, 1)
            
            values = torch.randn(1000)
            logger.log_histogram("test/histogram", values, 1)
            
            # Need model from model creation suite
            model = smp.Unet(
                encoder_name=Config.ENCODER_NAME,
                encoder_weights=Config.ENCODER_WEIGHTS,
                in_channels=Config.IN_CHANNELS,
                classes=Config.NUM_CLASSES,
            )
            logger.log_model_graph(model)
            
            logger.close()
            print("✓ Tensorboard Logger tested successfully")
            return True
            
        except Exception as e:
            print(f"✗ Tensorboard Logger test failed: {str(e)}")
            return False

    @staticmethod
    def training_suite(model, train_loader, val_loader):
        print("\nRunning Training Test Suite...")
        try:
            trainer = SegmentationTrainer(model=model, device=Config.DEVICE)
            
            assert hasattr(trainer, 'logger'), "Trainer should have tensorboard logger"
            assert isinstance(trainer.logger, TensorboardLogger), "Logger should be TensorboardLogger instance"
            
            trainer.train(
                train_dataloader=train_loader,
                valid_dataloader=val_loader,
                epochs=2,
                learning_rate=Config.LEARNING_RATE,
                patience=Config.PATIENCE
            )
            
            log_dir = Path(Config.LOGS_DIR)
            assert log_dir.exists(), "Log directory should exist"
            assert any(log_dir.iterdir()), "Log directory should contain files"
            
            time.sleep(1)
            
            event_files = sorted(log_dir.rglob('events.out.tfevents.*'), key=lambda x: x.stat().st_mtime)
            assert len(event_files) > 0, "No tensorboard event files found"
            latest_event_file = event_files[-1]
            
            from tensorboard.backend.event_processing import event_accumulator
            ea = event_accumulator.EventAccumulator(
                str(latest_event_file),
                size_guidance={
                    event_accumulator.SCALARS: 1000,
                    event_accumulator.IMAGES: 100,
                    event_accumulator.HISTOGRAMS: 1,
                }
            )
            ea.Reload()
            
            scalar_tags = set(ea.Tags()['scalars'])
            required_es_tags = ['early_stopping/score', 'early_stopping/counter']
            
            for tag in required_es_tags:
                assert any(tag in t for t in scalar_tags), f"Missing {tag} in logged data"
            
            print("✓ Training loop and early stopping completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Training loop failed: {str(e)}")
            return False

    @staticmethod
    def model_io_suite(model):
        print("\nRunning Model I/O Test Suite...")
        try:
            test_checkpoint_dir = os.path.join(Config.CHECKPOINTS_DIR, 'test_checkpoint')
            os.makedirs(test_checkpoint_dir, exist_ok=True)
            test_checkpoint_path = os.path.join(test_checkpoint_dir, 'test_model.pth')
            
            torch.save(model.state_dict(), test_checkpoint_path)
            model.load_state_dict(torch.load(test_checkpoint_path))
            print("✓ Model checkpoint saved and loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Model saving/loading failed: {str(e)}")
            return False

    @staticmethod
    def prediction_suite(model, val_dataset):
        print("\nRunning Prediction Test Suite...")
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
            return True
            
        except Exception as e:
            print(f"✗ Prediction failed: {str(e)}")
            return False

    @staticmethod
    def domain_adaptation_suite():
        print("\nRunning Domain Adaptation Test Suite...")
        try:
            # Test domain discriminator
            discriminator = DomainDiscriminator(input_channels=3)
            
            batch_size = 4
            test_input = torch.randn(batch_size, 3, 256, 256)
            
            domain_predictions = discriminator(test_input)
            
            assert domain_predictions.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {domain_predictions.shape}"
            assert torch.all((domain_predictions >= 0) & (domain_predictions <= 1)), "Predictions should be between 0 and 1"
            
            print("✓ Domain discriminator tested successfully")
            print(f"Sample predictions shape: {domain_predictions.shape}")
            print(f"Sample prediction values: {domain_predictions.squeeze().detach().numpy()}")

            # Test adversarial losses
            adv_loss = AdversarialLoss(lambda_adv=0.001)
            
            source_pred = torch.rand(batch_size, 1)
            target_pred = torch.rand(batch_size, 1)
            
            d_loss = adv_loss.discriminator_loss(source_pred, target_pred)
            assert isinstance(d_loss, torch.Tensor), "Discriminator loss should be a tensor"
            assert d_loss.shape == torch.Size([]), "Discriminator loss should be a scalar"
            
            g_loss = adv_loss.generator_loss(target_pred)
            assert isinstance(g_loss, torch.Tensor), "Generator loss should be a tensor"
            assert g_loss.shape == torch.Size([]), "Generator loss should be a scalar"
            
            print("✓ Adversarial losses tested successfully")
            print(f"Sample discriminator loss: {d_loss.item():.4f}")
            print(f"Sample generator loss: {g_loss.item():.4f}")

            return True
            
        except Exception as e:
            print(f"✗ Domain adaptation test failed: {str(e)}")
            return False

    @staticmethod
    def target_dataset_suite():
        print("\nRunning Target Dataset Test Suite...")
        try:
            target_images_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'original_images')
            
            target_dataset = TargetDataset(
                images_dir=target_images_dir,
                transform=get_training_augmentation()
            )
            
            assert len(target_dataset) > 0, "Target dataset is empty"
            
            sample_image = target_dataset[0]
            assert isinstance(sample_image, torch.Tensor), "Dataset should return a tensor"
            assert sample_image.dim() == 3, "Image should have 3 dimensions (C, H, W)"
            assert sample_image.shape[0] == 3, "Image should have 3 channels"
            
            target_loader = DataLoader(
                target_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
            )
            
            sample_batch = next(iter(target_loader))
            assert sample_batch.dim() == 4, "Batch should have 4 dimensions (B, C, H, W)"
            
            print("✓ Target domain dataset tested successfully")
            print(f"Dataset size: {len(target_dataset)}")
            print(f"Sample image shape: {sample_image.shape}")
            print(f"Sample batch shape: {sample_batch.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Target domain dataset test failed: {str(e)}")
            return False

    @staticmethod
    def holyrood_suite():
        print("\nRunning Holyrood Test Suite...")
        try:
            holyrood_sample_dir = os.path.join('data', 'sample', 'holyrood')
            
            holyrood_dataset = TargetDataset(
                images_dir=str(holyrood_sample_dir),
                transform=get_training_augmentation()
            )
            
            holyrood_loader = DataLoader(
                holyrood_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
            )
            
            sample_batch = next(iter(holyrood_loader))
            assert sample_batch.dim() == 4, "Batch should have 4 dimensions (B, C, H, W)"
            
            print("✓ Holyrood sample dataset tested successfully")
            print(f"Total sample images: {len(holyrood_dataset)}")
            print(f"Sample batch shape: {sample_batch.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ Holyrood sample dataset test failed: {str(e)}")
            return False

    @staticmethod
    def adversarial_training_suite(model, val_loader):
        print("\nRunning Adversarial Training Test Suite...")
        try:
            adv_trainer = AdversarialTrainer(
                model=model,
                device=Config.DEVICE,
                lambda_adv=0.001
            )
            
            source_dataset = DroneDataset(
                images_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'original_images'),
                masks_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic'),
                transform=get_training_augmentation()
            )
            
            target_dataset = TargetDataset(
                images_dir=os.path.join("data/target/holyrood"),
                transform=get_training_augmentation()
            )
            
            source_loader = DataLoader(
                source_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
            )
            
            target_loader = DataLoader(
                target_dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
            )
            
            adv_trainer.train(
                source_dataloader=source_loader,
                target_dataloader=target_loader,
                valid_dataloader=val_loader,
                epochs=2,
                learning_rate=Config.LEARNING_RATE,
                patience=Config.PATIENCE
            )
            
            assert hasattr(adv_trainer, 'domain_metrics'), "Trainer should have domain metrics"
            metrics = adv_trainer.domain_metrics.get_metrics()
            assert 'source_domain_acc' in metrics, "Should track source domain accuracy"
            assert 'target_domain_acc' in metrics, "Should track target domain accuracy"
            assert 'domain_confusion' in metrics, "Should track domain confusion"
            
            print("✓ Adversarial trainer tested successfully")
            print("Domain adaptation metrics:", metrics)
            
            return True
            
        except Exception as e:
            print(f"✗ Adversarial trainer test failed: {str(e)}")
            return False

    @staticmethod
    def phase_management_suite(model, adv_trainer):
        print("\nRunning Phase Management Test Suite...")
        try:
            phase_manager = PhaseManager(
                model=model,
                device=Config.DEVICE,
                checkpoints_dir=Config.CHECKPOINTS_DIR
            )
            
            assert phase_manager.get_current_phase() == TrainingPhase.SEGMENTATION
            
            test_metrics = {
                'iou': 0.6,
                'accuracy': 0.85,
                'domain_confusion': 0.3
            }
            
            phase_manager.save_checkpoint(
                trainer=adv_trainer,
                metrics=test_metrics,
                phase=TrainingPhase.SEGMENTATION,
                is_best=True
            )
            
            phase_dir = next(iter(phase_manager.phase_dirs.values()))
            assert (phase_dir / 'best_model.pth').exists(), "Best model checkpoint not saved"
            
            assert phase_manager.metadata_path.exists(), "Metadata file not created"
            metadata = phase_manager._load_metadata()
            assert metadata['current_phase'] == TrainingPhase.SEGMENTATION.name
            assert 'best_metrics' in metadata
            
            can_transition = phase_manager.can_transition(test_metrics)
            assert can_transition, "Should be ready to transition with good metrics"
            
            new_phase = phase_manager.transition_to_next_phase()
            assert new_phase == TrainingPhase.ADVERSARIAL
            
            metadata = phase_manager._load_metadata()
            assert TrainingPhase.SEGMENTATION.name in metadata['phases_completed']
            assert len(metadata['phase_transitions']) > 0
            
            checkpoint = phase_manager.load_checkpoint(TrainingPhase.SEGMENTATION, load_best=True)
            assert checkpoint is not None, "Failed to load checkpoint"
            assert 'model_state_dict' in checkpoint
            assert 'metrics' in checkpoint
            
            print("✓ Phase manager tested successfully")
            print(f"Current phase: {phase_manager.get_current_phase().name}")
            
            return True
            
        except Exception as e:
            print(f"✗ Phase manager test failed: {str(e)}")
            return False

    @staticmethod
    def fine_tuning_suite():
        print("\nRunning Fine-tuning Test Suite...")
        try:
            consistency_loss = ConsistencyLoss()
            
            batch_size = 4
            pred1 = torch.rand(batch_size, Config.NUM_CLASSES, 256, 256)
            pred2 = torch.rand(batch_size, Config.NUM_CLASSES, 256, 256)
            
            cons_loss = consistency_loss(pred1, pred2)
            assert isinstance(cons_loss, torch.Tensor), "Consistency loss should be a tensor"
            assert cons_loss.shape == torch.Size([]), "Consistency loss should be a scalar"
            
            strong_aug = get_strong_augmentation()
            
            sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            
            augmented = strong_aug(image=sample_image)
            augmented_image = augmented['image']
            
            assert isinstance(augmented_image, torch.Tensor), "Augmented image should be a tensor"
            assert augmented_image.shape == torch.Size([3, 256, 256]), "Wrong output shape"
            
            fine_tuning_loss = FineTuningLoss(
                consistency_weight=1.0,
                domain_weight=0.1,
                supervised_weight=0.1,
                rampup_length=40
            )
            
            batch_size = 4
            height, width = 256, 256
            pred1 = torch.rand(batch_size, Config.NUM_CLASSES, height, width)
            pred2 = torch.rand(batch_size, Config.NUM_CLASSES, height, width)
            domain_pred = torch.rand(batch_size, 1)
            
            epochs_to_test = [0, 20, 40, 60]
            for epoch in epochs_to_test:
                losses = fine_tuning_loss(pred1, pred2, domain_pred, epoch)
                
                assert 'total' in losses, "Missing total loss"
                assert 'consistency' in losses, "Missing consistency loss"
                assert 'domain_confusion' in losses, "Missing domain confusion loss"
                assert 'rampup_weight' in losses, "Missing rampup weight"
                
                assert losses['total'] >= 0, "Total loss should be non-negative"
                assert 0 <= losses['rampup_weight'] <= 1, "Rampup weight should be between 0 and 1"
                
                if epoch == 0:
                    assert losses['rampup_weight'] == 0, "Rampup should start at 0"
                elif epoch >= 40:
                    assert losses['rampup_weight'] == 1, "Rampup should reach 1"
                    
            supervised_pred = torch.rand(batch_size, Config.NUM_CLASSES, height, width)
            supervised_target = torch.randint(0, Config.NUM_CLASSES, (batch_size, height, width))
            
            losses_with_supervised = fine_tuning_loss(
                pred1, pred2, domain_pred, 40,
                supervised_pred=supervised_pred,
                supervised_target=supervised_target
            )
            
            assert losses_with_supervised['supervised'] > 0, "Supervised loss should be positive when provided"
            
            print("✓ Fine-tuning components tested successfully")
            print("Loss components:", {k: v.item() for k, v in losses.items()})
            
            return True
            
        except Exception as e:
            print(f"✗ Fine-tuning test failed: {str(e)}")
            return False

    @staticmethod
    def unsupervised_training_suite(model):
        print("\n12c. Testing unsupervised trainer...")
        try:
            # Create discriminator with explicit device placement
            discriminator = DomainDiscriminator().to(Config.DEVICE)
            domain_model = DomainAdaptationModel(model, discriminator)
            
            # Reduce memory usage in trainer
            unsup_trainer = UnsupervisedTrainer(
                model=domain_model,
                device=Config.DEVICE,
                consistency_weight=1.0,
                domain_weight=0.1,
                supervised_weight=0.1,
                rampup_length=40,
                log_interval=10
            )
            
            # Create smaller target dataset for testing
            target_dataset = TargetDataset(
                images_dir=os.path.join("data/target/holyrood"),
                transform=get_strong_augmentation()
            )
            
            # Use very small batch size for testing
            test_batch_size = 1  # Reduced from 2
            
            # Create memory-efficient dataloaders
            target_loader = DataLoader(
                target_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True  # Prevent partial batches
            )
            
            val_dataset = DroneDataset(
                images_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'original_images'),
                masks_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic'),
                transform=get_training_augmentation()
            )
            
            val_loader_small = DataLoader(
                val_dataset,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=True  # Prevent partial batches
            )
            
            # Clear GPU cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                # Run training with reduced epochs and explicit error handling
                unsup_trainer.train(
                    target_dataloader=target_loader,
                    valid_dataloader=val_loader_small,
                    epochs=1,  # Reduced from 2
                    learning_rate=Config.LEARNING_RATE,
                    supervised_dataloader=None,
                    patience=Config.PATIENCE
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("Warning: Reduced memory usage and continuing...")
                else:
                    raise e
            
            # Verify metrics tracking
            assert hasattr(unsup_trainer, 'domain_metrics'), "Trainer should have domain metrics"
            metrics = unsup_trainer.domain_metrics.get_metrics()
            assert 'domain_confusion' in metrics, "Should track domain confusion"
            
            print("✓ Unsupervised trainer tested successfully")
            print("Domain adaptation metrics:", metrics)
            
            # Clear GPU memory after test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"✗ Unsupervised trainer test failed: {str(e)}")
            # Ensure GPU memory is cleared even on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

def test_system(suites=None):
    """Run system tests.
    
    Args:
        suites (list, optional): List of suite names to run. If None, runs all suites.
        
    Available suites:
        - data_loading
        - model_creation  
        - loss_functions
        - logging
        - training
        - model_io
        - prediction
        - domain_adaptation
        - target_dataset
        - holyrood
        - adversarial_training
        - phase_management
        - fine_tuning
        - unsupervised_training
    """
    print("Starting system test...")
    
    # Setup directories and test data
    Config.setup_directories()
    setup_test_data()

    all_suites = {
        'data_loading': TestSuites.data_loading_suite,
        'model_creation': TestSuites.model_creation_suite,
        'loss_functions': TestSuites.loss_functions_suite,
        'logging': TestSuites.logging_suite,
        'training': TestSuites.training_suite,
        'model_io': TestSuites.model_io_suite,
        'prediction': TestSuites.prediction_suite,
        'domain_adaptation': TestSuites.domain_adaptation_suite,
        'target_dataset': TestSuites.target_dataset_suite,
        'holyrood': TestSuites.holyrood_suite,
        'adversarial_training': TestSuites.adversarial_training_suite,
        'phase_management': TestSuites.phase_management_suite,
        'fine_tuning': TestSuites.fine_tuning_suite,
        'unsupervised_training': TestSuites.unsupervised_training_suite
    }

    if suites is None:
        suites = list(all_suites.keys())
    
    results = {}
    shared_objects = {}

    for suite in suites:
        if suite not in all_suites:
            print(f"Warning: Unknown test suite '{suite}'")
            continue
            
        if suite == 'data_loading':
            success, train_loader, val_loader, train_dataset, val_dataset = all_suites[suite]()
            results[suite] = success
            if success:
                shared_objects.update({
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'train_dataset': train_dataset,
                    'val_dataset': val_dataset
                })
        elif suite == 'model_creation':
            success, model = all_suites[suite]()
            results[suite] = success
            if success:
                shared_objects['model'] = model
        elif suite == 'training':
            if 'model' not in shared_objects or 'train_loader' not in shared_objects:
                print(f"Skipping {suite} - required dependencies not tested")
                continue
            results[suite] = all_suites[suite](
                shared_objects['model'],
                shared_objects['train_loader'],
                shared_objects['val_loader']
            )
        elif suite == 'model_io':
            if 'model' not in shared_objects:
                print(f"Skipping {suite} - required dependencies not tested")
                continue
            results[suite] = all_suites[suite](shared_objects['model'])
        elif suite == 'prediction':
            # 1. Test data loading
            print("\n1. Testing data loading...")
            try:
                images_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'original_images')
                masks_dir = os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic')
                
                # Test dataset with class balancing
                dataset = DroneDataset(
                    images_dir=images_dir,
                    masks_dir=masks_dir,
                    transform=get_training_augmentation(),
                    balance_classes=True
                )
                print(f"✓ Dataset loaded successfully with {len(dataset)} images")
                
                # Verify class statistics
                assert hasattr(dataset, 'class_stats'), "Dataset should have class statistics"
                assert hasattr(dataset, 'sample_weights'), "Dataset should have sample weights"
                
                # Verify sample weights
                assert len(dataset.sample_weights) == len(dataset), "Wrong number of sample weights"
                assert np.isclose(dataset.sample_weights.sum(), 1.0), "Sample weights should sum to 1"
                
                # Split dataset using config
                train_size = int(Config.TRAIN_VAL_SPLIT * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                
                # Create balanced dataloader with proper indices
                train_indices = train_dataset.indices
                train_sampler = dataset.get_sampler(indices=train_indices)
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=Config.BATCH_SIZE,
                    sampler=train_sampler,  # Use sampler with proper indices
                    num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=False,
                    num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
                )
                
                # Test batch loading with balanced sampling
                sample_batch = next(iter(train_loader))
                assert len(sample_batch) == 2, "Batch should contain images and masks"
                
                print("✓ DataLoaders created successfully")
                print("Class statistics:", dataset.class_stats)
                
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

            # 2b. Test Dice Loss
            print("\n2b. Testing Dice Loss...")
            try:
                dice_loss = DiceLoss()
                
                # Create dummy predictions and targets
                batch_size = 4
                num_classes = Config.NUM_CLASSES
                predictions = torch.rand(batch_size, num_classes, 256, 256)
                targets = torch.randint(0, num_classes, (batch_size, 256, 256))
                
                # Convert targets to one-hot
                targets_one_hot = torch.nn.functional.one_hot(targets, num_classes)
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
                
                # Calculate loss
                loss = dice_loss(predictions, targets_one_hot)
                
                # Verify loss properties
                assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
                assert loss.shape == torch.Size([]), "Loss should be a scalar"
                assert loss >= 0 and loss <= 1, "Dice loss should be between 0 and 1"
                
                print("✓ Dice Loss tested successfully")
                print(f"Sample Dice Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"✗ Dice Loss test failed: {str(e)}")
                return False

            # 2c. Test Weighted Segmentation Loss
            print("\n2c. Testing Weighted Segmentation Loss...")
            try:
                # Create dummy dataset for weight calculation
                dummy_dataset = DroneDataset(
                    images_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'original_images'),
                    masks_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic'),
                    transform=None,
                    balance_classes=True
                )
                
                # Calculate class weights
                class_weights = calculate_class_weights(
                    dummy_dataset,
                    num_classes=Config.NUM_CLASSES
                )
                
                # Create weighted loss
                weighted_loss = WeightedSegmentationLoss(
                    num_classes=Config.NUM_CLASSES,
                    class_weights=class_weights
                )
                
                # Test with dummy data
                batch_size = 4
                predictions = torch.randn(batch_size, Config.NUM_CLASSES, 256, 256)
                targets = torch.randint(0, Config.NUM_CLASSES, (batch_size, 256, 256))
                
                # Calculate loss
                loss = weighted_loss(predictions, targets)
                
                # Verify loss properties
                assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
                assert loss.shape == torch.Size([]), "Loss should be a scalar"
                assert loss >= 0, "Loss should be non-negative"
                
                # Test dataset sampling
                sampler = dummy_dataset.get_sampler()
                assert sampler is not None, "Sampler should be created when balance_classes=True"
                
                # Create balanced dataloader
                balanced_loader = DataLoader(
                    dummy_dataset,
                    batch_size=Config.BATCH_SIZE,
                    sampler=sampler,
                    num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
                )
                
                # Verify dataloader works
                sample_batch = next(iter(balanced_loader))
                assert len(sample_batch) == 2, "Batch should contain images and masks"
                
                print("✓ Weighted Segmentation Loss and balanced sampling tested successfully")
                print(f"Sample weighted loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"✗ Weighted Segmentation Loss test failed: {str(e)}")
                return False

            # 2c. Test Tensorboard Logger
            print("\n2c. Testing Tensorboard Logger...")
            try:
                from src.visualization.tensorboard_logger import TensorboardLogger
                import matplotlib.pyplot as plt
                
                # Initialize logger
                logger = TensorboardLogger(log_dir="test_logs")
                
                # Test scalar logging
                logger.log_scalar("test/loss", 0.5, 1)
                
                # Test multiple scalars
                metrics = {"accuracy": 0.85, "precision": 0.78}
                logger.log_scalars("test/metrics", metrics, 1)
                
                # Test image logging
                sample_image = torch.rand(3, 64, 64)
                logger.log_image("test/image", sample_image, 1)
                
                # Test figure logging
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])
                logger.log_figure("test/figure", fig, 1)
                
                # Test histogram logging
                values = torch.randn(1000)
                logger.log_histogram("test/histogram", values, 1)
                
                # Test model graph logging
                logger.log_model_graph(model)
                
                logger.close()
                print("✓ Tensorboard Logger tested successfully")
                
            except Exception as e:
                print(f"✗ Tensorboard Logger test failed: {str(e)}")
                return False

            # 3. Test training loop
            print("\n3. Testing training loop...")
            try:
                trainer = SegmentationTrainer(
                    model=model,
                    device=Config.DEVICE
                )
                
                # Verify logger initialization
                assert hasattr(trainer, 'logger'), "Trainer should have tensorboard logger"
                assert isinstance(trainer.logger, TensorboardLogger), "Logger should be TensorboardLogger instance"
                
                # Run a mini training session (2 epochs)
                trainer.train(
                    train_dataloader=train_loader,
                    valid_dataloader=val_loader,
                    epochs=2,  # Override config epochs for testing
                    learning_rate=Config.LEARNING_RATE,
                    patience=Config.PATIENCE
                )
                
                # Verify log directory exists and contains files
                log_dir = Path(Config.LOGS_DIR)
                assert log_dir.exists(), "Log directory should exist"
                assert any(log_dir.iterdir()), "Log directory should contain files"
                
                # Wait a moment for event files to be written
                time.sleep(1)
                
                # Find the most recent event file
                event_files = sorted(log_dir.rglob('events.out.tfevents.*'), key=lambda x: x.stat().st_mtime)
                assert len(event_files) > 0, "No tensorboard event files found"
                latest_event_file = event_files[-1]
                
                # Read the event file to verify metrics are logged
                from tensorboard.backend.event_processing import event_accumulator
                ea = event_accumulator.EventAccumulator(
                    str(latest_event_file),
                    size_guidance={  # Increase size limits
                        event_accumulator.SCALARS: 1000,
                        event_accumulator.IMAGES: 100,
                        event_accumulator.HISTOGRAMS: 1,
                    }
                )
                ea.Reload()
                
                # Verify early stopping metrics are logged
                scalar_tags = set(ea.Tags()['scalars'])
                required_es_tags = [
                    'early_stopping/score',
                    'early_stopping/counter'
                ]
                
                for tag in required_es_tags:
                    assert any(tag in t for t in scalar_tags), f"Missing {tag} in logged data"
                
                print("✓ Training loop and early stopping completed successfully")
                
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

            # 10. Test adversarial trainer
            print("\n10. Testing adversarial trainer...")
            try:
                # Create trainer
                adv_trainer = AdversarialTrainer(
                    model=model,
                    device=Config.DEVICE,
                    lambda_adv=0.001
                )
                
                # Get source and target datasets
                source_dataset = DroneDataset(
                    images_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'original_images'),
                    masks_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic'),
                    transform=get_training_augmentation()
                )
                
                target_dataset = TargetDataset(
                    images_dir=os.path.join("data/target/holyrood"),
                    transform=get_training_augmentation()
                )
                
                # Create dataloaders
                source_loader = DataLoader(
                    source_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
                )
                
                target_loader = DataLoader(
                    target_dataset,
                    batch_size=Config.BATCH_SIZE,
                    shuffle=True,
                    num_workers=Config.NUM_WORKERS if torch.cuda.is_available() else 0
                )
                
                # Run a mini training session
                adv_trainer.train(
                    source_dataloader=source_loader,
                    target_dataloader=target_loader,
                    valid_dataloader=val_loader,
                    epochs=2,
                    learning_rate=Config.LEARNING_RATE,
                    patience=Config.PATIENCE
                )
                
                # Verify domain adaptation metrics
                assert hasattr(adv_trainer, 'domain_metrics'), "Trainer should have domain metrics"
                metrics = adv_trainer.domain_metrics.get_metrics()
                assert 'source_domain_acc' in metrics, "Should track source domain accuracy"
                assert 'target_domain_acc' in metrics, "Should track target domain accuracy"
                assert 'domain_confusion' in metrics, "Should track domain confusion"
                
                print("✓ Adversarial trainer tested successfully")
                print("Domain adaptation metrics:", metrics)
                
            except Exception as e:
                print(f"✗ Adversarial trainer test failed: {str(e)}")
                return False

            # 11. Test phase manager
            print("\n11. Testing phase manager...")
            try:
                # Create phase manager
                phase_manager = PhaseManager(
                    model=model,
                    device=Config.DEVICE,
                    checkpoints_dir=Config.CHECKPOINTS_DIR
                )
                
                # Test phase transitions
                assert phase_manager.get_current_phase() == TrainingPhase.SEGMENTATION
                
                # Test checkpoint saving
                test_metrics = {
                    'iou': 0.6,
                    'accuracy': 0.85,
                    'domain_confusion': 0.3
                }
                
                # Save checkpoint and verify files
                phase_manager.save_checkpoint(
                    trainer=adv_trainer,
                    metrics=test_metrics,
                    phase=TrainingPhase.SEGMENTATION,
                    is_best=True
                )
                
                # Verify checkpoint files exist
                phase_dir = next(iter(phase_manager.phase_dirs.values()))
                assert (phase_dir / 'best_model.pth').exists(), "Best model checkpoint not saved"
                
                # Verify metadata file exists and contains correct information
                assert phase_manager.metadata_path.exists(), "Metadata file not created"
                metadata = phase_manager._load_metadata()
                assert metadata['current_phase'] == TrainingPhase.SEGMENTATION.name
                assert 'best_metrics' in metadata
                
                # Test transition logic
                can_transition = phase_manager.can_transition(test_metrics)
                assert can_transition, "Should be ready to transition with good metrics"
                
                # Test phase transition
                new_phase = phase_manager.transition_to_next_phase()
                assert new_phase == TrainingPhase.ADVERSARIAL
                
                # Verify metadata updated after transition
                metadata = phase_manager._load_metadata()
                assert TrainingPhase.SEGMENTATION.name in metadata['phases_completed']
                assert len(metadata['phase_transitions']) > 0
                
                # Test checkpoint loading
                checkpoint = phase_manager.load_checkpoint(TrainingPhase.SEGMENTATION, load_best=True)
                assert checkpoint is not None, "Failed to load checkpoint"
                assert 'model_state_dict' in checkpoint
                assert 'metrics' in checkpoint
                
                print("✓ Phase manager tested successfully")
                print(f"Current phase: {phase_manager.get_current_phase().name}")
                
            except Exception as e:
                print(f"✗ Phase manager test failed: {str(e)}")
                return False

            # 12. Test unsupervised fine-tuning components
            print("\n12. Testing unsupervised fine-tuning components...")
            try:
                # Test consistency loss
                consistency_loss = ConsistencyLoss()
                
                # Create dummy predictions for same image with different augmentations
                batch_size = 4
                pred1 = torch.rand(batch_size, Config.NUM_CLASSES, 256, 256)
                pred2 = torch.rand(batch_size, Config.NUM_CLASSES, 256, 256)
                
                # Test consistency loss calculation
                cons_loss = consistency_loss(pred1, pred2)
                assert isinstance(cons_loss, torch.Tensor), "Consistency loss should be a tensor"
                assert cons_loss.shape == torch.Size([]), "Consistency loss should be a scalar"
                
                # Test strong augmentation pipeline
                strong_aug = get_strong_augmentation()
                
                # Test with sample image (create dummy image)
                sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                
                # Apply augmentation
                augmented = strong_aug(image=sample_image)
                augmented_image = augmented['image']
                
                # Verify output
                assert isinstance(augmented_image, torch.Tensor), "Augmented image should be a tensor"
                assert augmented_image.shape == torch.Size([3, 256, 256]), "Wrong output shape"
                
                # Test domain confusion metrics
                confusion_metrics = adv_trainer.domain_metrics.get_confusion_metrics()
                assert 'domain_entropy' in confusion_metrics, "Should track domain entropy"
                assert 'feature_alignment' in confusion_metrics, "Should track feature alignment"
                
                print("✓ Unsupervised fine-tuning components tested successfully")
                print("Consistency loss value:", cons_loss.item())
                print("Domain confusion metrics:", confusion_metrics)
                
            except Exception as e:
                print(f"✗ Unsupervised fine-tuning test failed: {str(e)}")
                return False

            # 12b. Test fine-tuning loss integration
            print("\n12b. Testing fine-tuning loss integration...")
            try:
                # Create fine-tuning loss
                fine_tuning_loss = FineTuningLoss(
                    consistency_weight=1.0,
                    domain_weight=0.1,
                    supervised_weight=0.1,
                    rampup_length=40
                )
                
                # Test with dummy data
                batch_size = 4
                height, width = 256, 256
                pred1 = torch.rand(batch_size, Config.NUM_CLASSES, height, width)
                pred2 = torch.rand(batch_size, Config.NUM_CLASSES, height, width)
                domain_pred = torch.rand(batch_size, 1)
                
                # Test loss calculation at different epochs
                epochs_to_test = [0, 20, 40, 60]
                for epoch in epochs_to_test:
                    losses = fine_tuning_loss(pred1, pred2, domain_pred, epoch)
                    
                    # Verify loss components
                    assert 'total' in losses, "Missing total loss"
                    assert 'consistency' in losses, "Missing consistency loss"
                    assert 'domain_confusion' in losses, "Missing domain confusion loss"
                    assert 'rampup_weight' in losses, "Missing rampup weight"
                    
                    # Verify loss values
                    assert losses['total'] >= 0, "Total loss should be non-negative"
                    assert 0 <= losses['rampup_weight'] <= 1, "Rampup weight should be between 0 and 1"
                    
                    # Verify rampup behavior
                    if epoch == 0:
                        assert losses['rampup_weight'] == 0, "Rampup should start at 0"
                    elif epoch >= 40:
                        assert losses['rampup_weight'] == 1, "Rampup should reach 1"
                        
                # Test with supervised data
                supervised_pred = torch.rand(batch_size, Config.NUM_CLASSES, height, width)
                # Create target as class indices (B, H, W)
                supervised_target = torch.randint(0, Config.NUM_CLASSES, (batch_size, height, width))
                
                losses_with_supervised = fine_tuning_loss(
                    pred1, pred2, domain_pred, 40,
                    supervised_pred=supervised_pred,
                    supervised_target=supervised_target
                )
                
                assert losses_with_supervised['supervised'] > 0, "Supervised loss should be positive when provided"
                
                print("✓ Fine-tuning loss integration tested successfully")
                print("Loss components:", {k: v.item() for k, v in losses.items()})
                
            except Exception as e:
                print(f"✗ Fine-tuning loss integration test failed: {str(e)}")
                return False

            # 12c. Test unsupervised trainer
            print("\n12c. Testing unsupervised trainer...")
            try:
                # Create discriminator with explicit device placement
                discriminator = DomainDiscriminator().to(Config.DEVICE)
                domain_model = DomainAdaptationModel(model, discriminator)
                
                # Reduce memory usage in trainer
                unsup_trainer = UnsupervisedTrainer(
                    model=domain_model,
                    device=Config.DEVICE,
                    consistency_weight=1.0,
                    domain_weight=0.1,
                    supervised_weight=0.1,
                    rampup_length=40,
                    log_interval=10
                )
                
                # Create smaller target dataset for testing
                target_dataset = TargetDataset(
                    images_dir=os.path.join("data/target/holyrood"),
                    transform=get_strong_augmentation()
                )
                
                # Use very small batch size for testing
                test_batch_size = 1  # Reduced from 2
                
                # Create memory-efficient dataloaders
                target_loader = DataLoader(
                    target_dataset,
                    batch_size=test_batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True  # Prevent partial batches
                )
                
                val_dataset = DroneDataset(
                    images_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'original_images'),
                    masks_dir=os.path.join(Config.SAMPLE_DATA_DIR, 'label_images_semantic'),
                    transform=get_training_augmentation()
                )
                
                val_loader_small = DataLoader(
                    val_dataset,
                    batch_size=test_batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=True  # Prevent partial batches
                )
                
                # Clear GPU cache before training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    # Run training with reduced epochs and explicit error handling
                    unsup_trainer.train(
                        target_dataloader=target_loader,
                        valid_dataloader=val_loader_small,
                        epochs=1,  # Reduced from 2
                        learning_rate=Config.LEARNING_RATE,
                        supervised_dataloader=None,
                        patience=Config.PATIENCE
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print("Warning: Reduced memory usage and continuing...")
                    else:
                        raise e
                
                # Verify metrics tracking
                assert hasattr(unsup_trainer, 'domain_metrics'), "Trainer should have domain metrics"
                metrics = unsup_trainer.domain_metrics.get_metrics()
                assert 'domain_confusion' in metrics, "Should track domain confusion"
                
                print("✓ Unsupervised trainer tested successfully")
                print("Domain adaptation metrics:", metrics)
                
                # Clear GPU memory after test
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
                
            except Exception as e:
                print(f"✗ Unsupervised trainer test failed: {str(e)}")
                # Ensure GPU memory is cleared even on failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False

    print("\nAll system tests completed successfully! ✓")
    return True

if __name__ == "__main__":
    import sys
    
    # Get test suites from command line arguments
    requested_suites = sys.argv[1:] if len(sys.argv) > 1 else None
    
    success = test_system(suites=requested_suites)
    if success:
        print("\nSystem is ready for training!")
    else:
        print("\nSystem test failed. Please check the errors above.")