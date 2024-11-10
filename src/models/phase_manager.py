import torch
from enum import Enum, auto
from pathlib import Path
import json
import os

class TrainingPhase(Enum):
    """Enum for different training phases"""
    SEGMENTATION = auto()      # Phase 1: Basic semantic segmentation
    ADVERSARIAL = auto()       # Phase 2: Adversarial domain adaptation
    FINE_TUNING = auto()       # Phase 3: Unsupervised fine-tuning

class PhaseManager:
    """
    Manages training phases and transitions for the domain adaptation model.
    """
    def __init__(self, model, device, checkpoints_dir):
        """
        Initialize phase manager.
        
        Args:
            model: Base segmentation model
            device: Device to use for training
            checkpoints_dir: Directory to save phase checkpoints
        """
        self.model = model
        self.device = device
        self.checkpoints_dir = Path(checkpoints_dir)
        self.current_phase = TrainingPhase.SEGMENTATION
        self.phase_metrics = {}
        
        # Create phase-specific checkpoint directories
        self.phase_dirs = {
            TrainingPhase.SEGMENTATION: self.checkpoints_dir / "phase1_segmentation",
            TrainingPhase.ADVERSARIAL: self.checkpoints_dir / "phase2_adversarial",
            TrainingPhase.FINE_TUNING: self.checkpoints_dir / "phase3_finetuning"
        }
        
        for phase_dir in self.phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, trainer, metrics, phase):
        """
        Save checkpoint for current phase.
        
        Args:
            trainer: Current trainer instance
            metrics: Current metrics
            phase: Training phase
        """
        phase_dir = self.phase_dirs[phase]
        
        # Save model state
        model_path = phase_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        
        # Save discriminator if in adversarial phase
        if phase in [TrainingPhase.ADVERSARIAL, TrainingPhase.FINE_TUNING]:
            if hasattr(trainer, 'discriminator'):
                disc_path = phase_dir / "discriminator.pth"
                torch.save(trainer.discriminator.state_dict(), disc_path)
        
        # Save metrics
        metrics_path = phase_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    def load_checkpoint(self, phase):
        """
        Load checkpoint for specified phase.
        
        Args:
            phase: Training phase to load
            
        Returns:
            bool: True if checkpoint was loaded successfully
        """
        phase_dir = self.phase_dirs[phase]
        model_path = phase_dir / "model.pth"
        
        if not model_path.exists():
            return False
            
        self.model.load_state_dict(torch.load(model_path))
        
        # Load metrics if they exist
        metrics_path = phase_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.phase_metrics[phase] = json.load(f)
                
        return True
    
    def can_transition(self, metrics):
        """
        Check if model is ready to transition to next phase.
        
        Args:
            metrics: Current phase metrics
            
        Returns:
            bool: True if ready to transition
        """
        if self.current_phase == TrainingPhase.SEGMENTATION:
            # Transition to adversarial phase if segmentation performance is good enough
            return float(metrics['iou']) > 0.5  # Adjust threshold as needed
            
        elif self.current_phase == TrainingPhase.ADVERSARIAL:
            # Transition to fine-tuning if domain confusion is high enough
            return float(metrics['domain_confusion']) > 0.4  # Adjust threshold as needed
            
        return False
    
    def transition_to_next_phase(self):
        """
        Transition to next training phase.
        
        Returns:
            TrainingPhase: New training phase
        """
        if self.current_phase == TrainingPhase.SEGMENTATION:
            self.current_phase = TrainingPhase.ADVERSARIAL
        elif self.current_phase == TrainingPhase.ADVERSARIAL:
            self.current_phase = TrainingPhase.FINE_TUNING
            
        return self.current_phase
    
    def get_current_phase(self):
        """Get current training phase."""
        return self.current_phase 