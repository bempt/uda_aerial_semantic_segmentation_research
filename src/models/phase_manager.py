import torch
from enum import Enum, auto
from pathlib import Path
import json
import os
from typing import Dict, Optional, Any
import shutil
import datetime

class TrainingPhase(Enum):
    """Enum for different training phases"""
    SEGMENTATION = auto()      # Phase 1: Basic semantic segmentation
    ADVERSARIAL = auto()       # Phase 2: Adversarial domain adaptation
    FINE_TUNING = auto()       # Phase 3: Unsupervised fine-tuning

class PhaseManager:
    """
    Manages training phases and transitions for the domain adaptation model.
    """
    def __init__(self, model, device, checkpoints_dir: str):
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
        
        # Create phase-specific checkpoint directories with timestamps
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.experiment_dir = self.checkpoints_dir / timestamp
        
        self.phase_dirs = {
            TrainingPhase.SEGMENTATION: self.experiment_dir / "phase1_segmentation",
            TrainingPhase.ADVERSARIAL: self.experiment_dir / "phase2_adversarial",
            TrainingPhase.FINE_TUNING: self.experiment_dir / "phase3_finetuning"
        }
        
        for phase_dir in self.phase_dirs.values():
            phase_dir.mkdir(parents=True, exist_ok=True)
            
        # Create metadata file
        self.metadata_path = self.experiment_dir / "training_metadata.json"
        self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Initialize training metadata file"""
        metadata = {
            "start_time": datetime.datetime.now().isoformat(),
            "phases_completed": [],
            "current_phase": self.current_phase.name,
            "phase_transitions": [],
            "best_metrics": {}
        }
        self._save_metadata(metadata)
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to JSON file"""
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_checkpoint(
        self, 
        trainer, 
        metrics: Dict[str, float], 
        phase: TrainingPhase,
        is_best: bool = False
    ):
        """
        Save checkpoint for current phase.
        
        Args:
            trainer: Current trainer instance
            metrics: Current metrics
            phase: Training phase
            is_best: Whether this is the best checkpoint for this phase
        """
        phase_dir = self.phase_dirs[phase]
        
        # Save model state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
            'phase': phase.name,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Add phase-specific components
        if phase in [TrainingPhase.ADVERSARIAL, TrainingPhase.FINE_TUNING]:
            if hasattr(trainer, 'discriminator'):
                checkpoint['discriminator_state_dict'] = trainer.discriminator.state_dict()
        
        # Save checkpoint
        checkpoint_path = phase_dir / ('best_model.pth' if is_best else 'latest_model.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata['best_metrics'][phase.name] = metrics if is_best else metadata['best_metrics'].get(phase.name, {})
        self._save_metadata(metadata)
            
    def load_checkpoint(
        self, 
        phase: TrainingPhase, 
        load_best: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for specified phase.
        
        Args:
            phase: Training phase to load
            load_best: Whether to load best checkpoint (vs latest)
            
        Returns:
            Dict containing checkpoint data if successful, None otherwise
        """
        phase_dir = self.phase_dirs[phase]
        checkpoint_name = 'best_model.pth' if load_best else 'latest_model.pth'
        checkpoint_path = phase_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            return None
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Update metadata
        metadata = self._load_metadata()
        metadata['last_loaded_checkpoint'] = {
            'phase': phase.name,
            'checkpoint_type': 'best' if load_best else 'latest',
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._save_metadata(metadata)
        
        return checkpoint
    
    def can_transition(self, metrics: Dict[str, float]) -> bool:
        """
        Check if model is ready to transition to next phase.
        
        Args:
            metrics: Current phase metrics
            
        Returns:
            bool: True if ready to transition
        """
        if self.current_phase == TrainingPhase.SEGMENTATION:
            # Transition to adversarial phase if segmentation performance is good enough
            return float(metrics.get('iou', 0)) > 0.5 and float(metrics.get('accuracy', 0)) > 0.75
            
        elif self.current_phase == TrainingPhase.ADVERSARIAL:
            # Transition to fine-tuning if domain confusion is high enough
            return (float(metrics.get('domain_confusion', 0)) > 0.4 and 
                   float(metrics.get('iou', 0)) > 0.45)
            
        return False
    
    def transition_to_next_phase(self) -> TrainingPhase:
        """
        Transition to next training phase.
        
        Returns:
            TrainingPhase: New training phase
        """
        # Update metadata before transition
        metadata = self._load_metadata()
        metadata['phases_completed'].append(self.current_phase.name)
        metadata['phase_transitions'].append({
            'from_phase': self.current_phase.name,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        # Transition to next phase
        if self.current_phase == TrainingPhase.SEGMENTATION:
            self.current_phase = TrainingPhase.ADVERSARIAL
        elif self.current_phase == TrainingPhase.ADVERSARIAL:
            self.current_phase = TrainingPhase.FINE_TUNING
            
        # Update metadata after transition
        metadata['current_phase'] = self.current_phase.name
        metadata['phase_transitions'][-1]['to_phase'] = self.current_phase.name
        self._save_metadata(metadata)
            
        return self.current_phase
    
    def get_current_phase(self) -> TrainingPhase:
        """Get current training phase."""
        return self.current_phase
    
    def get_phase_metrics(self, phase: Optional[TrainingPhase] = None) -> Dict[str, Any]:
        """
        Get metrics for specified phase or current phase.
        
        Args:
            phase: Training phase to get metrics for (default: current phase)
            
        Returns:
            Dict containing phase metrics
        """
        phase = phase or self.current_phase
        metadata = self._load_metadata()
        return metadata['best_metrics'].get(phase.name, {})
    
    def cleanup_old_checkpoints(self, keep_best: bool = True, keep_latest: bool = True):
        """
        Clean up old checkpoints to save disk space.
        
        Args:
            keep_best: Whether to keep best checkpoints
            keep_latest: Whether to keep latest checkpoints
        """
        for phase_dir in self.phase_dirs.values():
            for checkpoint_file in phase_dir.glob("*.pth"):
                if (keep_best and checkpoint_file.name == "best_model.pth" or
                    keep_latest and checkpoint_file.name == "latest_model.pth"):
                    continue
                checkpoint_file.unlink()