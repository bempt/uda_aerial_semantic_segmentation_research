from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as T

class TensorboardLogger:
    """Handles all Tensorboard logging operations"""
    
    def __init__(self, log_dir="logs"):
        """
        Initialize Tensorboard logger
        
        Args:
            log_dir: Base directory for logs
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(log_dir) / timestamp
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
    def log_scalar(self, tag, value, step):
        """Log a scalar value"""
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """Log multiple scalars under the same main tag"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
    def log_image(self, tag, image, step):
        """
        Log an image (handles both torch.Tensor and numpy.ndarray)
        
        Args:
            tag: Image identifier
            image: Image to log (torch.Tensor or numpy.ndarray)
            step: Training step
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach()
            if image.dim() == 4:  # Handle batched images
                image = image[0]  # Take first image from batch
            if image.dim() == 2:  # Handle grayscale
                image = image.unsqueeze(0)
            if image.dim() == 3:
                if image.dtype == torch.int64 or image.dtype == torch.long:
                    image = image.float()
                if image.size(0) == 1:  # Grayscale
                    image = image.repeat(3, 1, 1)
                image = T.ToPILImage()(image)
        elif isinstance(image, np.ndarray):
            if image.dtype == np.int64:
                image = image.astype(np.float32)
            if image.ndim == 2:  # Handle grayscale
                image = np.stack([image] * 3, axis=2)
            image = Image.fromarray(image)
            
        self.writer.add_image(tag, T.ToTensor()(image), step)
        
    def log_figure(self, tag, figure, step):
        """Log a matplotlib figure"""
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = T.ToTensor()(image)
        self.writer.add_image(tag, image_tensor, step)
        plt.close(figure)
        
    def log_histogram(self, tag, values, step, bins='auto'):
        """Log value distributions as histogram"""
        if isinstance(values, torch.Tensor):
            values = values.cpu().detach().numpy()
        self.writer.add_histogram(tag, values, step, bins=bins)
        
    def log_model_graph(self, model, input_shape=(1, 3, 256, 256)):
        """Log model architecture graph"""
        device = next(model.parameters()).device
        dummy_input = torch.rand(input_shape).to(device)
        self.writer.add_graph(model, dummy_input)
        
    def close(self):
        """Close the Tensorboard writer"""
        self.writer.close() 