"""
Training module for deepfake detection models.
"""

from .trainer import Trainer
from .dataset import DeepfakeDataset

__all__ = ['Trainer', 'DeepfakeDataset']
