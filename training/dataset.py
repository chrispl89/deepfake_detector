"""
Dataset classes for deepfake detection training.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import logging
from typing import List, Tuple, Optional, Callable
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DeepfakeDataset(Dataset):
    """
    Dataset class for deepfake detection.
    Expects directory structure:
        dataset_root/
            real/
                image1.jpg
                image2.jpg
                ...
            fake/
                image1.jpg
                image2.jpg
                ...
    """
    
    def __init__(self, 
                 root_dir: str,
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (299, 299),
                 max_samples: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing 'real' and 'fake' subdirectories
            transform: Optional transform to apply to images
            target_size: Target image size
            max_samples: Maximum number of samples to load (None for all)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        
        self.samples = []
        self.labels = []
        
        self._load_dataset(max_samples)
        
        logger.info(f"Loaded {len(self.samples)} samples from {root_dir}")
        logger.info(f"  Real: {sum(1 for l in self.labels if l == 0)}")
        logger.info(f"  Fake: {sum(1 for l in self.labels if l == 1)}")
    
    def _load_dataset(self, max_samples: Optional[int]):
        """Load dataset file paths and labels."""
        # Load real images (label 0)
        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
            for img_path in real_images:
                self.samples.append(str(img_path))
                self.labels.append(0)
        
        # Load fake images (label 1)
        fake_dir = self.root_dir / 'fake'
        if fake_dir.exists():
            fake_images = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))
            for img_path in fake_images:
                self.samples.append(str(img_path))
                self.labels.append(1)
        
        # Limit samples if specified
        if max_samples is not None and len(self.samples) > max_samples:
            indices = np.random.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image
        img_path = self.samples[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            logger.warning(f"Failed to load image: {img_path}")
            # Return a blank image
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        # Resize
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, self.target_size)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization
            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        label = self.labels[idx]
        
        return image, label


class VideoDeepfakeDataset(Dataset):
    """
    Dataset for video-based deepfake detection.
    Extracts frames from videos on-the-fly.
    """
    
    def __init__(self,
                 metadata_file: str,
                 video_dir: str,
                 frames_per_video: int = 10,
                 target_size: Tuple[int, int] = (299, 299),
                 transform: Optional[Callable] = None):
        """
        Initialize video dataset.
        
        Args:
            metadata_file: JSON file with video labels
            video_dir: Directory containing videos
            frames_per_video: Number of frames to extract per video
            target_size: Target image size
            transform: Optional transform to apply
        """
        self.video_dir = Path(video_dir)
        self.frames_per_video = frames_per_video
        self.target_size = target_size
        self.transform = transform
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.video_files = list(self.metadata.keys())
        
        logger.info(f"Loaded {len(self.video_files)} videos from {metadata_file}")
    
    def __len__(self) -> int:
        """Return number of videos."""
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get frames from a video.
        
        Args:
            idx: Video index
            
        Returns:
            Tuple of (frames_tensor, label)
                frames_tensor shape: (frames_per_video, 3, height, width)
        """
        video_name = self.video_files[idx]
        video_path = self.video_dir / video_name
        label = 1 if self.metadata[video_name]['label'] == 'FAKE' else 0
        
        # Extract frames
        frames = self._extract_frames(str(video_path))
        
        # Convert to tensor
        frames_tensor = torch.stack(frames)
        
        return frames_tensor, label
    
    def _extract_frames(self, video_path: str) -> List[torch.Tensor]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frame indices
        if total_frames < self.frames_per_video:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # Use blank frame if read fails
                frame = np.zeros((*self.target_size, 3), dtype=np.uint8)
            
            # Resize
            frame = cv2.resize(frame, self.target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transform
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = frame.astype(np.float32) / 255.0
                frame = (frame - 0.5) / 0.5
            
            # Convert to tensor
            if not isinstance(frame, torch.Tensor):
                frame = torch.from_numpy(frame).permute(2, 0, 1).float()
            
            frames.append(frame)
        
        cap.release()
        
        # Pad with zeros if we didn't get enough frames
        while len(frames) < self.frames_per_video:
            frames.append(torch.zeros_like(frames[0]))
        
        return frames


def create_dummy_dataset(output_dir: str, num_samples: int = 100):
    """
    Create a dummy dataset for testing.
    
    Args:
        output_dir: Directory to save dummy dataset
        num_samples: Number of samples to create
    """
    output_path = Path(output_dir)
    
    # Create directories
    real_dir = output_path / 'real'
    fake_dir = output_path / 'fake'
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy images
    for i in range(num_samples // 2):
        # Real images (mostly natural patterns)
        img = np.random.randint(100, 200, (299, 299, 3), dtype=np.uint8)
        cv2.imwrite(str(real_dir / f'real_{i:04d}.jpg'), img)
        
        # Fake images (more artificial patterns)
        img = np.random.randint(0, 100, (299, 299, 3), dtype=np.uint8)
        cv2.imwrite(str(fake_dir / f'fake_{i:04d}.jpg'), img)
    
    logger.info(f"Created dummy dataset at {output_dir}")
    logger.info(f"  Real samples: {num_samples // 2}")
    logger.info(f"  Fake samples: {num_samples // 2}")


if __name__ == "__main__":
    # Test dataset creation
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy dataset
    create_dummy_dataset('./data/test_dataset', num_samples=20)
    
    # Test loading
    dataset = DeepfakeDataset('./data/test_dataset')
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"Sample 0:")
        print(f"  Image shape: {img.shape}")
        print(f"  Label: {label}")
