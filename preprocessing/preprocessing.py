"""
Image preprocessing and augmentation for deepfake detection.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def preprocess_face(face_image: np.ndarray, 
                    target_size: Tuple[int, int] = (299, 299),
                    normalize: bool = True) -> np.ndarray:
    """
    Preprocess a face image for model input.
    
    Args:
        face_image: Input face image (BGR format)
        target_size: Target size for the image (default 299x299 for Xception)
        normalize: Whether to normalize pixel values to [-1, 1]
        
    Returns:
        Preprocessed image ready for model input
    """
    if face_image is None or face_image.size == 0:
        raise ValueError("Invalid input image")
    
    # Resize to target size
    if face_image.shape[:2] != target_size:
        face_image = cv2.resize(face_image, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Convert BGR to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Convert to float
    face_image = face_image.astype(np.float32)
    
    if normalize:
        # Normalize to [0, 1] range (MUST match training!)
        face_image = face_image / 255.0
    else:
        # No normalization
        pass
    
    return face_image


def augment_image(image: np.ndarray, 
                  rotation_range: float = 20.0,
                  brightness_range: Tuple[float, float] = (0.8, 1.2),
                  flip_horizontal: bool = True,
                  add_noise: bool = True) -> np.ndarray:
    """
    Apply random augmentation to an image.
    
    Args:
        image: Input image
        rotation_range: Maximum rotation angle in degrees
        brightness_range: Min and max brightness multipliers
        flip_horizontal: Whether to randomly flip horizontally
        add_noise: Whether to add random noise
        
    Returns:
        Augmented image
    """
    augmented = image.copy()
    h, w = augmented.shape[:2]
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), 
                                   borderMode=cv2.BORDER_REFLECT)
    
    # Random brightness
    if brightness_range is not None:
        brightness = np.random.uniform(brightness_range[0], brightness_range[1])
        augmented = np.clip(augmented * brightness, 0, 255).astype(np.uint8)
    
    # Random horizontal flip
    if flip_horizontal and np.random.random() > 0.5:
        augmented = cv2.flip(augmented, 1)
    
    # Random noise
    if add_noise and np.random.random() > 0.5:
        noise = np.random.normal(0, 5, augmented.shape).astype(np.int16)
        augmented = np.clip(augmented.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Random JPEG compression (simulates real-world artifacts)
    if np.random.random() > 0.5:
        quality = np.random.randint(70, 100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', augmented, encode_param)
        augmented = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    return augmented


def create_training_batch(images: list, 
                         labels: list,
                         batch_size: int = 32,
                         target_size: Tuple[int, int] = (299, 299),
                         augment: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a training batch with preprocessing and augmentation.
    
    Args:
        images: List of image paths or numpy arrays
        labels: List of labels (0 for real, 1 for fake)
        batch_size: Number of samples per batch
        target_size: Target image size
        augment: Whether to apply augmentation
        
    Returns:
        Tuple of (batch_images, batch_labels)
    """
    # Select random samples
    indices = np.random.choice(len(images), min(batch_size, len(images)), replace=False)
    
    batch_images = []
    batch_labels = []
    
    for idx in indices:
        # Load image if path is provided
        if isinstance(images[idx], str):
            img = cv2.imread(images[idx])
        else:
            img = images[idx]
        
        if img is None:
            continue
        
        # Apply augmentation
        if augment:
            img = augment_image(img)
        
        # Preprocess
        img = preprocess_face(img, target_size)
        
        batch_images.append(img)
        batch_labels.append(labels[idx])
    
    return np.array(batch_images), np.array(batch_labels)


def extract_frames_from_video(video_path: str, 
                              max_frames: int = 100,
                              sampling_rate: int = 10) -> list:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract
        sampling_rate: Extract every Nth frame
        
    Returns:
        List of frame images
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sampling_rate == 0:
            frames.append(frame)
        
        frame_count += 1
    
    cap.release()
    
    logger.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def compute_mean_std(images: list, target_size: Tuple[int, int] = (299, 299)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation for a dataset.
    
    Args:
        images: List of images or image paths
        target_size: Target image size
        
    Returns:
        Tuple of (mean, std) arrays
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    count = 0
    
    for img in images:
        if isinstance(img, str):
            img = cv2.imread(img)
        
        if img is None:
            continue
        
        img = cv2.resize(img, target_size)
        img = img.astype(np.float64) / 255.0
        
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
        count += img.shape[0] * img.shape[1]
    
    mean = pixel_sum / count
    std = np.sqrt(pixel_sq_sum / count - mean ** 2)
    
    return mean, std


if __name__ == "__main__":
    # Test preprocessing
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    processed = preprocess_face(test_img)
    print(f"Processed shape: {processed.shape}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    augmented = augment_image(test_img)
    print(f"Augmented shape: {augmented.shape}")
