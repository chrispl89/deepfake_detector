#!/usr/bin/env python
"""
Extract face crops from frames using MTCNN.

This script ensures preprocessing consistency between training and inference.

Usage:
    python extract_faces_from_frames.py --input ./data/datasets/faceforensics_frames --output ./data/datasets/faceforensics_faces
"""

import argparse
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.face_detection import FaceDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_face_from_image(image_path: Path, face_detector, margin: float = 0.2):
    """
    Extract face crop from an image using MTCNN.
    
    Args:
        image_path: Path to image file
        face_detector: MTCNN face detector instance
        margin: Margin to add around detected face (default 20%)
        
    Returns:
        Face crop as numpy array, or None if no face detected
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Failed to read image: {image_path}")
        return None
    
    # Detect faces
    faces = face_detector.detect_faces(image)
    
    if not faces:
        logger.debug(f"No face detected in {image_path.name}")
        return None
    
    # Use the first (largest) face
    face = faces[0]
    x, y, w, h = face['bbox']
    
    # Add margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(image.shape[1], x + w + margin_w)
    y2 = min(image.shape[0], y + h + margin_h)
    
    # Crop face
    face_crop = image[y1:y2, x1:x2]
    
    return face_crop


def main():
    parser = argparse.ArgumentParser(description='Extract face crops from frames using MTCNN')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing frames (with real/fake subdirs)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for face crops')
    parser.add_argument('--margin', type=float, default=0.2,
                       help='Margin around detected face (default 0.2 = 20%)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for MTCNN')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Initialize face detector
    logger.info("Initializing face detector (MTCNN with OpenCV fallback)...")
    try:
        face_detector = FaceDetector(use_mtcnn=True)
    except Exception as e:
        logger.error(f"Failed to initialize face detector: {e}")
        logger.info("Using OpenCV Haar Cascades fallback...")
        face_detector = FaceDetector(use_mtcnn=False)
    
    # Process each label directory (real/fake)
    total_faces_extracted = 0
    total_frames_processed = 0
    
    for label in ['real', 'fake']:
        label_input_dir = input_dir / label
        label_output_dir = output_dir / label
        
        if not label_input_dir.exists():
            logger.warning(f"Label directory not found: {label_input_dir}")
            continue
        
        label_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = list(label_input_dir.glob('*.jpg')) + list(label_input_dir.glob('*.png'))
        
        if not image_files:
            logger.warning(f"No images found in {label_input_dir}")
            continue
        
        logger.info(f"\nProcessing {len(image_files)} {label.upper()} frames...")
        
        faces_extracted = 0
        
        for image_path in tqdm(image_files, desc=f"Extracting {label} faces"):
            total_frames_processed += 1
            
            # Extract face
            face_crop = extract_face_from_image(image_path, face_detector, args.margin)
            
            if face_crop is not None:
                # Save face crop
                output_path = label_output_dir / f"{image_path.stem}_face.jpg"
                cv2.imwrite(str(output_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                faces_extracted += 1
                total_faces_extracted += 1
        
        logger.info(f"{label.upper()}: Extracted {faces_extracted} faces from {len(image_files)} frames")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"FACE EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total frames processed: {total_frames_processed}")
    logger.info(f"Total faces extracted: {total_faces_extracted}")
    logger.info(f"Success rate: {total_faces_extracted/total_frames_processed*100:.1f}%")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")
    
    # Count faces per label
    for label in ['real', 'fake']:
        label_dir = output_dir / label
        if label_dir.exists():
            face_count = len(list(label_dir.glob('*.jpg')))
            logger.info(f"{label.upper()} faces: {face_count}")
    
    logger.info(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
