#!/usr/bin/env python
"""
Test the trained deepfake detection model.

Usage:
    python test_trained_model.py --model ./checkpoints/model.pth --test-dir ./data/test_images
"""

import argparse
import sys
import logging
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detector import DeepfakeDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_on_images(detector, test_dir):
    """Test detector on all images in a directory."""
    test_path = Path(test_dir)
    
    if not test_path.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return
    
    # Find all image files
    image_files = (
        list(test_path.glob('*.jpg')) +
        list(test_path.glob('*.png')) +
        list(test_path.glob('*.jpeg'))
    )
    
    if not image_files:
        logger.warning(f"No images found in {test_dir}")
        return
    
    logger.info(f"Testing on {len(image_files)} images...")
    logger.info("="*60)
    
    results = []
    
    for img_path in image_files:
        logger.info(f"\nTesting: {img_path.name}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Failed to read: {img_path}")
            continue
        
        # Run inference
        faces = detector.infer_on_frame(image)
        
        if not faces:
            logger.info("  No faces detected")
            continue
        
        # Display results
        for i, face in enumerate(faces):
            logger.info(f"  Face {i+1}:")
            logger.info(f"    Score: {face.deepfake_score:.4f}")
            logger.info(f"    Verdict: {'FAKE' if face.is_fake else 'REAL'}")
            logger.info(f"    Confidence: {face.confidence:.4f}")
            
            results.append({
                'image': img_path.name,
                'face_id': i,
                'score': face.deepfake_score,
                'is_fake': face.is_fake,
                'confidence': face.confidence
            })
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total images tested: {len(image_files)}")
    logger.info(f"Total faces detected: {len(results)}")
    
    if results:
        fake_count = sum(1 for r in results if r['is_fake'])
        real_count = len(results) - fake_count
        avg_score = sum(r['score'] for r in results) / len(results)
        
        logger.info(f"Detected as FAKE: {fake_count}")
        logger.info(f"Detected as REAL: {real_count}")
        logger.info(f"Average score: {avg_score:.4f}")
    
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test trained deepfake detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--test-dir', type=str, default='./data/test_images',
                       help='Directory containing test images')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    logger.info("="*60)
    logger.info("DEEPFAKE DETECTOR - MODEL TESTING")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Test directory: {args.test_dir}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Device: {args.device}")
    logger.info("="*60)
    
    # Initialize detector
    try:
        logger.info("\nLoading model...")
        detector = DeepfakeDetector(
            model_path=args.model,
            device=args.device,
            threshold=args.threshold
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        return 1
    
    # Run tests
    test_on_images(detector, args.test_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
