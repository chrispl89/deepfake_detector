#!/usr/bin/env python
"""
Download FaceForensics++ dataset for training.

This script downloads a subset of FaceForensics++ videos:
- 50 real videos (original YouTube videos)
- 50 fake videos (Deepfakes manipulation)
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_dataset():
    """Download FaceForensics++ dataset."""
    
    output_path = Path('./data/datasets/faceforensics_videos')
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("DOWNLOADING FACEFORENSICS++ DATASET")
    logger.info("="*60)
    
    # Download original (real) videos
    logger.info("\n1. Downloading REAL videos (original YouTube)...")
    cmd_real = [
        'python', 'datasets/download_faceforensics.py',
        str(output_path),
        '--dataset', 'original',
        '--compression', 'c40',
        '--type', 'videos',
        '--num_videos', '50',
        '--server', 'EU'
    ]
    
    logger.info(f"Command: {' '.join(cmd_real)}")
    
    try:
        result = subprocess.run(cmd_real, check=True, capture_output=False)
        logger.info("✓ Real videos downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download real videos: {e}")
        logger.info("Continuing with fake videos...")
    except FileNotFoundError:
        logger.error("download_faceforensics.py not found or Python not in PATH")
        return 1
    
    # Download fake videos (Deepfakes)
    logger.info("\n2. Downloading FAKE videos (Deepfakes manipulation)...")
    cmd_fake = [
        'python', 'datasets/download_faceforensics.py',
        str(output_path),
        '--dataset', 'Deepfakes',
        '--compression', 'c40',
        '--type', 'videos',
        '--num_videos', '50',
        '--server', 'EU'
    ]
    
    logger.info(f"Command: {' '.join(cmd_fake)}")
    
    try:
        result = subprocess.run(cmd_fake, check=True, capture_output=False)
        logger.info("✓ Fake videos downloaded successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download fake videos: {e}")
        return 1
    except FileNotFoundError:
        logger.error("download_faceforensics.py not found or Python not in PATH")
        return 1
    
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*60)
    logger.info(f"Videos saved to: {output_path.absolute()}")
    logger.info("\nNext steps:")
    logger.info("1. Extract frames: python extract_frames.py")
    logger.info("2. Extract faces: python extract_faces_from_frames.py")
    logger.info("3. Train model: python train.py")
    logger.info("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(download_dataset())
